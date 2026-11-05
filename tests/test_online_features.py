"""Features built for one IP at request time must equal the training pipeline's features."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import app as app_module
from src.config import FEATURE_REFERENCE_PATH, IP_LOOKUP_PATH, ROOT
from src.features import FEATURES, build_features, parse_shodan_info
from src.online_features import SIGNAL_COLUMNS, build_ip_lookup, build_reference, features_from_signals
from tests.conftest import consistent_features, make_shodan, make_signals

client = TestClient(app_module.app)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    signals = make_signals(n_ips=30, seed=3)
    shodan = make_shodan(sorted(signals["attacker_ip_enum"].unique()))
    path = tmp_path_factory.mktemp("raw") / "train.parq"
    signals.to_parquet(path, index=False)
    return signals, shodan, build_features(path, shodan).set_index("attacker_ip_enum")


@pytest.mark.parametrize("ip", [0, 1, 2, 3, 7, 29])  # 0-2 have Shodan records, the rest have none
def test_features_from_signals_match_the_training_pipeline(data, ip):
    """Leave the IP out of the reference, as if it were new: the result must equal training exactly."""
    signals, shodan, expected = data
    reference = build_reference(signals[signals["attacker_ip_enum"] != ip], shodan[shodan["attacker_ip_enum"] != ip])
    cell = shodan.loc[shodan["attacker_ip_enum"] == ip, "shodan_info"].iloc[0]
    got = features_from_signals(signals[signals["attacker_ip_enum"] == ip], parse_shodan_info(cell), reference)
    assert list(got) == FEATURES
    np.testing.assert_allclose([got[f] for f in FEATURES], expected.loc[ip, FEATURES].to_numpy(dtype=float))


def test_unknown_network_and_fingerprints_count_only_the_ip_itself(data):
    signals, shodan, _ = data
    reference = build_reference(signals, shodan)
    one = signals[signals["attacker_ip_enum"] == 0].assign(attacker_as_num=999_999.0)
    got = features_from_signals(one, {"443/tcp": {"jarm": "never-seen", "ja3s": None, "headers_hash": 5}}, reference)
    assert got["asn_unique_ips"] == 1 and got["jarm_shared"] == 0 and got["headers_hash_shared"] == 0
    assert got["has_shodan"] == 1 and got["n_tcp_ports"] == 1 and got["no_print_with_ports"] == 0


def test_no_signals_is_an_error(data):
    with pytest.raises(ValueError):
        features_from_signals(pd.DataFrame(columns=SIGNAL_COLUMNS), {}, build_reference(data[0], None))


# ---- API endpoints ---------------------------------------------------------------------------


@pytest.fixture
def served(model_path, data, tmp_path):
    signals, shodan, features = data
    lookup = build_ip_lookup(features.reset_index())
    lookup.to_parquet(tmp_path / "lookup.parquet", index=False)
    (tmp_path / "reference.json").write_text(json.dumps(build_reference(signals, shodan)))
    app_module.load_model(model_path)
    app_module.load_ip_data(tmp_path / "lookup.parquet", tmp_path / "reference.json")
    yield lookup
    app_module.STATE.update(bundle=None, error=None, lookup=None, reference=None)


def signals_body(signals: pd.DataFrame) -> dict:
    rows = signals[SIGNAL_COLUMNS].assign(attack_time=signals["attack_time"].dt.strftime("%Y-%m-%dT%H:%M:%S"))
    return {"signals": json.loads(rows.to_json(orient="records")),
            "shodan_ports": {"443/tcp": {"jarm": "J1", "ja3s": "A1", "headers_hash": 11}}}


def test_lookup_returns_stored_features_and_true_label(served):
    row = served.iloc[4]
    body = client.get(f"/ips/{int(row['attacker_ip_enum'])}").json()
    assert body["features"] == {f: float(row[f]) for f in FEATURES} and body["actual_label"] == int(row["label"])


def test_predict_by_ip_equals_predict_with_the_same_features(served):
    ip = int(served["attacker_ip_enum"].iloc[0])
    body = client.get(f"/predict/ip/{ip}").json()
    direct = client.post("/predict", json=body["features"]).json()
    assert body["probability"] == direct["probability"] and body["ip_id"] == ip
    assert body["correct"] == (body["prediction"] == body["actual_label"])


def test_unknown_ip_is_404_and_points_to_the_other_endpoints(served):
    response = client.get("/predict/ip/123456")
    assert response.status_code == 404 and "/predict/signals" in response.json()["detail"]
    assert client.get("/ips/-3").status_code == 422


def test_predict_signals_uses_the_computed_features(served, data):
    signals = data[0]
    body = client.post("/predict/signals", json=signals_body(signals[signals["attacker_ip_enum"] == 5])).json()
    assert body["source"] == "computed from signals"
    assert body["features"]["n_signals"] == (signals["attacker_ip_enum"] == 5).sum()
    assert body["probability"] == client.post("/predict", json=body["features"]).json()["probability"]


def test_features_from_signals_endpoint_matches_the_library(served, data):
    signals, shodan, _ = data
    one = signals[signals["attacker_ip_enum"] == 6]
    body = signals_body(one)
    expected = features_from_signals(one, body["shodan_ports"], build_reference(signals, shodan))
    assert client.post("/features/from-signals", json=body).json() == pytest.approx(expected)


@pytest.mark.parametrize("change", [
    {"signals": []},                                                      # no signals
    {"shodan_ports": {"443": {}}},                                         # port without protocol
    {"signals": [{"attack_time": "not a time", "attack_type": "x", "watcher_uuid_enum": 1}]},
    {"signals": [{"attack_type": "http:scan", "watcher_uuid_enum": 1}]},   # time missing
    {"extra": 1},
])
def test_invalid_signal_requests_are_rejected(served, data, change):
    body = {**signals_body(data[0].head(3)), **change}
    assert client.post("/predict/signals", json=body).status_code == 422


def test_missing_lookup_data_only_disables_its_endpoints(model_path, tmp_path):
    app_module.load_model(model_path)
    app_module.load_ip_data(tmp_path / "none.parquet", tmp_path / "none.json")
    try:
        one = {"signals": [{"attack_time": "2023-09-01T10:00:00Z", "attack_type": "http:scan", "watcher_uuid_enum": 1}]}
        assert client.get("/predict/ip/1").status_code == 503
        assert client.post("/predict/signals", json=one).status_code == 503
        assert client.post("/predict", json=consistent_features()).status_code == 200
        health = client.get("/health").json()
        assert health["status"] == "ok" and not health["ip_lookup_loaded"] and not health["feature_reference_loaded"]
    finally:
        app_module.STATE.update(bundle=None, error=None, lookup=None, reference=None)


def test_committed_lookup_data_loads_and_matches_the_model():
    """The committed files let a fresh clone look up test IPs and score signals without the dataset."""
    app_module.load_ip_data(IP_LOOKUP_PATH, FEATURE_REFERENCE_PATH)
    try:
        assert app_module.STATE["lookup"] is not None and app_module.STATE["reference"] is not None
        assert len(app_module.STATE["lookup"]) > 20_000
        assert app_module.STATE["lookup"].index.is_unique
    finally:
        app_module.STATE.update(lookup=None, reference=None)


def test_signals_example_is_a_valid_request():
    example = json.loads((ROOT / "examples" / "signals_request.json").read_text())
    app_module.SignalsRequest(**example)


def _one(attack_type="http:scan", **ports) -> dict:
    return {"signals": [{"attack_time": "2023-09-01T10:00:00Z", "attack_type": attack_type, "attacker_as_num": 105,
                         "watcher_uuid_enum": "w1"}], "shodan_ports": ports}


@pytest.mark.parametrize("written", ["HTTP:SCAN", " http:scan ", "Http:Scan"])
def test_attack_types_ignore_case_and_spaces(served, written):
    body = client.post("/features/from-signals", json=_one(written)).json()
    assert body["share_http_scan"] == 1.0 and body["n_attack_types"] == 1


def test_header_hash_as_text_matches_the_integer(served):
    """The UI sends every port value as text: "0" must mean "no hash", like the integer 0 in training."""
    as_int = client.post("/features/from-signals", json=_one(**{"80/tcp": {"headers_hash": 0}})).json()
    as_text = client.post("/features/from-signals", json=_one(**{"80/tcp": {"headers_hash": "0"}})).json()
    assert as_int == as_text and as_text["no_print_with_ports"] == 1
    known_int = client.post("/features/from-signals", json=_one(**{"443/tcp": {"headers_hash": 11}})).json()
    known_text = client.post("/features/from-signals", json=_one(**{"443/tcp": {"headers_hash": " 11 "}})).json()
    assert known_int == known_text and known_text["headers_hash_shared"] > 0


def test_fingerprints_ignore_case(served):
    lower = client.post("/features/from-signals", json=_one(**{"443/tcp": {"jarm": "j1"}})).json()
    upper = client.post("/features/from-signals", json=_one(**{"443/tcp": {"jarm": " J1 "}})).json()
    assert lower == upper
