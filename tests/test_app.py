import pytest
from fastapi.testclient import TestClient

from src import app as app_module
from src.features import FEATURES
from tests.conftest import consistent_features

client = TestClient(app_module.app)


def valid_payload() -> dict:
    return consistent_features()


@pytest.fixture
def loaded(model_path):
    app_module.load_model(model_path)
    yield
    app_module.STATE.update(bundle=None, error=None)


def test_schema_matches_model_features():
    assert list(app_module.IPFeatures.model_fields) == FEATURES


def test_health_with_model(loaded):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_health_and_predict_without_model(tmp_path):
    app_module.load_model(tmp_path / "missing.joblib")
    assert client.get("/health").json()["status"] == "degraded"
    assert client.post("/predict", json=valid_payload()).status_code == 503


def test_predict_returns_probability_and_label(loaded):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["is_vpn_proxy"] == (body["probability"] >= body["threshold"])
    assert body["model_version"] == "test-logreg-test"


@pytest.mark.parametrize("change", [
    {"n_signals": -5},          # negative count
    {"share_http_scan": 1.7},   # share above 1
    {"has_shodan": 3},          # flag not 0/1
    {"unknown_field": 1},       # extra field
])
def test_predict_rejects_invalid_input(loaded, change):
    assert client.post("/predict", json={**valid_payload(), **change}).status_code == 422


def test_predict_rejects_missing_field(loaded):
    payload = valid_payload()
    del payload["n_watchers"]
    assert client.post("/predict", json=payload).status_code == 422


def test_explain_returns_sorted_contributions(loaded):
    response = client.post("/explain?top_k=5", json=valid_payload())
    assert response.status_code == 200
    contributions = response.json()["contributions"]
    assert len(contributions) == 5
    effects = [abs(c["effect"]) for c in contributions]
    assert effects == sorted(effects, reverse=True)
    assert contributions[0]["feature"] == "asn_unique_ips"  # the only informative feature in the fixture


def test_docs_available():
    assert client.get("/docs").status_code == 200


def test_docs_examples_are_valid_requests(loaded):
    for example in app_module.EXAMPLES:
        assert client.post("/predict", json=example).status_code == 200


@pytest.mark.parametrize("change", [
    {"n_signals": 12.5},                                    # a count must be whole
    {"has_shodan": 0.5},                                    # a flag must be 0 or 1
    {"has_shodan": 0},                                      # ...and match the ports
    {"n_tcp_ports": 0, "jarm_shared": 0},                   # has_shodan = 1 without ports
    {"no_print_with_ports": 1, "has_shodan": 0, "n_tcp_ports": 0},
    {"max_signals_per_hour": 13},                           # busiest hour above the total
    {"n_active_days": 13, "signals_per_day": 12 / 13},      # more active days than signals
    {"share_http_spam": 0.5},                               # attack mix adds up to 1.25
    {"signals_per_day": 9},                                 # not n_signals / n_active_days
    {"mean_gap_s": 2 * 3600 + 60},                          # a gap longer than the active span
])
def test_predict_rejects_impossible_combinations(loaded, change):
    response = client.post("/predict", json={**valid_payload(), **change})
    assert response.status_code == 422, change


def test_every_real_ip_passes_the_input_rules():
    """The rules must never reject what the training pipeline produces: check every committed test IP."""
    import pandas as pd

    from src.config import IP_LOOKUP_PATH

    for row in pd.read_parquet(IP_LOOKUP_PATH)[FEATURES].to_dict("records"):
        app_module.IPFeatures(**row)


def test_api_orders_features_by_name_and_uses_the_bundle_threshold(loaded):
    """The dashboard sends a JSON object, so the order of its keys must not change the answer."""
    ordered = valid_payload()
    reversed_keys = dict(reversed(list(ordered.items())))
    first, second = client.post("/predict", json=ordered).json(), client.post("/predict", json=reversed_keys).json()
    assert first == second
    bundle = app_module.STATE["bundle"]
    assert first["threshold"] == round(float(bundle["threshold"]), 4)
    info = client.get("/model-info").json()
    assert info["features"] == bundle["features"] == FEATURES


def test_schema_explains_each_feature(loaded):
    fields = client.get("/openapi.json").json()["components"]["schemas"]["IPFeatures"]["properties"]
    assert list(fields) == FEATURES
    assert all(fields[name]["kind"] and fields[name]["unit"] and fields[name]["meaning"] for name in FEATURES)
    assert fields["has_shodan"]["kind"] == "binary" and fields["share_tcp_scan"]["kind"] == "fraction"
