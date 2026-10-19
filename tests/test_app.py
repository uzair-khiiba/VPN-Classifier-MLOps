import pytest
from fastapi.testclient import TestClient

from src import app as app_module
from src.features import FEATURES

client = TestClient(app_module.app)


def valid_payload() -> dict:
    payload = {f: 1.0 for f in FEATURES}
    payload.update(n_signals=12, max_signals_per_hour=3, asn_unique_ips=40, share_http_scan=0.5)
    return payload


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
