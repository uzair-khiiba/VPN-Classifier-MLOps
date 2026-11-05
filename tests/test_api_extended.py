"""API behaviour the dashboard relies on: labels, safe errors, documentation and model info."""

import json

import pytest
from fastapi.testclient import TestClient

from src import app as app_module
from src.features import FEATURES

client = TestClient(app_module.app)


@pytest.fixture
def loaded(model_path):
    app_module.load_model(model_path)
    yield
    app_module.STATE.update(bundle=None, error=None)


@pytest.fixture
def no_model(tmp_path):
    app_module.load_model(tmp_path / "secret_folder" / "missing.joblib")
    yield tmp_path
    app_module.STATE.update(bundle=None, error=None)


def payload() -> dict:
    return {**{f: 1.0 for f in FEATURES}, "n_signals": 12, "asn_unique_ips": 40, "share_http_scan": 0.5}


def test_predict_returns_class_and_readable_label(loaded):
    body = client.post("/predict", json=payload()).json()
    assert body["prediction"] == int(body["is_vpn_proxy"])
    assert body["label"] == ("VPN/Proxy" if body["is_vpn_proxy"] else "Not VPN/Proxy")


def test_original_response_fields_are_kept(loaded):
    body = client.post("/predict", json=payload()).json()
    assert {"is_vpn_proxy", "probability", "threshold", "model_version"} <= body.keys()


def test_health_reports_model_status(loaded):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "model_status": "loaded",
                                            "model": "test-logreg", "version": "test"}


def test_missing_model_never_exposes_the_file_path(no_model, caplog):
    responses = [client.get("/health"), client.post("/predict", json=payload()), client.get("/model-info"),
                 client.post("/explain", json=payload())]
    assert client.get("/health").json()["model_status"] == "failed"
    assert [r.status_code for r in responses] == [200, 503, 503, 503]
    for response in responses:
        assert "secret_folder" not in response.text and str(no_model) not in response.text


def test_corrupt_model_is_handled_without_exposing_the_path(tmp_path):
    bad = tmp_path / "secret_folder.joblib"
    bad.write_bytes(b"not a model")
    app_module.load_model(bad)
    response = client.post("/predict", json=payload())
    assert response.status_code == 503 and "secret_folder" not in response.text
    app_module.STATE.update(bundle=None, error=None)


def test_explain_says_whether_a_feature_flips_the_class(loaded):
    body = client.post(f"/explain?top_k={len(FEATURES)}", json=payload()).json()
    assert len(body["contributions"]) == len(FEATURES)
    for item in body["contributions"]:
        flipped = (item["probability_at_reference"] >= body["threshold"]) != body["is_vpn_proxy"]
        # Scores are rounded for display, so only check values clearly away from the threshold.
        if abs(item["probability_at_reference"] - body["threshold"]) > 0.001:
            assert item["changes_prediction"] == flipped


def test_explain_rejects_invalid_top_k(loaded):
    assert client.post("/explain?top_k=0", json=payload()).status_code == 422


def test_model_info_reports_the_served_model(loaded):
    body = client.get("/model-info").json()
    assert body["model"] == "test-logreg" and body["n_features"] == len(FEATURES)
    assert body["metrics"] == {} and 0 < body["threshold"] < 1


def test_infinite_values_are_rejected(loaded):
    body = json.dumps({**payload(), "n_signals": "__X__"}).replace('"__X__"', "1e999")
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "n_signals"]


def test_model_is_loaded_once_not_per_request(model_path, monkeypatch):
    calls = []
    original = app_module.load_bundle
    monkeypatch.setattr(app_module, "load_bundle", lambda path: calls.append(path) or original(model_path))
    with TestClient(app_module.app) as started:
        for _ in range(3):
            assert started.post("/predict", json=payload()).status_code == 200
    assert len(calls) == 1
    app_module.STATE.update(bundle=None, error=None)


def test_openapi_documents_fields_and_errors():
    spec = client.get("/openapi.json").json()
    fields = spec["components"]["schemas"]["IPFeatures"]["properties"]
    assert list(fields) == FEATURES
    assert not [name for name, field in fields.items() if "description" not in field]
    for path in ("/predict", "/explain"):
        assert {"200", "422", "503"} <= spec["paths"][path]["post"]["responses"].keys()
    assert "503" in spec["paths"]["/model-info"]["get"]["responses"]
