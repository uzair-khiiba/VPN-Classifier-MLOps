"""Request monitoring: GET /metrics counts requests and latency per route, and predictions per class."""

import pytest
from fastapi.testclient import TestClient

from src import app as app_module
from src.monitoring import METRICS, RequestMetrics, _percentile
from tests.conftest import consistent_features

client = TestClient(app_module.app)


@pytest.fixture
def loaded(model_path):
    app_module.load_model(model_path)
    METRICS.reset()
    yield
    app_module.STATE.update(bundle=None, error=None)


def test_metrics_starts_empty():
    METRICS.reset()
    body = client.get("/metrics").json()
    # The /metrics call itself is recorded only after its response is built.
    assert body["total_requests"] == 0 and body["avg_latency_ms"] == 0.0
    assert body["predictions"] == {"VPN/Proxy": 0, "Not VPN/Proxy": 0}
    assert "started_at" in body and body["uptime_s"] >= 0


def test_requests_are_counted_per_route_with_latency(loaded):
    for _ in range(3):
        assert client.post("/predict", json=consistent_features()).status_code == 200
    client.get("/health")
    body = client.get("/metrics").json()
    predict = body["endpoints"]["POST /predict"]
    assert predict["count"] == 3 and predict["errors"] == 0
    assert 0 < predict["avg_latency_ms"] <= predict["max_latency_ms"]
    assert body["endpoints"]["GET /health"]["count"] == 1
    assert body["total_requests"] == 4
    assert sum(body["predictions"].values()) == 3


def test_path_parameters_are_grouped_and_unknown_paths_collapsed(loaded):
    client.get("/ips/1")
    client.get("/ips/2")
    client.get("/wp-login.php")
    client.get("/.env")
    endpoints = client.get("/metrics").json()["endpoints"]
    assert endpoints["GET /ips/{ip_id}"]["count"] == 2
    assert endpoints["GET <unmatched>"]["count"] == 2
    assert not any("wp-login" in key for key in endpoints)


def test_server_errors_are_counted(tmp_path):
    app_module.load_model(tmp_path / "missing.joblib")
    METRICS.reset()
    assert client.post("/predict", json=consistent_features()).status_code == 503
    body = client.get("/metrics").json()
    assert body["endpoints"]["POST /predict"]["errors"] == 1 and body["total_errors"] == 1


def test_validation_errors_are_not_server_errors(loaded):
    client.post("/predict", json={"n_signals": -1})
    assert client.get("/metrics").json()["endpoints"]["POST /predict"]["errors"] == 0


def test_responses_carry_their_processing_time(loaded):
    response = client.get("/health")
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_each_request_is_logged(loaded, caplog):
    with caplog.at_level("INFO", logger="uvicorn.error"):
        client.get("/health")
    assert any("request method=GET path=/health status=200 latency_ms=" in r.message for r in caplog.records)


def test_latency_window_and_percentile():
    metrics = RequestMetrics()
    for ms in range(1, 1201):
        metrics.record("GET", "/x", 200, float(ms))
    stats = metrics.snapshot()["endpoints"]["GET /x"]
    assert stats["count"] == 1200 and stats["max_latency_ms"] == 1200
    assert stats["avg_latency_ms"] == pytest.approx(600.5)
    assert 1100 <= stats["p95_latency_ms"] <= 1200  # p95 of the most recent 1000 (201..1200)
    assert _percentile([], 95) == 0.0


def test_stats_is_an_alias_of_metrics(loaded):
    client.post("/predict", json=consistent_features())
    from_metrics = client.get("/metrics").json()
    from_stats = client.get("/stats").json()
    assert set(from_stats) == set(from_metrics)
    # /stats is answered one request later, so it also counts the /metrics call before it.
    assert from_stats["total_requests"] == from_metrics["total_requests"] + 1
    assert from_stats["predictions"] == from_metrics["predictions"]
    assert from_stats["endpoints"]["GET /metrics"]["count"] == 1
    paths = client.get("/openapi.json").json()["paths"]
    assert "get" in paths["/metrics"] and "get" in paths["/stats"]


def test_monitoring_tracks_success_invalid_input_and_server_errors(model_path):
    """The four cases shown in the demo: a health check, a prediction, an invalid request and a 503."""
    app_module.load_model(model_path)
    METRICS.reset()
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json=consistent_features()).status_code == 200
    assert client.post("/predict", json={"n_signals": -1}).status_code == 422
    app_module.load_model(model_path.parent / "missing.joblib")
    assert client.post("/predict", json=consistent_features()).status_code == 503
    app_module.STATE.update(bundle=None, error=None)
    body = client.get("/stats").json()
    assert body["total_requests"] == 4 and body["total_errors"] == 1
    assert body["endpoints"]["POST /predict"] == {**body["endpoints"]["POST /predict"], "count": 3, "errors": 1}
    assert body["endpoints"]["GET /health"]["count"] == 1
    assert sum(body["predictions"].values()) == 1 and body["avg_latency_ms"] > 0
