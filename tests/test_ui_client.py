"""The dashboard's HTTP client: real calls to the FastAPI app, plus every way a backend can fail."""

import ast
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src import app as app_module
from src.features import FEATURES
from ui.api_client import (
    ApiClient,
    ApiInvalidInputError,
    ApiResponseError,
    ApiUnavailableError,
    FeatureSpec,
    ModelUnavailableError,
    get_api_base_url,
)
from ui.features import LABELS, consistency_warnings, group_specs, validate_inputs

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def mocked(handler) -> ApiClient:
    return ApiClient("http://api.test", client=httpx.Client(transport=httpx.MockTransport(handler)))


def answering(status: int = 200, **kwargs) -> ApiClient:
    return mocked(lambda request: httpx.Response(status, **kwargs))


@pytest.fixture
def real_api(model_path):
    app_module.load_model(model_path)
    yield ApiClient("http://testserver", client=TestClient(app_module.app))
    app_module.STATE.update(bundle=None, error=None)


def payload() -> dict:
    return {**{f: 1.0 for f in FEATURES}, "n_signals": 12, "asn_unique_ips": 40}


def test_predict_through_real_http_stack(real_api):
    data = real_api.predict(payload()).data
    assert data["label"] in {"VPN/Proxy", "Not VPN/Proxy"} and 0 <= data["probability"] <= 1


def test_explain_through_real_http_stack(real_api):
    data = real_api.explain(payload(), top_k=3).data
    assert len(data["contributions"]) == 3 and "median" in data["method"]


def test_health_and_model_info_through_real_http_stack(real_api):
    result = real_api.health()
    assert result.data["model_loaded"] is True and result.elapsed_ms >= 0
    assert real_api.model_info().data["model"] == "test-logreg"


def test_schema_comes_from_the_api(real_api):
    specs, examples = real_api.feature_schema()
    assert [spec.name for spec in specs] == FEATURES
    assert len(examples) == 2 and set(examples[0]) == set(FEATURES)
    by_name = {spec.name: spec for spec in specs}
    assert by_name["n_signals"].minimum == 1 and by_name["share_http_scan"].maximum == 1


def test_real_api_rejection_names_the_field(real_api):
    with pytest.raises(ApiInvalidInputError) as caught:
        real_api.predict({**payload(), "n_signals": -5})
    assert ("n_signals", "Input should be greater than or equal to 1") in caught.value.field_errors


def test_real_api_without_model_raises_model_unavailable(tmp_path):
    app_module.load_model(tmp_path / "missing.joblib")
    client = ApiClient("http://testserver", client=TestClient(app_module.app))
    with pytest.raises(ModelUnavailableError):
        client.predict(payload())
    assert client.health().data["model_loaded"] is False
    app_module.STATE.update(bundle=None, error=None)


def test_backend_down_raises_unavailable():
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ApiUnavailableError, match="Could not reach"):
        mocked(refuse).health()


def test_timeout_raises_unavailable():
    def too_slow(request):
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(ApiUnavailableError):
        mocked(too_slow).predict({})


def test_a_real_closed_port_raises_unavailable():
    with pytest.raises(ApiUnavailableError):
        ApiClient("http://127.0.0.1:9", timeout=1.0).health()


@pytest.mark.parametrize("response", [{"content": b"<html>not json</html>"}, {"json": ["a", "list"]},
                                      {"json": {"status": "ok"}}])
def test_invalid_response_is_rejected(response):
    with pytest.raises(ApiResponseError):
        answering(200, **response).health()


def test_prediction_missing_fields_is_rejected():
    with pytest.raises(ApiResponseError, match="missing"):
        answering(200, json={"label": "VPN/Proxy"}).predict(payload())


def test_non_boolean_model_loaded_is_rejected():
    with pytest.raises(ApiResponseError):
        answering(200, json={"status": "ok", "model_loaded": "yes"}).health()


def test_server_error_is_reported():
    with pytest.raises(ApiResponseError, match="500"):
        answering(500, json={"detail": "boom"}).predict(payload())


def test_schema_without_features_is_rejected():
    with pytest.raises(ApiResponseError):
        answering(200, json={"components": {"schemas": {}}}).feature_schema()


def test_base_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://example.test:9000/")
    assert get_api_base_url() == "http://example.test:9000"
    assert ApiClient().base_url == "http://example.test:9000"


def test_base_url_falls_back_to_the_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('API_BASE_URL="http://from-file.test:8000"\n')
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.setattr("ui.api_client.ENV_FILE", env_file)
    assert get_api_base_url() == "http://from-file.test:8000"


def test_every_model_feature_has_a_label():
    assert set(LABELS) == set(FEATURES)


def test_unknown_api_feature_still_reaches_the_form():
    grouped = group_specs([FeatureSpec("brand_new", None, 0, None), FeatureSpec("n_signals", None, 1, None)])
    assert [spec.name for spec in grouped["Other"]] == ["brand_new"]


def test_validation_blocks_missing_and_out_of_range_values():
    specs = [FeatureSpec("n_signals", None, 1, None), FeatureSpec("share_tcp_scan", None, 0, 1),
             FeatureSpec("n_watchers", None, 1, None)]
    problems = validate_inputs({"n_signals": 0, "share_tcp_scan": 2, "n_watchers": None}, specs)
    assert len(problems) == 3
    assert "at least 1" in problems[0] and "at most 1" in problems[1] and "required" in problems[2]


def test_consistency_warnings_flag_impossible_combinations():
    assert consistency_warnings({"has_shodan": 0, "n_tcp_ports": 3, "n_udp_ports": 0})
    assert consistency_warnings({"share_http_scan": 0.8, "share_http_spam": 0.8})
    assert consistency_warnings({"n_signals": 10, "n_active_days": 2, "signals_per_day": 9})
    assert not consistency_warnings({"has_shodan": 1, "n_tcp_ports": 3, "n_udp_ports": 0, "n_signals": 10,
                                     "n_active_days": 2, "signals_per_day": 5})


def test_the_ui_never_imports_the_model_or_the_backend():
    forbidden = {"joblib", "sklearn", "pickle", "src"}
    for path in UI_DIR.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not {name.split(".")[0] for name in names} & forbidden, f"{path} imports {names}"
