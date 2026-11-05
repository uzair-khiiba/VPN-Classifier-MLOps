"""Run the Streamlit pages headlessly, against the real FastAPI app over its HTTP interface."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import app as app_module  # noqa: E402
from src.features import FEATURES  # noqa: E402
from ui import state  # noqa: E402
from ui.api_client import ApiClient  # noqa: E402

APP_FILE = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")
TIMEOUT = 60


def run_detect():
    from ui.views import detect

    detect.render()


def run_explain():
    from ui.views import explain

    explain.render()


def run_home():
    from ui.views import home

    home.render()


def run_performance():
    from ui.views import performance

    performance.render()


def run_status():
    from ui.views import status

    status.render()


def page(script) -> AppTest:
    return AppTest.from_function(script, default_timeout=TIMEOUT)


def all_text(app: AppTest) -> str:
    groups = (app.markdown, app.caption, app.error, app.warning, app.info, app.success, app.title, app.subheader)
    return " ".join(str(element.value) for group in groups for element in group)


def metric_value(app: AppTest, label: str) -> str:
    return next(metric.value for metric in app.metric if metric.label == label)


@pytest.fixture
def api(model_path, monkeypatch):
    app_module.load_model(model_path)
    monkeypatch.setattr(state, "get_client", lambda: ApiClient("http://testserver", client=TestClient(app_module.app)))
    _rebind_views(monkeypatch)
    yield
    app_module.STATE.update(bundle=None, error=None)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(state, "get_client", lambda: ApiClient("http://127.0.0.1:9", timeout=1.0))
    _rebind_views(monkeypatch)


def _rebind_views(monkeypatch) -> None:
    from ui.views import detect, explain, home, performance, status

    for module in (detect, explain, home, performance, status):
        monkeypatch.setattr(module, "get_client", state.get_client)


def filled_with_sample(app: AppTest) -> AppTest:
    return app.button(key="load_Sample labelled VPN/proxy").click().run()


def test_app_starts_and_reports_offline(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")
    app = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT).run()
    assert not app.exception
    assert "API offline" in " ".join(item.value for item in app.sidebar.markdown)
    assert "Explainable VPN & Proxy Detector" in all_text(app)


def test_status_page_never_claims_online_when_offline(offline):
    app = page(run_status).run()
    assert not app.exception
    assert metric_value(app, "API") == "Offline"
    assert "Online" not in [metric.value for metric in app.metric]
    assert "uvicorn" in " ".join(item.value for item in app.code)


def test_status_page_reports_a_real_online_response(api):
    app = page(run_status).run()
    assert not app.exception
    assert metric_value(app, "API") == "Online" and metric_value(app, "Model") == "Loaded"
    assert metric_value(app, "Model identifier") == "test-logreg"
    assert metric_value(app, "Response time").endswith("ms")


def test_status_page_reports_a_missing_model(tmp_path, monkeypatch):
    app_module.load_model(tmp_path / "missing.joblib")
    monkeypatch.setattr(state, "get_client", lambda: ApiClient("http://testserver", client=TestClient(app_module.app)))
    _rebind_views(monkeypatch)
    app = page(run_status).run()
    assert metric_value(app, "Model") == "Not loaded" and not app.success


def test_form_starts_empty_with_every_model_feature(api):
    app = page(run_detect).run()
    assert not app.exception
    inputs = {item.key: item.value for item in [*app.number_input, *app.selectbox]}
    assert set(inputs) == {f"feature_{name}" for name in FEATURES}
    assert all(value is None for value in inputs.values())


def test_empty_form_sends_nothing(api, monkeypatch):
    sent = []
    original = ApiClient.predict
    monkeypatch.setattr(ApiClient, "predict", lambda self, features: sent.append(features) or original(self, features))
    app = page(run_detect).run()
    app = app.button(key="predict").click().run()
    assert not sent and "Nothing was sent" in all_text(app)


def test_prediction_flow_shows_the_api_result_and_labels_the_demo(api):
    app = filled_with_sample(page(run_detect).run())
    app = app.button(key="predict").click().run()
    assert not app.exception
    text = all_text(app)
    expected = app.session_state[state.LAST_PREDICTION]
    assert expected["label"] in text and "Demo record" in text and "not a detection of a live IP" in text
    assert metric_value(app, "Decision threshold") == f"{expected['threshold']:.4f}"


def test_edited_sample_is_not_labelled_as_a_demo(api):
    app = filled_with_sample(page(run_detect).run())
    app.number_input(key="feature_n_signals").set_value(46.0)
    app = app.button(key="predict").click().run()
    assert app.session_state[state.LAST_SOURCE] is None and "Demo record" not in all_text(app)


def test_offline_prediction_shows_no_result(api, monkeypatch):
    app = filled_with_sample(page(run_detect).run())
    from ui.views import detect

    monkeypatch.setattr(detect, "get_client", lambda: ApiClient("http://127.0.0.1:9", timeout=1.0))
    app = app.button(key="predict").click().run()
    assert not app.exception and "API is offline" in all_text(app)
    assert state.LAST_PREDICTION not in app.session_state


def test_explain_page_asks_for_a_prediction_first(api):
    app = page(run_explain).run()
    assert not app.exception and "Nothing to explain yet" in all_text(app)


def test_explain_page_shows_the_api_explanation_and_limits(api):
    app = filled_with_sample(page(run_detect).run())
    app = app.button(key="predict").click().run()
    features, prediction = app.session_state[state.LAST_FEATURES], app.session_state[state.LAST_PREDICTION]

    explain_app = page(run_explain)
    explain_app.session_state[state.LAST_FEATURES] = features
    explain_app.session_state[state.LAST_PREDICTION] = prediction
    explain_app.run()
    explain_app = explain_app.button(key="explain").click().run()
    assert not explain_app.exception
    text = all_text(explain_app)
    assert "not SHAP" in text and "show causes" in text and "Most influential features" in text
    assert len(explain_app.session_state[state.LAST_EXPLANATION]["contributions"]) == 8


def test_home_and_performance_render_with_the_api(api):
    for script in (run_home, run_performance):
        app = page(script).run()
        assert not app.exception
        assert "test-logreg" in all_text(app)


def test_performance_page_is_honest_when_offline(offline):
    app = page(run_performance).run()
    assert not app.exception and "API is offline" in all_text(app)
    assert "has not confirmed" in all_text(app)
