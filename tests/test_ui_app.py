"""Run the Streamlit pages headlessly, against the real FastAPI app over its HTTP interface."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import app as app_module  # noqa: E402
from src.features import FEATURES  # noqa: E402
from tests.conftest import consistent_features  # noqa: E402
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
def api_with_data(api):
    """The API with the committed test-IP lookup table and feature reference loaded."""
    app_module.load_ip_data()
    yield
    app_module.STATE.update(lookup=None, reference=None)


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
    inputs = {item.key: item.value for item in [*app.number_input, *app.selectbox] if item.key.startswith("feature_")}
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
    app.number_input(key="feature_asn_unique_ips").set_value(12_345.0)
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


def test_ip_lookup_shows_the_prediction_and_the_true_class(api_with_data):
    app = page(run_detect).run()
    app = app.button(key="example_ip_47584").click().run()
    app = app.button(key="predict_ip").click().run()
    assert not app.exception
    result = app.session_state[state.LAST_PREDICTION]
    assert result["ip_id"] == 47584 and result["actual_label_name"] == "VPN/Proxy"
    assert metric_value(app, "Actual class (test set)") == "VPN/Proxy"
    assert "held-out test IP 47584" in all_text(app)
    assert app.session_state[state.LAST_FEATURES] == result["features"]  # the Explainability page uses these


def test_ip_not_in_the_test_set_says_so_and_shows_no_result(api_with_data):
    app = page(run_detect).run()
    app.number_input(key="ip_id").set_value(199_999)
    app = app.button(key="predict_ip").click().run()
    assert not app.exception and "not in the held-out test set" in all_text(app)
    assert state.LAST_PREDICTION not in app.session_state


def test_empty_ip_id_sends_nothing(api_with_data):
    app = page(run_detect).run()
    app = app.button(key="predict_ip").click().run()
    assert "Enter an IP ID first" in all_text(app) and state.LAST_PREDICTION not in app.session_state


def test_example_attack_events_are_scored_from_computed_features(api_with_data):
    app = page(run_detect).run()
    app = app.button(key="load_signals_example").click().run()
    app = app.button(key="predict_signals").click().run()
    assert not app.exception
    result = app.session_state[state.LAST_PREDICTION]
    assert result["source"] == "computed from signals" and len(result["features"]) == len(FEATURES)
    assert app.session_state[state.LAST_SOURCE] is None  # a computed IP is not labelled as a demo record


def test_empty_attack_events_send_nothing(api_with_data):
    app = page(run_detect).run()
    app = app.button(key="predict_signals").click().run()
    assert "Add at least one attack signal" in all_text(app) and state.LAST_PREDICTION not in app.session_state


def fake_export(monkeypatch):
    from ui import reports
    from ui.views import detect

    rows = []
    for ip, (split, label) in enumerate([("test", 1), ("test", 0), ("train", 1)], start=100):
        rows.append({"attacker_ip_enum": ip, "label": label, "split": split,
                     "label_name": "VPN/proxy" if label else "Not VPN/proxy", **consistent_features()})
    n_signals = [10.0, 20.0, 30.0]  # different per row; signals_per_day stays n_signals / n_active_days
    table = pd.DataFrame(rows).assign(n_signals=n_signals, signals_per_day=[n / 2 for n in n_signals])
    detect._export.clear()
    monkeypatch.setattr(reports, "load_export", lambda: table)
    return table


def test_record_loads_by_id_into_all_28_fields_and_predicts(api, monkeypatch):
    fake_export(monkeypatch)
    app = page(run_detect).run()
    app.text_input(key="export_ip_id").set_value("101")
    app = app.button(key="load_by_id").click().run()
    assert not app.exception
    assert app.number_input(key="feature_n_signals").value == 20.0
    assert app.session_state["loaded_sample"].startswith("IP 101")
    app = app.button(key="predict").click().run()
    assert app.session_state[state.LAST_SOURCE].startswith("IP 101") and "Demo record" in all_text(app)
    assert "Model that answered" in all_text(app)


def test_training_record_is_flagged_as_unrepresentative(api, monkeypatch):
    fake_export(monkeypatch)
    app = page(run_detect).run()
    app.text_input(key="export_ip_id").set_value("102")
    app = app.button(key="load_by_id").click().run()
    app = app.button(key="predict").click().run()
    assert "trained on this record" in all_text(app)


def test_unknown_id_says_so_and_fills_nothing(api, monkeypatch):
    fake_export(monkeypatch)
    app = page(run_detect).run()
    app.text_input(key="export_ip_id").set_value("999")
    app = app.button(key="load_by_id").click().run()
    assert "No record has attacker_ip_enum 999" in all_text(app)
    assert app.number_input(key="feature_n_signals").value is None


def test_random_record_respects_the_filters(api, monkeypatch):
    fake_export(monkeypatch)
    app = page(run_detect).run()
    app.radio(key="export_label").set_value("VPN/proxy")
    app = app.button(key="load_random").click().run()
    assert "IP 100" in app.session_state["loaded_sample"]


def test_pasted_excel_row_fills_the_form_and_isnt_called_a_demo(api):
    app = page(run_detect).run()
    row = ["2"] * len(FEATURES)
    row[0] = "77"
    row[FEATURES.index("share_http_scan")] = "0.5"
    row[FEATURES.index("has_shodan")] = "1"
    row[FEATURES.index("no_print_with_ports")] = "0"
    app.text_area(key="pasted_row").set_value("\t".join(row))
    app = app.button(key="load_pasted").click().run()
    assert app.number_input(key="feature_n_signals").value == 77.0
    assert app.number_input(key="feature_share_http_scan").value == 0.5
    assert "loaded_sample" not in app.session_state


def test_bad_paste_is_explained(api):
    app = page(run_detect).run()
    app.text_area(key="pasted_row").set_value("1\t2\t3")
    app = app.button(key="load_pasted").click().run()
    assert "Expected 28 numbers" in all_text(app)


def test_reset_clears_all_28_fields(api):
    app = filled_with_sample(page(run_detect).run())
    app = app.button(key="clear_form").click().run()
    inputs = [item.value for item in [*app.number_input, *app.selectbox] if item.key.startswith("feature_")]
    assert len(inputs) == len(FEATURES) and all(value is None for value in inputs)


def test_form_explains_each_field_and_that_it_needs_features_not_an_ip(api):
    app = page(run_detect).run()
    assert "engineered features, not an IP address" in all_text(app)
    field = app.number_input(key="feature_share_http_scan")
    assert "Unit:" in field.help and "fraction" in field.help
    assert "seconds" in app.number_input(key="feature_mean_gap_s").help


def test_prediction_shows_shap_without_claiming_cause(api):
    app = filled_with_sample(page(run_detect).run())
    app = app.button(key="predict").click().run()
    assert not app.exception
    shap_result = app.session_state[state.LAST_SHAP]
    assert shap_result["model_version"] == app.session_state[state.LAST_PREDICTION]["model_version"]
    text = all_text(app)
    assert "Why the model gave this score" in text and "toward" in text and "away" in text
    assert "Explainer that ran: PermutationExplainer" in text  # the test model is logistic regression
    assert "doesn't show that the feature causes" in text


def test_prediction_survives_shap_being_unavailable(api, monkeypatch):
    from ui.api_client import ModelUnavailableError

    def unavailable(self, features, top_k=28):
        raise ModelUnavailableError("no shap")

    monkeypatch.setattr(ApiClient, "explain_shap", unavailable)
    app = filled_with_sample(page(run_detect).run())
    app = app.button(key="predict").click().run()
    assert not app.exception and app.session_state[state.LAST_PREDICTION]
    assert "isn't available for this prediction" in all_text(app)


def test_explain_page_keeps_both_methods_clearly_apart(api):
    app = filled_with_sample(page(run_detect).run())
    app = app.button(key="predict").click().run()
    explain_app = page(run_explain)
    explain_app.session_state[state.LAST_FEATURES] = app.session_state[state.LAST_FEATURES]
    explain_app.session_state[state.LAST_PREDICTION] = app.session_state[state.LAST_PREDICTION]
    explain_app.run()
    assert [tab.label for tab in explain_app.tabs] == ["SHAP", "Median replacement"]
    explain_app = explain_app.button(key="explain_shap").click().run()
    assert not explain_app.exception
    assert len(explain_app.session_state[state.LAST_SHAP]["contributions"]) == len(FEATURES)
