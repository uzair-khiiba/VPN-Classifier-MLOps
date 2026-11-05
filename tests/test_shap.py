"""SHAP must explain the VPN/proxy class, add up to the model's output, and report the explainer that really ran."""

import json
import random

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

pytest.importorskip("shap")

from src import app as app_module  # noqa: E402
from src import shap_report  # noqa: E402
from src.config import ROOT  # noqa: E402
from src.features import FEATURES  # noqa: E402
from src.model_io import save_bundle  # noqa: E402
from src.shap_explain import (  # noqa: E402
    ShapExplainer,
    ShapUnavailableError,
    global_ranking,
    log_odds,
)

client = TestClient(app_module.app)
SIGNAL = "asn_unique_ips"


def data(n: int = 400) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((n, len(FEATURES))) * 10, columns=FEATURES)
    return X, (X[SIGNAL] > 5).astype(int).to_numpy()


def bundle_for(model) -> dict:
    X, y = data()
    model.fit(X, y)
    return {"model": model, "features": FEATURES, "threshold": 0.5, "reference": X.median().to_dict(),
            "metadata": {"model_name": "test-model", "version": "v1"}}


@pytest.fixture(scope="module")
def tree_bundle():
    return bundle_for(HistGradientBoostingClassifier(max_iter=20, random_state=0))


def test_tree_model_uses_tree_explainer_and_says_so(tree_bundle):
    explainer = ShapExplainer(tree_bundle)
    assert explainer.explainer_name == "TreeExplainer" and "TreeExplainer" in explainer.detail


def test_values_add_up_to_the_positive_class_log_odds(tree_bundle):
    X, _ = data()
    values, base = ShapExplainer(tree_bundle).values(X.head(50))
    assert values.shape == (50, len(FEATURES))
    expected = tree_bundle["model"].decision_function(X.head(50))
    assert np.allclose(base + values.sum(axis=1), expected)
    proba = tree_bundle["model"].predict_proba(X.head(50))[:, 1]
    assert np.allclose(1 / (1 + np.exp(-(base + values.sum(axis=1)))), proba)


def test_direction_refers_to_the_vpn_class(tree_bundle):
    """Higher SIGNAL means class 1 in this data, so a high value must push toward VPN/proxy and a low one away."""
    explainer = ShapExplainer(tree_bundle)
    low, high = {f: 1.0 for f in FEATURES}, {f: 1.0 for f in FEATURES}
    low[SIGNAL], high[SIGNAL] = 0.5, 9.5
    by_feature = {c["feature"]: c for c in explainer.explain_row(high)["contributions"]}
    assert by_feature[SIGNAL]["shap_value"] > 0 and by_feature[SIGNAL]["direction"] == "toward VPN/proxy"
    by_feature = {c["feature"]: c for c in explainer.explain_row(low)["contributions"]}
    assert by_feature[SIGNAL]["shap_value"] < 0 and by_feature[SIGNAL]["direction"] == "away from VPN/proxy"


def test_feature_order_of_the_input_does_not_matter(tree_bundle):
    X, _ = data()
    explainer = ShapExplainer(tree_bundle)
    row = X.iloc[0].to_dict()
    shuffled = list(row.items())
    random.Random(0).shuffle(shuffled)
    assert explainer.explain_row(row) == explainer.explain_row(dict(shuffled))
    by_feature = {c["feature"]: c for c in explainer.explain_row(row)["contributions"]}
    assert by_feature[SIGNAL]["value"] == row[SIGNAL]


def test_missing_values_are_handled_and_still_add_up(tree_bundle):
    X, _ = data()
    rows = X.head(5).copy()
    rows.iloc[0, 3] = np.nan
    values, base = ShapExplainer(tree_bundle).values(rows)
    assert np.allclose(base + values.sum(axis=1), tree_bundle["model"].decision_function(rows))


def test_non_tree_model_falls_back_and_reports_it():
    bundle = bundle_for(LogisticRegression(max_iter=2000))
    explainer = ShapExplainer(bundle)
    assert explainer.explainer_name == "PermutationExplainer"
    X, _ = data()
    values, base = explainer.values(X.head(3))
    assert np.allclose(base + values.sum(axis=1), log_odds(bundle["model"], X.head(3)))


def test_wrong_classes_are_refused():
    X, y = data()
    model = HistGradientBoostingClassifier(max_iter=5).fit(X, y + 1)
    with pytest.raises(ShapUnavailableError, match="binary"):
        ShapExplainer({"model": model, "features": FEATURES, "reference": X.median().to_dict()})


def test_global_ranking_sorts_by_size_of_effect():
    ranking = global_ranking(np.array([[1.0, -5.0], [1.0, 5.0]]), ["a", "b"])
    assert ranking["feature"].tolist() == ["b", "a"] and ranking["mean_shap"].tolist() == [0.0, 1.0]


@pytest.fixture
def served(tree_bundle, tmp_path):
    app_module.load_model(save_bundle(tree_bundle, tmp_path / "model.joblib"))
    yield
    app_module.STATE.update(bundle=None, error=None, shap=None)


def payload() -> dict:
    return {**{f: 1.0 for f in FEATURES}, SIGNAL: 8.0}  # within the limits the API accepts


def test_api_returns_values_that_add_up_and_match_predict(served):
    body = client.post("/explain/shap", json=payload()).json()
    predicted = client.post("/predict", json=payload()).json()
    assert body["explainer"] == "TreeExplainer" and body["positive_class"] == 1
    assert body["probability"] == predicted["probability"] and body["model_version"] == predicted["model_version"]
    assert len(body["contributions"]) == len(FEATURES)
    total = body["base_value"] + sum(c["shap_value"] for c in body["contributions"])
    assert total == pytest.approx(body["output_value"])
    assert 1 / (1 + np.exp(-body["output_value"])) == pytest.approx(body["probability"], abs=1e-4)
    assert "cause" in body["limitation"]


def test_api_top_k_keeps_the_largest_and_the_full_output(served):
    every = client.post("/explain/shap", json=payload()).json()
    top = client.post("/explain/shap", params={"top_k": 3}, json=payload()).json()
    assert top["contributions"] == every["contributions"][:3] and top["output_value"] == every["output_value"]


def test_api_reports_when_shap_fails_and_old_explain_still_works(served, monkeypatch):
    def broken(_):
        raise ShapUnavailableError("broken")

    monkeypatch.setattr(app_module, "ShapExplainer", broken)
    app_module.STATE["shap"] = None
    answer = client.post("/explain/shap", json=payload())
    assert answer.status_code == 503 and "broken" not in answer.text
    old = client.post("/explain", json=payload()).json()
    assert old["method"] == "single-feature replacement with training median" and "reference" in old["contributions"][0]


def test_api_needs_a_model(tmp_path):
    app_module.load_model(tmp_path / "missing.joblib")
    assert client.post("/explain/shap", json=payload()).status_code == 503


def test_global_endpoint_only_returns_a_report_for_the_served_model(served, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "SHAP_REPORTS", tmp_path)
    assert client.get("/explain/shap/global").status_code == 404
    folder = tmp_path / "test-model"
    folder.mkdir()
    (folder / "shap_global.json").write_text(json.dumps({"model": "test-model-OLD", "ranking": []}))
    assert client.get("/explain/shap/global").status_code == 404
    (folder / "shap_global.json").write_text(json.dumps({"model": "test-model-v1", "ranking": [{"feature": "a"}] * 5}))
    assert len(client.get("/explain/shap/global", params={"top_k": 2}).json()["ranking"]) == 2


def test_report_writes_every_plot_and_the_ranking(tree_bundle, tmp_path):
    X, _ = data(120)
    report = shap_report.build_report(tree_bundle, X, tmp_path, {"example": X.iloc[0].to_dict()})
    for name in ("shap_summary.png", "shap_bar.png", "shap_ranking.csv", "shap_global.json", "local_example.png"):
        assert (tmp_path / name).stat().st_size > 0, name
    assert report["explainer"] == "TreeExplainer" and len(report["ranking"]) == len(FEATURES)


def test_served_model_is_explained_with_tree_explainer_and_adds_up():
    """The real deployed model, on the example records."""
    app_module.load_model(ROOT / "models" / "model.joblib")
    try:
        for name in ("vpn_ip", "regular_ip"):
            row = json.loads((ROOT / "examples" / f"{name}.json").read_text())
            body = client.post("/explain/shap", json=row).json()
            predicted = client.post("/predict", json=row).json()
            assert body["explainer"] == "TreeExplainer"
            assert 1 / (1 + np.exp(-body["output_value"])) == pytest.approx(predicted["probability"], abs=1e-4)
    finally:
        app_module.STATE.update(bundle=None, error=None, shap=None)
