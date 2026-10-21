import numpy as np
import pandas as pd

from src.explain import f1_at_threshold, permutation_f1_importance, plot_importance
from src.features import FEATURES
from src.model_io import load_bundle


def test_permutation_importance_ranks_the_informative_feature_first(model_path):
    bundle = load_bundle(model_path)
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.random((300, len(FEATURES))) * 10, columns=FEATURES)
    y = (X["asn_unique_ips"] > 5).astype(int).to_numpy()
    importance = permutation_f1_importance(bundle, X, y, n_repeats=3)
    assert sorted(importance["feature"]) == sorted(FEATURES)
    assert importance.loc[0, "feature"] == "asn_unique_ips"  # the only feature the fixture model learned
    assert importance["f1_drop"].is_monotonic_decreasing


def test_f1_scorer_uses_the_given_threshold(model_path):
    bundle = load_bundle(model_path)
    X = pd.DataFrame([{f: 10.0 for f in FEATURES}])
    assert f1_at_threshold(0.01)(bundle["model"], X, np.array([1])) == 1.0
    assert f1_at_threshold(1.01)(bundle["model"], X, np.array([1])) == 0.0  # nothing reaches it


def test_plot_importance_writes_png(tmp_path):
    importance = pd.DataFrame({"feature": FEATURES[:3], "f1_drop": [0.3, 0.2, 0.1], "f1_drop_std": [0.01] * 3})
    plot_importance(importance, tmp_path / "imp.png")
    assert (tmp_path / "imp.png").stat().st_size > 0
