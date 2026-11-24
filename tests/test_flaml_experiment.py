"""Experiment G (FLAML) on a tiny synthetic table. Skipped when FLAML isn't installed (as in CI)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("flaml")

from sklearn.base import clone  # noqa: E402

from experiments.exp_g_flaml import evaluate_candidate, search  # noqa: E402
from src.config import TARGET  # noqa: E402
from src.features import FEATURES  # noqa: E402


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df[TARGET] = (df[FEATURES[0]] + 0.5 * df[FEATURES[1]] + rng.normal(scale=0.5, size=n) > 1).astype(int)
    return df


def test_flaml_search_and_evaluation_follow_the_project_protocol():
    train, test = _frame(600, 0), _frame(200, 1)
    automl = search(train[FEATURES], train[TARGET].to_numpy(), budget=8)
    assert automl.best_estimator in {"lgbm", "xgboost", "rf", "extra_tree", "lrl1"}

    result = evaluate_candidate(clone(automl.model.estimator), train, test)
    m = result["metrics"]
    assert 0 < m["threshold"] < 1
    assert m["cv_f1"] > 0.5 and m["test_f1"] > 0.5  # the signal is easy; a broken pipeline would score ~0
    assert len(result["proba"]) == len(test)
    assert {"test_tp", "test_fp", "test_fn", "test_tn", "cv_f1_nested"} <= m.keys()
