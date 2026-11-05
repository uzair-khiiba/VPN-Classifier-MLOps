"""Tests for the experiment helpers. Kept here, so the project's own test suite is unchanged.

Usage: python -m pytest experiments/test_experiment_helpers.py
"""

import numpy as np
import pandas as pd
import pytest

from experiments.common import folds, hgb, nested_threshold_f1, resample
from experiments.exp_c_hyperparameters import sample_configs
from experiments.exp_d_features import inductive_counts
from experiments.final_test_eval import spec_key
from src.config import ID_COLUMN


def test_baseline_factory_matches_train_candidate():
    from src.train import CANDIDATES

    assert hgb().get_params() == CANDIDATES["hgb_slow"]().get_params()


def test_overrides_change_only_named_parameters():
    base, changed = hgb().get_params(), hgb(max_iter=400).get_params()
    assert {k for k in base if base[k] != changed[k]} == {"max_iter"}


def test_folds_match_the_pipeline_split():
    from sklearn.model_selection import StratifiedKFold

    y = np.array([0, 0, 0, 1] * 50)
    theirs = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    for (a_fit, a_val), (b_fit, b_val) in zip(folds(y), theirs, strict=True):
        assert np.array_equal(a_fit, b_fit) and np.array_equal(a_val, b_val)


@pytest.mark.parametrize("strategy,ratio,pos,neg", [("over", 1.0, 100, 100), ("over", 0.1, 10, 100),
                                                    ("under", 1.0, 5, 5), ("under", 0.5, 5, 10)])
def test_resample_reaches_requested_ratio(strategy, ratio, pos, neg):
    y = np.array([1] * 5 + [0] * 100)
    chosen = y[resample(y, strategy, seed=1, ratio=ratio)]
    assert (chosen == 1).sum() == pos and (chosen == 0).sum() == neg


def test_oversampling_leaves_a_fold_already_above_the_ratio_unchanged():
    y = np.array([1] * 10 + [0] * 90)
    chosen = y[resample(y, "over", seed=1, ratio=0.1)]
    assert (chosen == 1).sum() == 10 and (chosen == 0).sum() == 90


def test_no_resampling_keeps_every_row_once():
    y = np.array([1, 0, 0, 0])
    assert resample(y, None, seed=1).tolist() == [0, 1, 2, 3]


def test_resample_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        resample(np.array([0, 1]), "smote", seed=1)


def test_nested_threshold_f1_is_perfect_for_a_perfect_ranker():
    y = np.array([0, 0, 0, 1] * 50)
    assert nested_threshold_f1(y, y.astype(float)) == 1.0


def test_random_search_is_reproducible():
    cfg = {"seed": 42, "n_configs": 3, "space": {"learning_rate": {"loguniform": [0.01, 0.2]}, "max_depth": [None, 6]}}
    assert sample_configs(cfg, 2000) == sample_configs(cfg, 2000)


def test_spec_key_ignores_order_but_not_content():
    assert spec_key({"a": 1, "b": 2}) == spec_key({"b": 2, "a": 1})
    assert spec_key({"a": 1}) != spec_key({"a": 2})


@pytest.fixture
def lookups():
    prints = {i: {"jarm": set(), "ja3s": set(), "headers_hash": {"h"} if i < 3 else set()} for i in range(4)}
    pairs = pd.DataFrame({ID_COLUMN: [0, 1, 2, 3], "attacker_as_num": [7.0, 7.0, 9.0, 7.0]})
    return prints, pairs


def test_inductive_counts_never_count_held_out_ips(lookups):
    prints, pairs = lookups
    fit, new = inductive_counts(np.array([0, 1]), np.array([2, 3]), prints, pairs)
    assert fit.loc[0, "headers_hash_shared"] == 1  # only the other fit IP, not IP 2
    assert new.loc[2, "headers_hash_shared"] == 2  # both fit IPs
    assert fit.loc[0, "asn_unique_ips"] == 2
    assert new.loc[3, "asn_unique_ips"] == 3  # 2 fit IPs plus itself


def test_aligned_counts_use_one_table_for_every_ip(lookups):
    prints, pairs = lookups
    fit, new = inductive_counts(np.array([0, 1]), np.array([2, 3]), prints, pairs, aligned=True)
    assert fit.loc[0, "headers_hash_shared"] == new.loc[2, "headers_hash_shared"] == 2
    assert fit.loc[0, "asn_unique_ips"] == new.loc[3, "asn_unique_ips"] == 2
