"""Follow-up to B, C and D, run on CV before any test scoring. It answers two questions they raised.

1. Was oversampling's win really caused by more trees? Every option below has early stopping off and
   the tree count fixed, so weighting method and tree count can be read separately.
2. Was the inductive drop in D caused by an off-by-one in the method? The "aligned" variant has every
   IP look up one frozen table, which is also how a deployed service would work.

Overall selection rule, fixed before these results existed: the highest 5-fold CV F1 across B, C and
this follow-up; it must beat the baseline by min_cv_f1_gain; within that margin of the best, the
simplest wins (no resampling, then fewer trees).

Usage: python -m experiments.exp_c2_followup
"""

from experiments.common import load_records, load_split, save_records
from experiments.exp_d_features import load_lookups, run_inductive
from experiments.runner import run_cv

TREES = {
    "balanced": {"class_weight": "balanced"},
    "unweighted": {"class_weight": None},
}


def main() -> None:
    train, _ = load_split()
    records = []
    for label, base in TREES.items():
        for trees in ([600, 800] if label == "balanced" else [200, 400, 600]):
            spec = {"kind": "hgb", "overrides": {**base, "early_stopping": False, "max_iter": trees}}
            records.append(run_cv(f"{label}_trees_{trees}", "exp_c2", spec, train,
                                  f"{label} weighting, {trees} trees, early stopping off"))
            save_records("exp_c2", records)
    for trees in (200, 400, 600):
        spec = {"kind": "hgb", "overrides": {"class_weight": None, "early_stopping": False, "max_iter": trees},
                "strategy": "over", "ratio": 1.0}
        records.append(run_cv(f"oversample_1to1_trees_{trees}", "exp_c2", spec, train,
                              f"oversample 1:1, {trees} trees, early stopping off"))
        save_records("exp_c2", records)

    exp_d = load_records("exp_d")
    if not any(r["name"] == "inductive_aligned" for r in exp_d):
        prints, pairs = load_lookups()
        exp_d.append(run_inductive(train, prints, pairs, aligned=True))
        save_records("exp_d", exp_d)


if __name__ == "__main__":
    main()
