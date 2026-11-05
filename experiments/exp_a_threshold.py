"""Experiment A: how does the decision threshold change precision, recall and F1?

Only the cut-off changes. One set of out-of-fold probabilities from hgb_slow is reused, so
no new model is trained for each threshold. Test scoring happens later, in final_test_eval.

Usage: python -m experiments.exp_a_threshold
"""

import numpy as np

from experiments.common import ARTIFACT_DIR, counts, load_config, load_split, save_records
from experiments.runner import run_cv
from src.config import TARGET
from src.train import evaluate

BASE = {"kind": "hgb", "overrides": {}}


def main() -> None:
    train, _ = load_split()
    y = train[TARGET].to_numpy()
    tuned = run_cv("tuned_threshold", "exp_a", BASE, train, "threshold tuned on out-of-fold predictions",
                   cache_as="hgb_slow_base")
    oof = np.load(ARTIFACT_DIR / "oof_hgb_slow_base.npz")["oof"]
    records = [tuned]
    print(f"{'thr':>6}{'prec':>8}{'recall':>8}{'F1':>8}{'TP':>6}{'FP':>7}{'FN':>6}")
    for t in load_config()["threshold_grid"]:
        m, c = evaluate(y, oof, t), counts(y, oof, t)
        print(f"{t:6.2f}{m['precision']:8.3f}{m['recall']:8.3f}{m['f1']:8.4f}{c['tp']:6d}{c['fp']:7d}{c['fn']:6d}",
              flush=True)
        records.append({"run_name": f"exp_a_threshold_{t:.2f}", "name": f"threshold_{t:.2f}", "experiment": "exp_a",
                        "change": f"fixed threshold {t:.2f}", "spec": BASE, "params": {"threshold": t},
                        "metrics": {"threshold": t, "cv_f1": m["f1"], "cv_precision": m["precision"],
                                    "cv_recall": m["recall"], "cv_pr_auc": m["pr_auc"],
                                    **{f"cv_{k}": v for k, v in c.items()}}})
    save_records("exp_a", records)


if __name__ == "__main__":
    main()
