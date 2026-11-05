"""How much do the four whole-population features change if counted over training IPs only?

src/features.py builds asn_unique_ips and the three *_shared counts before the split, across every IP in the raw
files. That includes test IPs, and Shodan rows for the Kaggle test set. No labels are involved, so label information
can't leak this way. It's transductive: the test IPs' existence affects the counts.

This measures the size of that effect, and can write an alternative table with counts frozen from training IPs only.
Existing data, models and results stay as they are. The alternative table is opt-in:
PROCESSED_DIR=data/processed-inductive

Usage:
    python -m experiments.audit_population_features
    python -m experiments.audit_population_features --write-inductive
"""

import argparse
import json

import numpy as np
import pandas as pd

from experiments.common import RESULTS_DIR, load_records, load_split
from experiments.exp_d_features import inductive_counts, load_lookups
from src.config import ID_COLUMN, ROOT, TARGET
from src.feature_docs import POPULATION_FEATURES

INDUCTIVE_DIR = ROOT / "data" / "processed-inductive"


def feature_shift(current: pd.Series, frozen: pd.Series) -> dict:
    """How far the counts move when only training IPs are counted."""
    delta = current.to_numpy(dtype=float) - frozen.to_numpy(dtype=float)
    return {"rows_changed": float((delta != 0).mean()), "median_change": float(np.median(delta)),
            "p99_change": float(np.percentile(delta, 99)), "max_change": float(delta.max()),
            "spearman": float(pd.Series(current.to_numpy()).corr(pd.Series(frozen.to_numpy()), method="spearman"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-inductive", action="store_true", help=f"also write {INDUCTIVE_DIR.name}/")
    args = parser.parse_args()

    train, test = load_split()
    prints, pairs = load_lookups()
    train_ids, test_ids = train[ID_COLUMN].to_numpy(), test[ID_COLUMN].to_numpy()
    fit_table, test_table = inductive_counts(train_ids, test_ids, prints, pairs, aligned=True)

    shifts = {}
    for feature in POPULATION_FEATURES:
        shifts[feature] = {"train": feature_shift(train[feature], fit_table.loc[train_ids, feature]),
                           "test": feature_shift(test[feature], test_table.loc[test_ids, feature])}
    effect = {r["name"]: {k: r["metrics"].get(k) for k in ("cv_f1", "test_f1", "threshold")}
              for r in load_records("exp_d") if r["name"] in ("all_features", "inductive_aligned", "inductive_counts")}
    comparison = json.loads((RESULTS_DIR / "test_comparison.json").read_text())
    versus = comparison["vs_baseline"].get("exp_d/inductive_aligned")
    report = {
        "features": POPULATION_FEATURES,
        "population_counted": {"signal_ips": int(pairs[ID_COLUMN].nunique()), "shodan_ips": len(prints),
                               "train_ips": len(train), "test_ips": len(test)},
        "labels_used_in_these_features": False,
        "finding": "Transductive by design, with no label leakage. The counts include test IPs and Kaggle test IPs "
                   "but never read a label.",
        "shift_when_counting_training_ips_only": shifts,
        "saved_retraining_results": effect,
        "test_f1_difference_frozen_vs_current": versus,
        "existing_results_unchanged": True,
    }
    path = RESULTS_DIR / "population_feature_audit.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("population_counted", "shift_when_counting_training_ips_only",
                                             "saved_retraining_results", "test_f1_difference_frozen_vs_current")},
                     indent=1)[:3500])
    print(f"Wrote {path}")

    if args.write_inductive:
        INDUCTIVE_DIR.mkdir(parents=True, exist_ok=True)
        for name, frame, table in (("train", train, fit_table), ("test", test, test_table)):
            out = frame.copy()
            for feature in POPULATION_FEATURES:
                out[feature] = table.loc[frame[ID_COLUMN], feature].to_numpy()
            if out[TARGET].isna().any():
                raise RuntimeError("missing labels")
            out.to_parquet(INDUCTIVE_DIR / f"{name}.parquet", index=False)
        print(f"Wrote {INDUCTIVE_DIR}. Existing data/processed is unchanged.")


if __name__ == "__main__":
    main()
