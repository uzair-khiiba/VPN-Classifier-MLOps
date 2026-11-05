"""Experiment B: which way of handling class imbalance works best?

Only the imbalance method changes. Resampling touches the training part of each fold only.
Every option gets its own threshold, tuned on untouched out-of-fold predictions.

Usage: python -m experiments.exp_b_imbalance
"""

from experiments.common import load_config, load_split, save_records
from experiments.runner import run_cv


def main() -> None:
    train, _ = load_split()
    records = []
    for cfg in load_config()["imbalance"]:
        overrides = {"class_weight": cfg["class_weight"]}
        for key in ("early_stopping", "max_iter"):
            if key in cfg:
                overrides[key] = cfg[key]
        spec = {"kind": "hgb", "overrides": overrides, "strategy": cfg["strategy"], "ratio": cfg.get("ratio", 1.0)}
        cache = "hgb_slow_base" if cfg["name"] == "class_weight_balanced" else None
        records.append(run_cv(cfg["name"], "exp_b", spec, train, f"imbalance method: {cfg['name']}", cache_as=cache))
        save_records("exp_b", records)


if __name__ == "__main__":
    main()
