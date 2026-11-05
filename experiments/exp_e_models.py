"""Experiment E: compare the project's existing algorithms on identical folds.

Usage: python -m experiments.exp_e_models
"""

from experiments.common import load_split, save_records
from experiments.runner import run_cv
from src.train import CANDIDATES


def main() -> None:
    train, _ = load_split()
    records = []
    for name in CANDIDATES:
        spec = {"kind": "candidate", "candidate": name}
        cache = "hgb_slow_base" if name == "hgb_slow" else None
        records.append(run_cv(name, "exp_e", spec, train, f"algorithm: {name}", cache_as=cache))
        save_records("exp_e", records)


if __name__ == "__main__":
    main()
