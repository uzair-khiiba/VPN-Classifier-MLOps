"""Experiment C: can hyperparameter tuning beat the baseline?

Part 1 changes one parameter at a time. Part 2 is a random search: screen with 3 folds, then
re-check the best configurations on the standard 5 folds. Selection uses CV only, and a
challenger must beat the baseline by min_cv_f1_gain (set in the config before any results).

Usage: python -m experiments.exp_c_hyperparameters
"""

import json

import numpy as np

from experiments.common import RESULTS_DIR, load_config, load_split, save_records
from experiments.runner import run_cv


def sample_configs(random_cfg: dict, max_iter: int) -> list[dict]:
    rng = np.random.default_rng(random_cfg["seed"])
    configs = []
    for _ in range(random_cfg["n_configs"]):
        cfg = {"max_iter": max_iter}
        for param, choices in random_cfg["space"].items():
            if isinstance(choices, dict):
                low, high = choices["loguniform"]
                cfg[param] = round(float(np.exp(rng.uniform(np.log(low), np.log(high)))), 4)
            else:
                cfg[param] = choices[int(rng.integers(len(choices)))]
        configs.append(cfg)
    return configs


def main() -> None:
    cfg = load_config()["hyperparameters"]
    train, _ = load_split()
    records = [run_cv("baseline", "exp_c", {"kind": "hgb", "overrides": {}}, train, "baseline hgb_slow",
                      cache_as="hgb_slow_base")]

    for param, values in cfg["one_at_a_time"].items():
        for value in values:
            if param == "fixed_trees_no_early_stop":
                overrides, label = {"early_stopping": False, "max_iter": value}, f"trees_{value}_no_early_stop"
            else:
                overrides, label = {param: value}, f"{param}_{value}"
            records.append(run_cv(label, "exp_c", {"kind": "hgb", "overrides": overrides}, train, f"only {label}"))
            save_records("exp_c", records)

    rs = cfg["random_search"]
    screened = []
    for i, overrides in enumerate(sample_configs(rs, cfg["max_iter_with_early_stopping"])):
        spec = {"kind": "hgb", "overrides": overrides, "n_splits": rs["n_splits"]}
        rec = run_cv(f"random_{i:02d}_3fold", "exp_c", spec, train, f"random search screen {i}: {overrides}")
        rec["screening_only"] = True
        screened.append(rec)
        records.append(rec)
        save_records("exp_c", records)

    finalists = sorted(screened, key=lambda r: -r["metrics"]["cv_f1"])[: rs["n_finalists"]]
    for rec in finalists:
        spec = {"kind": "hgb", "overrides": rec["spec"]["overrides"]}
        records.append(run_cv(rec["name"].replace("_3fold", "_5fold"), "exp_c", spec, train,
                              f"random search finalist: {spec['overrides']}"))
        save_records("exp_c", records)

    # Selection rule: best 5-fold CV F1, and the gain must be larger than the agreed margin.
    baseline_f1 = records[0]["metrics"]["cv_f1"]
    eligible = [r for r in records if not r.get("screening_only")]
    best = max(eligible, key=lambda r: r["metrics"]["cv_f1"])
    gain = best["metrics"]["cv_f1"] - baseline_f1
    selected = best if gain > cfg["min_cv_f1_gain"] else records[0]
    out = {"best_by_cv": best["name"], "best_cv_f1": best["metrics"]["cv_f1"], "baseline_cv_f1": baseline_f1,
           "gain": gain, "required_gain": cfg["min_cv_f1_gain"], "selected": selected["name"]}
    (RESULTS_DIR / "exp_c_selection.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
