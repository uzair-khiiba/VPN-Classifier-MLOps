"""Stage 2: lock the selections made on cross-validation, THEN score on the test set.

The order matters. selections.json is written before any test prediction, so no choice can
depend on test results. Test scores of the configurations that were not selected are
retrospective: they are reported for learning and must not be used to pick a model.

Usage: python -m experiments.final_test_eval
"""

import hashlib
import json

import joblib
import numpy as np

from experiments.common import (
    ARTIFACT_DIR,
    RESULTS_DIR,
    bootstrap_f1_difference,
    environment,
    hgb,
    load_config,
    load_records,
    load_split,
    model_size_kb,
    save_records,
    split_fingerprint,
    test_summary,
)
from experiments.exp_d_features import inductive_counts, inductive_frame, load_lookups
from experiments.runner import refit_and_predict
from src.config import ID_COLUMN, TARGET
from src.features import FEATURES


def selections() -> dict:
    """Apply each experiment's rule using CV results only."""
    margin = load_config()["hyperparameters"]["min_cv_f1_gain"]
    exp_b = load_records("exp_b")
    baseline_b = next(r for r in exp_b if r["name"] == "class_weight_balanced")
    best_b = max(exp_b, key=lambda r: r["metrics"]["cv_f1"])
    b_gain = best_b["metrics"]["cv_f1"] - baseline_b["metrics"]["cv_f1"]
    exp_e = load_records("exp_e")
    tuned_a = next(r for r in load_records("exp_a") if r["name"] == "tuned_threshold")

    # Overall rule (see exp_c2_followup): best 5-fold CV F1, beat the baseline by the margin,
    # and within the margin of the best, prefer no resampling, then fewer trees.
    pool = [r for experiment in ("exp_b", "exp_c", "exp_c2") for r in load_records(experiment)
            if not r.get("screening_only")]
    best_f1 = max(r["metrics"]["cv_f1"] for r in pool)
    near_best = [r for r in pool if best_f1 - r["metrics"]["cv_f1"] <= margin]
    simplest = min(near_best, key=lambda r: (r["spec"].get("strategy") is not None, r["metrics"]["mean_trees"]))
    overall_gain = simplest["metrics"]["cv_f1"] - baseline_b["metrics"]["cv_f1"]
    return {
        "overall": {"selected_run": simplest["run_name"] if overall_gain > margin else baseline_b["run_name"],
                    "best_cv_f1_in_pool": best_f1, "near_best": [r["run_name"] for r in near_best],
                    "gain_vs_baseline": overall_gain, "required_gain": margin,
                    "rule": "highest 5-fold CV F1, must beat baseline by margin, simplest within margin of best"},
        "exp_a": {"selected": "tuned_threshold", "threshold": tuned_a["metrics"]["threshold"],
                  "rule": "threshold with the highest out-of-fold F1"},
        "exp_b": {"selected": best_b["name"] if b_gain > margin else baseline_b["name"], "best_by_cv": best_b["name"],
                  "gain": b_gain, "required_gain": margin, "rule": "best CV F1, must beat the baseline by the margin"},
        "exp_c": json.loads((RESULTS_DIR / "exp_c_selection.json").read_text()),
        "exp_d": {"selected": None, "rule": "analysis only: the feature sets are compared, not selected"},
        "exp_e": {"selected": max(exp_e, key=lambda r: r["metrics"]["cv_f1"])["name"], "rule": "best CV F1"},
    }


def spec_key(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def main() -> None:
    locked = selections()
    (RESULTS_DIR / "selections.json").write_text(json.dumps(locked, indent=2))
    print("selections locked before test scoring:\n" + json.dumps(locked, indent=2), flush=True)

    train, test = load_split()
    y_test = test[TARGET].to_numpy()
    cache: dict[str, dict] = {}
    models_dir = ARTIFACT_DIR / "selected_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    def predict(spec: dict) -> dict:
        key = spec_key(spec)
        if key not in cache:
            proba, model, seconds = refit_and_predict(spec, train, test)
            np.save(ARTIFACT_DIR / f"test_proba_{key}.npy", proba)
            cache[key] = {"proba": proba, "seconds": seconds, "size": model_size_kb(model), "model": model}
        return cache[key]

    base_spec = {"kind": "hgb", "overrides": {}}
    base = predict(base_spec)
    served = joblib.load(ARTIFACT_DIR / "baseline" / "model.joblib")
    same = np.allclose(served["model"].predict_proba(test[FEATURES])[:, 1], base["proba"])
    print(f"refit baseline identical to reproduced model: {same}", flush=True)
    if not same:
        raise RuntimeError("the refit baseline differs from the reproduced model; investigate before comparing")

    all_records = {}
    for experiment in ("exp_a", "exp_b", "exp_c", "exp_c2", "exp_d", "exp_e"):
        records = load_records(experiment)
        for rec in records:
            if rec.get("screening_only"):
                continue
            threshold = rec["metrics"]["threshold"]
            if rec["spec"].get("kind") == "inductive":
                continue  # scored separately below
            result = predict(rec["spec"] if experiment != "exp_a" else base_spec)
            rec["metrics"].update(test_summary(y_test, result["proba"], threshold))
            rec["metrics"].update(refit_seconds=result["seconds"], model_size_kb=result["size"])
            rec["test_proba_key"] = spec_key(rec["spec"] if experiment != "exp_a" else base_spec)
            print(f"  {rec['run_name']:<46} cv_f1={rec['metrics']['cv_f1']:.4f} "
                  f"test_f1={rec['metrics']['test_f1']:.4f}", flush=True)
        all_records[experiment] = records

    # Inductive evaluation: test IPs are counted against training IPs only, as new arrivals would be.
    # The threshold was tuned where counts come from 64% of IPs; a final model's counts come from 80%.
    prints, pairs = load_lookups()
    inductive_probas = {}
    for rec in (r for r in all_records["exp_d"] if r["spec"].get("kind") == "inductive"):
        fit_table, test_table = inductive_counts(train[ID_COLUMN].to_numpy(), test[ID_COLUMN].to_numpy(), prints, pairs,
                                                 rec["spec"].get("aligned", False))
        model = hgb()
        model.fit(inductive_frame(train, fit_table), train[TARGET].to_numpy())
        proba = model.predict_proba(inductive_frame(test, test_table))[:, 1]
        np.save(ARTIFACT_DIR / f"test_proba_{rec['name']}.npy", proba)
        rec["metrics"].update(test_summary(y_test, proba, rec["metrics"]["threshold"]))
        inductive_probas[rec["name"]] = (proba, rec["metrics"]["threshold"])
        print(f"  {rec['name']}: cv_f1={rec['metrics']['cv_f1']:.4f} test_f1={rec['metrics']['test_f1']:.4f}",
              flush=True)
    for experiment, records in all_records.items():
        save_records(experiment, records)

    # Is a difference in test F1 real or chance? Paired bootstrap against the baseline.
    base_rec = next(r for r in all_records["exp_e"] if r["name"] == "hgb_slow")
    base_pred = base["proba"] >= base_rec["metrics"]["threshold"]
    comparisons = {}
    candidates = [(f"exp_d/{name}", proba, threshold) for name, (proba, threshold) in inductive_probas.items()]
    for experiment in ("exp_b", "exp_c", "exp_c2", "exp_d", "exp_e"):
        for r in all_records[experiment]:
            if r.get("screening_only") or r["spec"].get("kind") == "inductive" or r.get("test_proba_key") is None:
                continue
            candidates.append((f"{experiment}/{r['name']}", cache[r["test_proba_key"]]["proba"],
                               r["metrics"]["threshold"]))
    for name, proba, threshold in candidates:
        comparisons[name] = bootstrap_f1_difference(y_test, proba >= threshold, base_pred)
    baseline_ci = bootstrap_f1_difference(y_test, base_pred, base_pred)
    out = {"baseline_test_f1_ci95": [baseline_ci["a_low"], baseline_ci["a_high"]], "vs_baseline": comparisons,
           "split_fingerprint": split_fingerprint(train, test), "environment": environment()}
    (RESULTS_DIR / "test_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"baseline test F1 95% interval: {out['baseline_test_f1_ci95']}", flush=True)

    # Keep the selected models as local artifacts (never in models/).
    chosen = [next(r for r in all_records[experiment] if r["name"] == locked[experiment]["selected"])
              for experiment in ("exp_b", "exp_c", "exp_e")]
    overall = locked["overall"]["selected_run"]
    chosen += [r for records in all_records.values() for r in records if r["run_name"] == overall]
    for rec in chosen:
        path = models_dir / f"{rec['run_name']}.joblib"
        joblib.dump(cache[rec["test_proba_key"]]["model"], path, compress=3)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
