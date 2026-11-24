"""Experiment G: FLAML tree-based AutoML against the hand-tuned model, under the same rules.

Why this exists: Ludwig searches neural networks, so "manual trees vs Ludwig" mixes two
questions (AutoML vs manual, and trees vs neural nets). FLAML searches tree models
(LightGBM, XGBoost, random forest, extra trees) and logistic regression, so comparing it with
the hand-tuned HistGradientBoosting model isolates "AutoML vs manual".

Fairness rules, identical to experiments A-F and to the production model:
  * same prepared train/test split (the split fingerprint is recorded);
  * FLAML searches on the training set only, with the project's folds:
    StratifiedKFold(5, shuffle=True, random_state=42);
  * the winning configuration is then cross-validated on those same folds, and the decision
    threshold is tuned on its out-of-fold predictions, exactly like hgb_slow;
  * the test set is used once, at the end, at that CV threshold.

Nothing here changes or deploys the production model. The fitted FLAML model is saved to the
git-ignored artifacts folder only.

Usage: python -m experiments.exp_g_flaml [--budget 600]
Needs:  pip install -r requirements-flaml.txt
"""

import argparse
import time

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from experiments.common import (
    ARTIFACT_DIR,
    bootstrap_f1_difference,
    cross_val_oof,
    cv_summary,
    environment,
    fold_f1s,
    load_split,
    model_size_kb,
    nested_threshold_f1,
    publish,
    split_fingerprint,
    test_summary,
)
from src.config import RANDOM_STATE, ROOT, TARGET
from src.features import FEATURES

ESTIMATORS = ["lgbm", "xgboost", "rf", "extra_tree", "lrl1"]
OBJECTIVE = "ap"  # average precision (PR-AUC): threshold-free, like Ludwig v2's objective


def search(X, y, budget: int, log_file: str | None = None):
    """Run FLAML on training data only, scored by cross-validation on the project's folds."""
    from flaml import AutoML

    automl = AutoML()
    automl.fit(X, y, task="classification", metric=OBJECTIVE, time_budget=budget,
               estimator_list=ESTIMATORS, eval_method="cv",
               split_type=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
               seed=RANDOM_STATE, n_jobs=-1, log_file_name=log_file or "", verbose=1)
    return automl


def evaluate_candidate(estimator, train, test, n_splits: int = 5) -> dict:
    """Cross-validate an unfitted estimator like hgb_slow, then score the test set once."""
    X, y = train[FEATURES], train[TARGET].to_numpy()
    out = cross_val_oof(estimator, X, y, n_splits)
    summary = cv_summary(y, out["oof"])
    per_fold = fold_f1s(y, out["oof"], summary["threshold"], n_splits)
    final = clone(estimator)
    started = time.perf_counter()
    final.fit(X, y)
    refit_seconds = time.perf_counter() - started
    proba = final.predict_proba(test[FEATURES])[:, 1]
    metrics = {**summary, "cv_f1_fold_std": float(np.std(per_fold)),
               "cv_f1_nested": nested_threshold_f1(y, out["oof"], n_splits),
               "mean_fit_seconds": float(np.mean(out["fit_seconds"])), "refit_seconds": refit_seconds,
               **test_summary(test[TARGET].to_numpy(), proba, summary["threshold"])}
    return {"metrics": metrics, "model": final, "proba": proba}


def production_reference(test) -> dict:
    """Score the deployed model (models/model.joblib) on the same test set, at its own threshold."""
    from src.model_io import load_bundle

    bundle = load_bundle(ROOT / "models" / "model.joblib")
    proba = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]
    return {"name": bundle["metadata"]["model_name"], "version": bundle["metadata"]["version"],
            "threshold": float(bundle["threshold"]), "proba": proba,
            "metrics": test_summary(test[TARGET].to_numpy(), proba, float(bundle["threshold"]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budget", type=int, default=600, help="FLAML search time in seconds (default 600)")
    args = parser.parse_args()

    train, test = load_split()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"FLAML search: {args.budget}s budget, metric={OBJECTIVE}, estimators={ESTIMATORS}", flush=True)
    started = time.perf_counter()
    automl = search(train[FEATURES], train[TARGET].to_numpy(), args.budget,
                    str(ARTIFACT_DIR / "flaml_search.log"))
    search_seconds = time.perf_counter() - started
    print(f"best: {automl.best_estimator} {automl.best_config} cv_ap={1 - automl.best_loss:.4f}", flush=True)

    print("Cross-validating the winner on the project's folds, then scoring the test set once", flush=True)
    result = evaluate_candidate(clone(automl.model.estimator), train, test)
    joblib.dump(result["model"], ARTIFACT_DIR / "flaml_best.joblib", compress=3)

    prod = production_reference(test)
    y_test = test[TARGET].to_numpy()
    diff = bootstrap_f1_difference(y_test, result["proba"] >= result["metrics"]["threshold"],
                                   prod["proba"] >= prod["threshold"])

    m = result["metrics"]
    record = {
        "run_name": "exp_g_flaml_best", "name": "flaml_best", "experiment": "exp_g",
        "change": f"FLAML AutoML ({args.budget}s, {OBJECTIVE}) picked {automl.best_estimator}",
        "params": {"automl": "flaml", "flaml_budget_s": args.budget, "flaml_metric": OBJECTIVE,
                   "estimator_list": ",".join(ESTIMATORS), "best_estimator": automl.best_estimator,
                   **{f"best_{k}": v for k, v in automl.best_config.items()}},
        "metrics": {**m, "flaml_search_seconds": search_seconds, "flaml_cv_ap": 1 - automl.best_loss,
                    "flaml_logged_improvements": len(automl.config_history),
                    "model_size_kb": model_size_kb(result["model"]),
                    "test_f1_minus_production": diff["diff"], "test_f1_minus_production_low": diff["diff_low"],
                    "test_f1_minus_production_high": diff["diff_high"]},
        "best_loss_per_estimator": {k: (None if v is None else 1 - v)
                                    for k, v in automl.best_loss_per_estimator.items()},
        "production_reference": {"model": prod["name"], "version": prod["version"], "threshold": prod["threshold"],
                                 **prod["metrics"]},
        "split_fingerprint": split_fingerprint(train, test),
        "environment": {**environment(), "flaml": _version("flaml"), "lightgbm": _version("lightgbm"),
                        "xgboost": _version("xgboost")},
    }
    publish([record], "exp_g_flaml")
    print(f"\nFLAML {automl.best_estimator}: cv_f1={m['cv_f1']:.4f} nested={m['cv_f1_nested']:.4f} "
          f"test_f1={m['test_f1']:.4f} (threshold {m['threshold']:.3f})", flush=True)
    print(f"Production {prod['name']}: test_f1={prod['metrics']['test_f1']:.4f} (threshold {prod['threshold']:.3f})")
    print(f"FLAML - production test F1: {diff['diff']:+.4f} (95% CI {diff['diff_low']:+.4f} to "
          f"{diff['diff_high']:+.4f})", flush=True)


def _version(package: str) -> str:
    from importlib.metadata import version

    return version(package)


if __name__ == "__main__":
    main()
