"""Train candidate models, track them in MLflow, and save the best one.

Each candidate is one MLflow run. Model selection uses only the training set
(stratified k-fold cross-validation); the held-out test set is scored once per run
for reporting and is never used to choose a model or a threshold.

Primary metric: F1 of the VPN/proxy class (test_f1). Accuracy is logged too, but
is misleading here because only a few percent of IPs are VPN/proxy.

Usage:
    python -m src.train                       # all candidates
    python -m src.train --models logreg rf    # a subset
"""

import argparse
import json
import shutil
from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    TARGET,
)
from src.features import FEATURES
from src.model_io import save_bundle

CANDIDATES = {
    "logreg": lambda: make_pipeline(
        FunctionTransformer(np.log1p),  # counts are heavy-tailed
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000),
    ),
    "rf": lambda: RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE,
    ),
    "rf_shallow": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE,
    ),
    "hgb": lambda: HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=300, class_weight="balanced", random_state=RANDOM_STATE,
    ),
    "hgb_slow": lambda: HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=63, l2_regularization=1.0,
        class_weight="balanced", random_state=RANDOM_STATE,
    ),
}


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Probability threshold that maximises F1, and that F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    i = int(np.argmax(f1[:-1]))  # the last point has no threshold
    return float(thresholds[i]), float(f1[i])


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
    }


def _estimator_params(model) -> dict:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    return {k: v for k, v in est.get_params().items() if isinstance(v, (int, float, str, bool, type(None)))}


def run_candidate(name: str, X_train, y_train, X_test, y_test, cv_folds: int) -> dict:
    model = CANDIDATES[name]()
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({"model": name, "cv_folds": cv_folds, "n_features": len(FEATURES),
                           "n_train": len(y_train), **_estimator_params(model)})

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold, cv_f1 = best_f1_threshold(y_train, oof)

        model.fit(X_train, y_train)
        test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test, test_proba, threshold)

        metrics = {"cv_f1": cv_f1, "cv_pr_auc": average_precision_score(y_train, oof),
                   "threshold": threshold, **{f"test_{k}": v for k, v in test_metrics.items()}}
        mlflow.log_metrics(metrics)
        print(f"  {name:<11} cv_f1={cv_f1:.4f}  test_f1={metrics['test_f1']:.4f}  "
              f"test_pr_auc={metrics['test_pr_auc']:.4f}  threshold={threshold:.3f}")
        return {"name": name, "run_id": run.info.run_id, "model": model,
                "threshold": threshold, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X_train, y_train = train[FEATURES], train[TARGET].to_numpy()
    X_test, y_test = test[FEATURES], test[TARGET].to_numpy()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"Training on {len(train):,} IPs, testing on {len(test):,} IPs")
    results = [run_candidate(n, X_train, y_train, X_test, y_test, args.cv_folds) for n in args.models]

    # Select on cross-validated F1 (training data only), not on the test score.
    best = max(results, key=lambda r: r["metrics"]["cv_f1"])
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("best_model", "true")

    version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    metadata = {"model_name": best["name"], "version": version, "mlflow_run_id": best["run_id"],
                "trained_at": version, "n_train": len(train),
                "metrics": {k: round(float(v), 4) for k, v in best["metrics"].items()}}
    bundle = {"model": best["model"], "features": FEATURES, "threshold": best["threshold"],
              "reference": X_train.median().to_dict(), "metadata": metadata}

    versioned = save_bundle(bundle, MODELS_DIR / f"vpn-clf_{best['name']}_{version}.joblib")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(versioned, MODEL_PATH)
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(
        {**metadata, "file": versioned.name,
         "all_runs": {r["name"]: {k: round(float(v), 4) for k, v in r["metrics"].items()} for r in results}},
        indent=2))
    print(f"Best: {best['name']} (cv_f1={best['metrics']['cv_f1']:.4f}) -> {versioned} and {MODEL_PATH}")


if __name__ == "__main__":
    main()
