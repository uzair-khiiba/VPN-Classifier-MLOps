"""Wrap the selected experimental estimator in the bundle format the API can load, without deploying it.

The experiment saved a bare estimator. The API also needs the feature order, threshold and training medians. The
estimator isn't changed or retrained, and its test results must match the saved record or this stops.

Writes models/candidates/<run name>.joblib (git-ignored). models/model.joblib is never touched:
experiments.promote_model is the only way to deploy a candidate.

Usage: python -m experiments.package_candidate
"""

import json

import joblib
import numpy as np
import pandas as pd

from experiments.common import ARTIFACT_DIR, load_records, load_split, test_summary
from src.config import MODELS_DIR, TARGET
from src.features import FEATURES
from src.model_io import load_bundle, save_bundle

CANDIDATE_RUN = "exp_c_trees_400_no_early_stop"
CANDIDATE_DIR = MODELS_DIR / "candidates"
TOLERANCE = 1e-9


def build_candidate(estimator, record: dict, train: pd.DataFrame) -> dict:
    version = f"experiment-{record['test_proba_key']}"
    metadata = {"model_name": record["run_name"], "version": version, "status": "candidate, not deployed",
                "source_experiment": record["experiment"],
                "selected_by": "highest 5-fold CV F1 within margin, simplest",
                "params": record["params"], "n_train": len(train), "metrics": record["metrics"]}
    return {"model": estimator, "features": FEATURES, "threshold": float(record["metrics"]["threshold"]),
            "reference": train[FEATURES].median().to_dict(), "metadata": metadata}


def check_against_record(bundle: dict, record: dict, test: pd.DataFrame) -> dict:
    """Rescore the test set and require the saved metrics back, so a mismatched file can't be packaged."""
    proba = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]
    scored = test_summary(test[TARGET].to_numpy(), proba, bundle["threshold"])
    for key in ("test_f1", "test_precision", "test_recall", "test_pr_auc", "test_roc_auc", "test_tp", "test_fp"):
        if abs(scored[key] - record["metrics"][key]) > TOLERANCE:
            raise RuntimeError(f"{key}: rescored {scored[key]}, saved {record['metrics'][key]}")
    saved = ARTIFACT_DIR / f"test_proba_{record['test_proba_key']}.npy"
    if saved.exists() and not np.allclose(proba, np.load(saved)):
        raise RuntimeError(f"predictions differ from {saved.name}")
    return scored


def main() -> None:
    record = next(r for r in load_records("exp_c") if r["run_name"] == CANDIDATE_RUN)
    estimator = joblib.load(ARTIFACT_DIR / "selected_models" / f"{CANDIDATE_RUN}.joblib")
    train, test = load_split()
    if list(estimator.feature_names_in_) != FEATURES:
        raise RuntimeError("the estimator's feature order differs from src/features.py")
    bundle = build_candidate(estimator, record, train)
    scored = check_against_record(bundle, record, test)
    path = save_bundle(bundle, CANDIDATE_DIR / f"{CANDIDATE_RUN}.joblib")
    load_bundle(path)
    (CANDIDATE_DIR / f"{CANDIDATE_RUN}.json").write_text(json.dumps(
        {**bundle["metadata"], "threshold": bundle["threshold"], "features": FEATURES}, indent=2))
    print(f"Saved {path}: test F1 {scored['test_f1']:.4f} matches the saved record. Not deployed.")


if __name__ == "__main__":
    main()
