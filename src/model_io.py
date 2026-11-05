"""Saving and loading the model bundle used by the API.

A bundle is a plain dict so it is easy to inspect:
    model       fitted scikit-learn estimator with predict_proba
    features    ordered list of feature names the model expects
    threshold   probability cut-off for the VPN/proxy class (tuned for F1)
    reference   per-feature training medians, used as the baseline by /explain
    metadata    model name, version, metrics, MLflow run id, ...
"""

from pathlib import Path

import joblib

REQUIRED_KEYS = {"model", "features", "threshold", "reference", "metadata"}


class ModelLoadError(RuntimeError):
    """Raised when the model file is missing or not a valid bundle."""


def save_bundle(bundle: dict, path: Path) -> Path:
    missing = REQUIRED_KEYS - bundle.keys()
    if missing:
        raise ValueError(f"bundle is missing keys: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(f"model file not found: {path} (run `make train` first)")
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # corrupted or incompatible pickle
        raise ModelLoadError(f"could not load model from {path}: {exc}") from exc
    if not isinstance(bundle, dict) or REQUIRED_KEYS - bundle.keys():
        raise ModelLoadError(f"{path} is not a valid model bundle")
    if not hasattr(bundle["model"], "predict_proba"):
        raise ModelLoadError("bundled model has no predict_proba")
    try:
        threshold = float(bundle["threshold"])
    except (TypeError, ValueError) as exc:
        raise ModelLoadError(f"threshold is not a number: {bundle['threshold']!r}") from exc
    if not 0.0 < threshold < 1.0:
        raise ModelLoadError(f"invalid threshold {bundle['threshold']}")
    features = bundle["features"]
    if not isinstance(features, list | tuple) or not features or not all(isinstance(f, str) for f in features):
        raise ModelLoadError("bundle features must be a non-empty list of names")
    missing = [f for f in features if f not in bundle["reference"]]
    if missing:
        raise ModelLoadError(f"bundle reference has no median for: {missing}")
    if not isinstance(bundle["metadata"], dict) or {"model_name", "version"} - bundle["metadata"].keys():
        raise ModelLoadError("bundle metadata needs model_name and version")
    return bundle
