"""Run one configuration through cross-validation, or refit it for the final test evaluation.

A "spec" fully describes a configuration, so the final test script can rebuild it exactly:
    kind         "hgb" (baseline hgb_slow plus overrides) or "candidate" (a model from src.train)
    overrides    hgb parameters that differ from the baseline
    candidate    name in src.train.CANDIDATES
    strategy     None, "over" or "under", with ratio
    features     feature list (default: all 28)
"""

import time

import numpy as np

from experiments.common import (
    ARTIFACT_DIR,
    BASELINE_PARAMS,
    cross_val_oof,
    cv_summary,
    fold_f1s,
    hgb,
    nested_threshold_f1,
    resample,
)
from src.config import RANDOM_STATE, TARGET
from src.features import FEATURES
from src.train import CANDIDATES


def build_model(spec: dict):
    if spec.get("kind", "hgb") == "candidate":
        return CANDIDATES[spec["candidate"]]()
    return hgb(**spec.get("overrides", {}))


def describe_params(spec: dict) -> dict:
    if spec.get("kind", "hgb") == "candidate":
        return {"candidate": spec["candidate"]}
    if spec.get("kind") == "inductive":
        return {**BASELINE_PARAMS, "population_counts": "training IPs only"}
    feats = spec.get("features")
    return {**BASELINE_PARAMS, **spec.get("overrides", {}), "strategy": spec.get("strategy"),
            "ratio": spec.get("ratio"), "n_features": len(feats) if feats else len(FEATURES)}


def run_cv(name: str, experiment: str, spec: dict, train, change: str, cache_as: str | None = None) -> dict:
    """Cross-validate one spec on training data only. Out-of-fold predictions are cached."""
    run_name = f"{experiment}_{name}"
    feats = spec.get("features") or FEATURES
    n_splits = spec.get("n_splits", 5)
    y = train[TARGET].to_numpy()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"oof_{cache_as or run_name}.npz"
    if path.exists():
        cached = np.load(path)
        oof, n_iters, fit_seconds = cached["oof"], cached["n_iters"].tolist(), cached["fit_seconds"].tolist()
        print(f"  {run_name}: using cached predictions", flush=True)
    else:
        print(f"  {run_name}: {change}", flush=True)
        out = cross_val_oof(build_model(spec), train[feats], y, n_splits, spec.get("strategy"), spec.get("ratio", 1.0))
        oof, n_iters, fit_seconds = out["oof"], out["n_iters"], out["fit_seconds"]
        np.savez(path, oof=oof, n_iters=np.array(n_iters), fit_seconds=np.array(fit_seconds))
    return make_record(run_name, name, experiment, spec, y, oof, n_iters, fit_seconds, change, n_splits)


def make_record(run_name, name, experiment, spec, y, oof, n_iters, fit_seconds, change, n_splits=5) -> dict:
    summary = cv_summary(y, oof)
    per_fold = fold_f1s(y, oof, summary["threshold"], n_splits)
    metrics = {**summary, "cv_f1_fold_std": float(np.std(per_fold)),
               "cv_f1_nested": nested_threshold_f1(y, oof, n_splits),
               "mean_fit_seconds": float(np.mean(fit_seconds)),
               "mean_trees": float(np.mean(n_iters)) if len(n_iters) else None}
    print(f"    -> cv_f1={metrics['cv_f1']:.4f} (+/-{metrics['cv_f1_fold_std']:.4f}) "
          f"nested={metrics['cv_f1_nested']:.4f} p={metrics['cv_precision']:.3f} r={metrics['cv_recall']:.3f} "
          f"pr_auc={metrics['cv_pr_auc']:.4f} thr={metrics['threshold']:.3f} trees={metrics['mean_trees']} "
          f"fit={metrics['mean_fit_seconds']:.0f}s", flush=True)
    return {"run_name": run_name, "name": name, "experiment": experiment, "change": change, "spec": spec,
            "params": describe_params(spec), "metrics": metrics}


def refit_and_predict(spec: dict, train, test) -> tuple[np.ndarray, object, float]:
    """Fit on all training rows (resampled if the spec says so) and predict test probabilities."""
    feats = spec.get("features") or FEATURES
    y = train[TARGET].to_numpy()
    keep = resample(y, spec.get("strategy"), seed=RANDOM_STATE, ratio=spec.get("ratio", 1.0))
    model = build_model(spec)
    started = time.perf_counter()
    model.fit(train[feats].iloc[keep], y[keep])
    seconds = time.perf_counter() - started
    return model.predict_proba(test[feats])[:, 1], model, seconds
