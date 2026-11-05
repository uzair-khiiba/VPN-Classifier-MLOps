"""Shared helpers for the experiment track. Nothing here changes the main pipeline.

Rules these helpers enforce:
  * Same data: the prepared train/test tables, refused if they look like the synthetic sample.
  * Same folds: StratifiedKFold(5, shuffle=True, random_state=42), identical to src/train.py.
  * Resampling happens only on the training part of each fold, never on validation or test.
  * Results are saved to reports/experiments/results/ BEFORE MLflow logging, so a network
    failure can never lose a long computation. `python -m experiments.sync_mlflow` retries.
"""

import hashlib
import json
import os
import platform
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from src.config import ID_COLUMN, PROCESSED_DIR, RANDOM_STATE, ROOT
from src.train import best_f1_threshold, evaluate

EXPERIMENT_NAME = "zainab_experiments"
REPORT_DIR = ROOT / "reports" / "experiments"
RESULTS_DIR = REPORT_DIR / "results"
FIGURES_DIR = REPORT_DIR / "figures"
ARTIFACT_DIR = REPORT_DIR / "artifacts"  # models and cached predictions, git-ignored by its own .gitignore
CONFIG_PATH = ROOT / "configs" / "experiments" / "experiments.yaml"
MIN_REAL_IPS = 100_000  # the synthetic sample has only 300 IPs

BASELINE_PARAMS = {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 63, "l2_regularization": 1.0,
                   "class_weight": "balanced", "random_state": RANDOM_STATE}


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def hgb(**overrides) -> HistGradientBoostingClassifier:
    """The baseline hgb_slow model, with only the named parameters changed."""
    return HistGradientBoostingClassifier(**{**BASELINE_PARAMS, **overrides})


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    if len(train) + len(test) < MIN_REAL_IPS:
        raise RuntimeError(f"{PROCESSED_DIR} holds only {len(train) + len(test)} IPs: this is not the real data")
    if set(train[ID_COLUMN]) & set(test[ID_COLUMN]):
        raise RuntimeError("IP overlap between train and test")
    return train, test


def split_fingerprint(train: pd.DataFrame, test: pd.DataFrame) -> str:
    """Hash of the test IP list, recorded with every run to prove the split never changed."""
    return hashlib.sha256(np.sort(test[ID_COLUMN].to_numpy()).tobytes()).hexdigest()[:16]


def folds(y: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(y)), y))


def resample(y: np.ndarray, strategy: str | None, seed: int, ratio: float = 1.0) -> np.ndarray:
    """Row indices of a resampled training fold. `ratio` is positives per negative afterwards."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    if strategy is None:
        return np.arange(len(y))
    if strategy == "over":
        extra = rng.choice(pos, size=max(int(len(neg) * ratio) - len(pos), 0), replace=True)
        chosen = np.concatenate([pos, extra, neg])
    elif strategy == "under":
        chosen = np.concatenate([pos, rng.choice(neg, size=int(len(pos) / ratio), replace=False)])
    else:
        raise ValueError(f"unknown resampling strategy {strategy}")
    return rng.permutation(chosen)


def cross_val_oof(model, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, strategy: str | None = None,
                  ratio: float = 1.0) -> dict:
    """Out-of-fold probabilities. Each validation fold is untouched real data; any resampling
    is applied to that fold's training rows only."""
    oof = np.zeros(len(y))
    n_iters, fit_seconds = [], []
    for i, (fit_idx, val_idx) in enumerate(folds(y, n_splits)):
        keep = resample(y[fit_idx], strategy, seed=RANDOM_STATE + i, ratio=ratio)
        est = clone(model)
        started = time.perf_counter()
        est.fit(X.iloc[fit_idx[keep]], y[fit_idx[keep]])
        fit_seconds.append(time.perf_counter() - started)
        oof[val_idx] = est.predict_proba(X.iloc[val_idx])[:, 1]
        if hasattr(est, "n_iter_"):
            n_iters.append(int(np.max(est.n_iter_)))
        print(f"    fold {i + 1}/{n_splits}: {fit_seconds[-1]:.0f}s", flush=True)
    return {"oof": oof, "n_iters": n_iters, "fit_seconds": fit_seconds}


def counts(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, proba >= threshold, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def cv_summary(y: np.ndarray, oof: np.ndarray) -> dict:
    """Threshold tuned on out-of-fold predictions, with the metrics at that threshold."""
    threshold, f1 = best_f1_threshold(y, oof)
    metrics = evaluate(y, oof, threshold)
    return {"threshold": threshold, "cv_f1": f1, "cv_precision": metrics["precision"], "cv_recall": metrics["recall"],
            "cv_pr_auc": average_precision_score(y, oof), "cv_roc_auc": metrics["roc_auc"],
            **{f"cv_{k}": v for k, v in counts(y, oof, threshold).items()}}


def test_summary(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    return {**{f"test_{k}": v for k, v in evaluate(y, proba, threshold).items()},
            **{f"test_{k}": v for k, v in counts(y, proba, threshold).items()}}


def model_size_kb(model) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.joblib"
        joblib.dump(model, path, compress=3)
        return round(path.stat().st_size / 1024, 1)


def environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(), "random_state": RANDOM_STATE,
            **{p: version(p) for p in ("scikit-learn", "numpy", "pandas", "duckdb", "mlflow")}}


def save_records(experiment: str, records: list[dict]) -> Path:
    """Write one experiment's results to JSON. Only JSON-friendly values belong in a record."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{experiment}.json"
    path.write_text(json.dumps(records, indent=2, default=_json_default))
    return path


def load_records(experiment: str) -> list[dict]:
    return json.loads((RESULTS_DIR / f"{experiment}.json").read_text())


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def fold_f1s(y: np.ndarray, oof: np.ndarray, threshold: float, n_splits: int = 5) -> list[float]:
    """F1 of each validation fold at one threshold: shows how much F1 moves by chance alone."""
    return [float(f1_score(y[val], oof[val] >= threshold, zero_division=0)) for _, val in folds(y, n_splits)]


def nested_threshold_f1(y: np.ndarray, oof: np.ndarray, n_splits: int = 5) -> float:
    """Honest CV F1: each fold is scored with a threshold tuned on the OTHER folds only."""
    tp = fp = fn = 0
    for fit, val in folds(y, n_splits):
        threshold, _ = best_f1_threshold(y[fit], oof[fit])
        c = counts(y[val], oof[val], threshold)
        tp, fp, fn = tp + c["tp"], fp + c["fp"], fn + c["fn"]
    return 2 * tp / (2 * tp + fp + fn)


def bootstrap_f1_difference(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n: int = 2000) -> dict:
    """Paired bootstrap on the test set: F1(a) - F1(b), resampling the same IPs for both models."""
    rng = np.random.default_rng(RANDOM_STATE)

    def f1(pred, idx):
        tp = np.sum(pred[idx] & (y[idx] == 1))
        return 2 * tp / max(pred[idx].sum() + (y[idx] == 1).sum(), 1)

    diffs, a_vals = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        a = f1(pred_a, idx)
        a_vals.append(a)
        diffs.append(a - f1(pred_b, idx))
    return {"diff": float(np.mean(diffs)), "diff_low": float(np.percentile(diffs, 2.5)),
            "diff_high": float(np.percentile(diffs, 97.5)),
            "a_low": float(np.percentile(a_vals, 2.5)), "a_high": float(np.percentile(a_vals, 97.5))}


def publish(records: list[dict], experiment: str) -> None:
    """Save results to disk first, then log them to MLflow. A logging failure is reported loudly."""
    path = save_records(experiment, records)
    print(f"saved {path}", flush=True)
    try:
        uri = connect_tracking()
        logged, skipped = log_records(records, experiment)
        print(f"MLflow {uri}: {logged} runs logged, {skipped} already there", flush=True)
    except Exception as exc:
        print(f"!!! MLFLOW LOGGING FAILED ({type(exc).__name__}: {exc}). Results are safe in {path}. "
              "Retry with: python -m experiments.sync_mlflow", flush=True)


def connect_tracking() -> str:
    """Point MLflow at a separate local database (default), or at DagsHub with EXPERIMENT_TRACKING=dagshub.

    The default is local because the name `zainab_experiments` is currently unusable on the
    DagsHub server (see reports/experiments/final_experiment_report.md). Credentials come from
    the token dagshub.init() already cached on this machine; they are
    never read from, or written to, this repository.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")  # fail fast instead of hanging on a slow server
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
    import mlflow

    if os.getenv("EXPERIMENT_TRACKING", "local") == "local":
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{ARTIFACT_DIR / 'zainab_mlflow.db'}")
        if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
            mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(ARTIFACT_DIR / "mlruns").as_uri())
    else:
        import dagshub
        import dagshub.common.config as dagshub_config

        dagshub_config.http_timeout = 120  # the default 30 s is too short for this repo's API
        for attempt in range(1, 5):  # DagsHub's API returns occasional 502s and timeouts
            try:
                dagshub.init(repo_owner="uzair-khiiba", repo_name="VPN-Classifier-MLOps", mlflow=True)
                break
            except Exception as exc:
                print(f"  dagshub.init attempt {attempt}/4 failed: {type(exc).__name__}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(30 * attempt)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.get_tracking_uri()


def log_records(records: list[dict], experiment: str) -> tuple[int, int]:
    """Log result records as MLflow runs under zainab_experiments.

    A run whose name already exists is skipped rather than duplicated. Failures are raised to
    the caller: the JSON results stay on disk, and experiments.sync_mlflow can retry.
    """
    import mlflow

    existing = {r.info.run_name: r.info.run_id
                for r in mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], output_format="list")}
    logged = skipped = 0
    for rec in records:
        name = rec["run_name"]
        metrics = {k: float(v) for k, v in rec.get("metrics", {}).items() if v is not None}
        if name in existing:  # add metrics computed later (test scores), never overwrite a run's history
            run = mlflow.get_run(existing[name])
            missing = {k: v for k, v in metrics.items() if k not in run.data.metrics}
            if missing:
                with mlflow.start_run(run_id=existing[name]):
                    mlflow.log_metrics(missing)
            if rec.get("artifacts"):  # refresh files that were regenerated, such as figures
                with mlflow.start_run(run_id=existing[name]):
                    for artifact in rec["artifacts"]:
                        mlflow.log_artifact(str(ROOT / artifact))
            skipped += 1
            continue
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"experiment_track": experiment, "owner": "zainab", "description": rec.get("change", "")})
            params = {k: v for k, v in rec.get("params", {}).items() if v is not None}
            if params:
                mlflow.log_params({k[:250]: str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics(metrics)
            for artifact in rec.get("artifacts", []):
                path = ROOT / artifact
                if path.exists():
                    mlflow.log_artifact(str(path))
                else:
                    print(f"  WARNING: artifact not found, not logged: {artifact}", flush=True)
        logged += 1
    return logged, skipped
