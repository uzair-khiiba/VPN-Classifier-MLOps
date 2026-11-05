"""Log the three trained models to MLflow, reload each one, and check it predicts the same as the original.

Original model files are only read. Nothing is retrained, and models/model.joblib is never changed.

  served     hgb_slow, models/model.joblib                  sklearn flavor, plus the API bundle
  candidate  exp_c_trees_400_no_early_stop                  sklearn flavor, plus the API bundle
  ludwig     ludwig_results/best_model                      plain artifacts, because MLflow has no Ludwig flavor here

Where models go:
  --target local    (default) a separate database, so the experiment history stays byte-for-byte unchanged:
                    reports/experiments/artifacts/model_packages_mlflow.db
  --target dagshub  the run each model came from on DagsHub, found through its mirror.source_run_id tag.
                    --register also adds the sklearn models to the Model Registry. Ludwig is never registered.

The results are written to reports/experiments/results/model_logging_<target>.json.

Usage:  python -m experiments.log_models [--target local|dagshub] [--register]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import ARTIFACT_DIR, RESULTS_DIR, connect_dagshub, load_records, load_split
from experiments.package_candidate import CANDIDATE_DIR, CANDIDATE_RUN
from experiments.publish_history import MAIN_REMOTE_NAME, SOURCE_TAG, list_all_runs
from src.config import MODEL_PATH, ROOT, TARGET
from src.features import FEATURES
from src.model_io import load_bundle
from src.train import evaluate

PACKAGE_DB = ARTIFACT_DIR / "model_packages_mlflow.db"
PACKAGE_ARTIFACTS = ARTIFACT_DIR / "model_packages_mlruns"
PACKAGE_EXPERIMENT = "zainab_model_packages"
LUDWIG_DIR = ROOT / "ludwig_results" / "best_model"
REGISTERED_MODEL = "vpn-proxy-classifier"
LOGGED_TAG = "model_logged"
EQUIVALENCE_TOLERANCE = 1e-12
CHECK_ROWS = 5000
# These are our own models. MLflow requires naming the one class it can't vet automatically.
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]

LUDWIG_LOADING = """# Loading this Ludwig model

MLflow has no Ludwig flavor in this setup, so this is a plain folder of artifacts. It isn't registered, and
`mlflow.pyfunc.load_model` can't load it.

It needs the Python 3.11 environment from requirements-ludwig.txt:

    from ludwig.api import LudwigModel
    model = LudwigModel.load("ludwig_model", backend="local")
    predictions = model.predict(dataset=frame_with_the_28_features)[0]

Use the decision threshold in model_card.json. The API can't serve this model: it runs Python 3.12 without Ludwig,
and needs a model with predict_proba.
"""


def _versions(*packages: str) -> dict[str, str]:
    return {"python": platform.python_version(), **{name: version(name) for name in packages}}


def model_card(key: str, bundle: dict | None = None, record: dict | None = None, run_id: str | None = None) -> dict:
    """What someone needs to use the model correctly. Metrics are copied from saved results."""
    if key == "ludwig":
        metrics = record["metrics"]
        return {"key": key, "model_name": "ludwig_automl", "version": record["params"].get("model", "ludwig_ecd"),
                "status": "experiment, not deployed", "algorithm": "Ludwig ECD neural network, concat combiner",
                "features": FEATURES, "threshold": metrics["threshold"], "threshold_tuned_on": "validation split",
                "training_config": record["params"], "metrics": metrics,
                "metric_sources": "val_* = validation split, test_* = held-out test set",
                "environment": {"python": "3.11", "ludwig": "0.11.0", "requirements": "requirements-ludwig.txt"},
                "source_run_id": run_id}
    meta = bundle["metadata"]
    model = bundle["model"]
    return {"key": key, "model_name": meta["model_name"], "version": meta["version"],
            "status": "deployed" if key == "served" else "candidate, not deployed",
            "algorithm": type(model).__name__, "n_trees": int(np.max(model.n_iter_)), "features": bundle["features"],
            "threshold": float(bundle["threshold"]),
            "threshold_tuned_on": "out-of-fold predictions on the training set",
            "training_config": model.get_params(), "n_train": meta.get("n_train"), "metrics": meta.get("metrics", {}),
            "metric_sources": "cv_* = 5-fold CV on training IPs, test_* = held-out test set",
            "environment": _versions("scikit-learn", "numpy", "pandas", "mlflow"),
            "predict_proba_output": "probability of VPN/proxy is column 1; apply the threshold yourself",
            "source_run_id": meta.get("mlflow_run_id") or run_id}


def _write_extras(folder: Path, card: dict, extra: dict[str, str] | None = None) -> None:
    (folder / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    (folder / "features.json").write_text(json.dumps(card["features"], indent=2))
    for name, text in (extra or {}).items():
        (folder / name).write_text(text)


def log_sklearn(mlflow, client, run_id: str, key: str, bundle: dict, bundle_path: Path, X: pd.DataFrame) -> dict:
    """Log with mlflow.sklearn, keeping the run's status and end time as they were."""
    from mlflow.models import infer_signature

    run = client.get_run(run_id)
    # Only the training medians, already part of the bundle: the DagsHub repository is public, so no real IP records.
    example = pd.DataFrame([bundle["reference"]])[bundle["features"]]
    card = model_card(key, bundle, run_id=run_id)
    with mlflow.start_run(run_id=run_id):
        info = mlflow.sklearn.log_model(
            bundle["model"], name="model", signature=infer_signature(example, bundle["model"].predict_proba(example)),
            input_example=example, pyfunc_predict_fn="predict_proba", skops_trusted_types=TRUSTED_TYPES,
            metadata={"threshold": card["threshold"],
                      "features_order": "see features.json"},
            pip_requirements=[f"scikit-learn=={version('scikit-learn')}", f"numpy=={version('numpy')}",
                              f"pandas=={version('pandas')}"])
        with tempfile.TemporaryDirectory() as tmp:
            _write_extras(Path(tmp), card)
            mlflow.log_artifacts(tmp, "model_info")
        mlflow.log_artifact(str(bundle_path), "api_bundle")
    client.set_terminated(run_id, run.info.status, run.info.end_time)
    client.set_tag(run_id, f"{LOGGED_TAG}.{key}", card["version"])
    return {"key": key, "run_id": run_id, "model_uri": info.model_uri, "logged_model_id": info.model_id,
            "method": "mlflow.sklearn.log_model", "threshold": card["threshold"], "version": card["version"],
            "bundle_artifact": f"runs:/{run_id}/api_bundle/{bundle_path.name}"}


def check_sklearn(mlflow, client, entry: dict, bundle: dict, X: pd.DataFrame, y: np.ndarray) -> dict:
    """Download what's logged and compare it with the original, as sklearn, pyfunc and the API bundle."""
    rows = X.head(CHECK_ROWS)
    expected = bundle["model"].predict_proba(rows)[:, 1]
    sklearn_proba = mlflow.sklearn.load_model(entry["model_uri"]).predict_proba(rows)[:, 1]
    pyfunc_proba = np.asarray(mlflow.pyfunc.load_model(entry["model_uri"]).predict(rows))[:, 1]
    with tempfile.TemporaryDirectory() as tmp:
        path = client.download_artifacts(entry["run_id"], entry["bundle_artifact"].split("/", 2)[2], tmp)
        restored = load_bundle(path)
    bundle_proba = restored["model"].predict_proba(rows)[:, 1]
    differences = {"sklearn": float(np.abs(sklearn_proba - expected).max()),
                   "pyfunc": float(np.abs(pyfunc_proba - expected).max()),
                   "api_bundle": float(np.abs(bundle_proba - expected).max())}
    return {"rows_compared": len(rows), "max_abs_difference": differences,
            "predictions_equal": all(value <= EQUIVALENCE_TOLERANCE for value in differences.values()),
            "bundle_features_match": restored["features"] == bundle["features"],
            "bundle_threshold_matches": float(restored["threshold"]) == float(bundle["threshold"]),
            "test_f1_from_downloaded_bundle": evaluate(y, restored["model"].predict_proba(X)[:, 1],
                                                       restored["threshold"])["f1"]}


def log_ludwig(mlflow, client, run_id: str, record: dict) -> dict:
    card = model_card("ludwig", record=record, run_id=run_id)
    with tempfile.TemporaryDirectory() as tmp:
        _write_extras(Path(tmp), card, {"LOADING.md": LUDWIG_LOADING})
        client.log_artifacts(run_id, tmp, "model_info")
    client.log_artifacts(run_id, str(LUDWIG_DIR), "ludwig_model")
    client.set_tag(run_id, f"{LOGGED_TAG}.ludwig", card["version"])
    return {"key": "ludwig", "run_id": run_id, "model_uri": f"runs:/{run_id}/ludwig_model",
            "method": "plain artifacts (no MLflow flavor)", "threshold": card["threshold"], "version": card["version"]}


def check_ludwig(client, entry: dict) -> dict:
    """Download the logged folder and let the Ludwig environment score the test set with it."""
    python = ROOT / ".venv-ludwig" / "Scripts" / "python.exe"
    if not python.exists():
        python = ROOT / ".venv-ludwig" / "bin" / "python"
    if not python.exists():
        return {"predictions_equal": None, "note": "the Ludwig environment isn't installed, so this wasn't checked"}
    with tempfile.TemporaryDirectory() as tmp:
        folder = client.download_artifacts(entry["run_id"], "ludwig_model", tmp)
        done = subprocess.run([str(python), "-m", "experiments.check_ludwig_artifact", folder], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"})
    if done.returncode:
        return {"predictions_equal": False, "error": done.stderr[-600:]}
    return json.loads(done.stdout.strip().splitlines()[-1])


def register(client, entry: dict, alias: str, card_status: str) -> dict:
    """Add a model version and point an alias at it. Repeating this doesn't create another version."""
    if not client.search_registered_models(filter_string=f"name='{REGISTERED_MODEL}'"):
        client.create_registered_model(REGISTERED_MODEL, description="VPN/proxy classifiers. Deployment only changes "
                                                                     "through experiments.promote_model.")
    existing = [v for v in client.search_model_versions(f"name='{REGISTERED_MODEL}'") if v.run_id == entry["run_id"]]
    version_info = existing[0] if existing else client.create_model_version(
        REGISTERED_MODEL, source=entry["model_uri"], run_id=entry["run_id"],
        tags={"status": card_status, "threshold": str(entry["threshold"]), "model_version": entry["version"]})
    client.set_registered_model_alias(REGISTERED_MODEL, alias, version_info.version)
    return {"registered_model": REGISTERED_MODEL, "version": version_info.version, "alias": alias}


def _local_run(client, key: str, experiment_id: str, source: str) -> str:
    """One run per package, reused on repeat so the separate database doesn't collect duplicates."""
    name = f"package_{key}"
    for run in list_all_runs(client, experiment_id):
        if run.info.run_name == name:
            return run.info.run_id
    run = client.create_run(experiment_id, run_name=name, tags={"source_run": source})
    client.set_terminated(run.info.run_id, "FINISHED")
    return run.info.run_id


def _find_remote_run(client, experiment_name: str, source_run_id: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"{experiment_name} isn't on the server. Run experiments.publish_history first.")
    for run in list_all_runs(client, experiment.experiment_id):
        if run.data.tags.get(SOURCE_TAG) == source_run_id:
            return run.info.run_id
    raise RuntimeError(f"no copy of run {source_run_id} in {experiment_name}. Run experiments.publish_history first.")


def _local_source_ids() -> dict[str, str]:
    """Local run ids of the two experiment runs, read through a temporary copy of the database."""
    import shutil
    import sqlite3

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "read.db"
        shutil.copy2(ARTIFACT_DIR / "zainab_mlflow.db", copy)
        con = sqlite3.connect(copy)
        rows = con.execute("SELECT r.name, r.run_uuid FROM runs r JOIN experiments e USING (experiment_id) "
                           "WHERE e.name='zainab_experiments' AND r.name IN (?, ?)",
                           (CANDIDATE_RUN, "ludwig_automl")).fetchall()
        con.close()
    return dict(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=("local", "dagshub"), default="local")
    parser.add_argument("--register", action="store_true", help="add the sklearn models to the Model Registry")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from mlflow.tracking import MlflowClient

    _, test = load_split()
    X, y = test[FEATURES], test[TARGET].to_numpy()
    served_path = MODEL_PATH
    candidate_path = CANDIDATE_DIR / f"{CANDIDATE_RUN}.joblib"
    if not candidate_path.exists():
        raise SystemExit("Candidate bundle missing. Run: python -m experiments.package_candidate")
    bundles = {"served": (load_bundle(served_path), served_path),
               "candidate": (load_bundle(candidate_path), candidate_path)}
    ludwig_record = load_records("exp_f")[0]
    source_ids = _local_source_ids()

    if args.target == "dagshub":
        uri = connect_dagshub()
        client = MlflowClient(tracking_uri=uri)
        runs = {"served": bundles["served"][0]["metadata"]["mlflow_run_id"],
                "candidate": _find_remote_run(client, MAIN_REMOTE_NAME, source_ids[CANDIDATE_RUN]),
                "ludwig": _find_remote_run(client, MAIN_REMOTE_NAME, source_ids["ludwig_automl"])}
    else:
        uri = f"sqlite:///{PACKAGE_DB.as_posix()}"
        mlflow.set_tracking_uri(uri)
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(PACKAGE_EXPERIMENT)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(
            PACKAGE_EXPERIMENT, artifact_location=PACKAGE_ARTIFACTS.as_uri())
        served_run = bundles["served"][0]["metadata"]["mlflow_run_id"]
        runs = {"served": _local_run(client, "served", experiment_id, served_run),
                "candidate": _local_run(client, "candidate", experiment_id, source_ids[CANDIDATE_RUN]),
                "ludwig": _local_run(client, "ludwig", experiment_id, source_ids["ludwig_automl"])}
    mlflow.set_tracking_uri(uri)

    path = RESULTS_DIR / f"model_logging_{args.target}.json"
    previous = {e["key"]: e for e in json.loads(path.read_text())["models"]} if path.exists() else {}
    entries = []
    for key, run_id in runs.items():
        print(f"{key}: run {run_id}", flush=True)
        already = client.get_run(run_id).data.tags.get(f"{LOGGED_TAG}.{key}")
        if already and key in previous and previous[key]["run_id"] == run_id:
            entry = previous[key]
        elif key == "ludwig":
            entry = log_ludwig(mlflow, client, run_id, ludwig_record)
        else:
            entry = log_sklearn(mlflow, client, run_id, key, *bundles[key], X)
        if key == "ludwig":
            entry["reload_check"] = check_ludwig(client, entry)
        else:
            entry["reload_check"] = check_sklearn(mlflow, client, entry, bundles[key][0], X, y)
            if args.register:
                alias, status = ("production", "deployed") if key == "served" else ("candidate", "not deployed")
                try:
                    entry["registry"] = register(client, entry, alias, status)
                except Exception as exc:
                    entry["registry"] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
        print(json.dumps(entry, indent=1, default=str)[:1200], flush=True)
        entries.append(entry)
    path.write_text(json.dumps({"tracking_uri": uri, "target": args.target, "models": entries}, indent=2, default=str))
    failed = [e["key"] for e in entries if e["reload_check"].get("predictions_equal") is False]
    print(f"Wrote {path}. Failed reload checks: {failed or 'none'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
