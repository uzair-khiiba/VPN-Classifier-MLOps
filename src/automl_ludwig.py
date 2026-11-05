"""Ludwig hyperparameter search over one tabular neural network (ECD architecture).

This tunes the settings of a fixed model family. It doesn't compare Random Forest, Logistic Regression or other
families, and it doesn't choose features. It uses the same per-IP features and held-out test IPs as src/train.py.

Steps:
  1. Split the training IPs into train/validation (stratified). The test set is not given to the search.
  2. Ludwig hyperopt (Ray Tune + hyperopt) searches `hyperopt.parameters`, scoring each trial on the configured
     validation metric. It first draws n_initial_points random trials, then TPE proposes the rest.
  3. Retrain the best trial's settings as the trials ran (no early stopping; Ludwig keeps the best epoch).
  4. Tune the F1 threshold on validation only, then score the test set once.
  5. Log a parent MLflow run with one nested run per trial, including failed and retried trials.

Results go to a new folder. A folder with results in it is refused, so earlier results can't be overwritten.

Usage (needs the Ludwig environment, see requirements-ludwig.txt):
    python -m src.automl_ludwig                                        # full search from the config
    python -m src.automl_ludwig --num-samples 3 --n-initial-points 2 --output-dir ludwig_results_smoke   # quick check
"""

import argparse
import copy
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, PROCESSED_DIR, RANDOM_STATE, ROOT, TARGET
from src.features import FEATURES
from src.ludwig_trials import log_trials, read_events, summarise_events, trial_rows
from src.train import best_f1_threshold, evaluate

CONFIG_PATH = ROOT / "configs" / "ludwig_automl_v2.yaml"
OUTPUT_DIR = ROOT / "ludwig_results_v2"
FIRST_SEARCH_DIR = ROOT / "ludwig_results"  # the search already reported
DEFAULT_STARTUP_POINTS = 20  # Ray's HyperOptSearch default, used when the config doesn't set n_initial_points
EVENTS_FILE = "trial_events.jsonl"


def load_config(path: Path = CONFIG_PATH) -> dict:
    config = yaml.safe_load(path.read_text())
    names = [f["name"] for f in config["input_features"]]
    missing, extra = set(FEATURES) - set(names), set(names) - set(FEATURES)
    if missing or extra or len(names) != len(set(names)):
        raise ValueError(f"{path} input_features out of sync with src/features.py "
                         f"(missing={sorted(missing)}, extra={sorted(extra)}, or duplicates)")
    return config


def apply_params(config: dict, params: dict) -> dict:
    """Write hyperopt-style dotted params (e.g. 'trainer.learning_rate',
    'label.loss.positive_class_weight') into a copy of the config."""
    config = copy.deepcopy(config)
    config.pop("hyperopt", None)
    features = {f["name"]: f for f in config["input_features"] + config["output_features"]}
    for key, value in params.items():
        head, *rest = key.split(".")
        node = features[head] if head in features else config.setdefault(head, {})
        for part in rest[:-1]:
            node = node.setdefault(part, {})
        node[rest[-1]] = value.item() if isinstance(value, np.generic) else value
    return config


def positive_proba(predictions: pd.DataFrame) -> np.ndarray:
    """Probability of label=True from a Ludwig predictions frame (column names vary by version)."""
    if hasattr(predictions, "compute"):  # Dask frame from Ludwig's Ray backend
        predictions = predictions.compute()
    for col in (f"{TARGET}_probabilities_True", f"{TARGET}_probabilities_1"):
        if col in predictions:
            return predictions[col].to_numpy(dtype=float)
    if f"{TARGET}_probabilities" in predictions:
        return np.array([p[-1] for p in predictions[f"{TARGET}_probabilities"]], dtype=float)
    prob, pred = predictions[f"{TARGET}_probability"], predictions[f"{TARGET}_predictions"].astype(bool)
    return np.where(pred, prob, 1 - prob).astype(float)


def search_budget(config: dict) -> dict:
    """How many trials are random and how many can be adaptive. TPE only starts after the random ones."""
    total = config["hyperopt"]["executor"]["num_samples"]
    startup = config["hyperopt"]["search_alg"].get("n_initial_points", DEFAULT_STARTUP_POINTS)
    return {"num_samples": total, "n_initial_points": startup, "random_trials": min(startup, total),
            "adaptive_trials": max(total - startup, 0)}


def final_training_config(config: dict, best_params: dict) -> dict:
    """The best trial's settings, trained the way trials were: Ludwig disables early stopping during a
    scheduled search, so it's disabled here too. Ludwig still restores the best validation epoch at the end."""
    final = apply_params(config, best_params)
    final["trainer"]["early_stop"] = -1
    return final


def _trial_logger(path: Path):
    from ray.tune import Callback

    class TrialEventLog(Callback):
        """Writes each trial's start, error, retry and completion as it happens, so failures aren't lost."""

        def _write(self, event: str, trial) -> None:
            config = {k: v.item() if isinstance(v, np.generic) else v for k, v in dict(trial.config).items()
                      if isinstance(v, int | float | str | np.generic)}
            with path.open("a", encoding="utf-8") as out:
                out.write(json.dumps({"event": event, "trial_id": trial.trial_id, "config": config}) + "\n")

        def on_trial_start(self, iteration, trials, trial, **info):
            self._write("start", trial)

        def on_trial_error(self, iteration, trials, trial, **info):
            self._write("error", trial)

        def on_trial_recover(self, iteration, trials, trial, **info):
            self._write("recover", trial)

        def on_trial_complete(self, iteration, trials, trial, **info):
            self._write("complete", trial)

    return TrialEventLog()


def main() -> None:
    from ludwig.api import LudwigModel  # imported here: Ludwig lives in its own environment
    from ludwig.hyperopt.execution import RayTuneExecutor
    from ludwig.hyperopt.run import hyperopt

    from src.ludwig_metrics import RegisterMetrics, SearchPhaseRecorder


    # Ludwig 0.11 bug: after the search it re-evaluates trials that ASHA (or the time budget)
    # stopped early, loading model/model_weights from their checkpoint - but early-stopped
    # trials only saved training_checkpoints/, so the whole run crashes. We retrain and
    # evaluate the best trial ourselves below, so that step is skipped.
    RayTuneExecutor._evaluate_best_model = staticmethod(lambda *args, **kwargs: None)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=None, help="override hyperopt.executor.num_samples")
    parser.add_argument("--n-initial-points", type=int, default=None, help="override search_alg.n_initial_points")
    parser.add_argument("--time-budget-s", type=int, default=None, help="override hyperopt.executor.time_budget_s")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"{output_dir} already has results. Choose a new --output-dir, so nothing is overwritten.")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    executor = config["hyperopt"]["executor"]
    if args.num_samples:
        executor["num_samples"] = args.num_samples
    if args.n_initial_points:
        config["hyperopt"]["search_alg"]["n_initial_points"] = args.n_initial_points
    if args.time_budget_s:
        executor["time_budget_s"] = args.time_budget_s
    budget = search_budget(config)
    print(f"Search budget: {budget}")
    if not budget["adaptive_trials"]:
        print("WARNING: every trial is random, so the adaptive TPE stage won't start. Raise num_samples.")

    columns = [*FEATURES, TARGET]
    train_full = pd.read_parquet(PROCESSED_DIR / "train.parquet")[columns]
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")[columns]
    train, val = train_test_split(train_full, test_size=0.2, stratify=train_full[TARGET], random_state=RANDOM_STATE)

    # Ludwig refuses to z-score a constant column, and it carries no signal anyway.
    constant = [f for f in FEATURES if train[f].nunique() <= 1]
    if constant:
        print(f"Dropping constant features for Ludwig: {constant}")
        config["input_features"] = [f for f in config["input_features"] if f["name"] not in constant]
    print(f"Ludwig search: {len(train):,} train / {len(val):,} validation / {len(test):,} test IPs (test not searched)")

    metric = config["hyperopt"]["metric"]
    events_path = output_dir / EVENTS_FILE
    recorder = SearchPhaseRecorder(budget["n_initial_points"], config["hyperopt"]["search_alg"]["random_state_seed"])
    recorder.install()
    results = hyperopt(config, training_set=train, validation_set=val, output_directory=str(output_dir),
                       experiment_name="hyperopt", backend="local", random_seed=RANDOM_STATE,
                       skip_save_processed_input=True, callbacks=[RegisterMetrics()],
                       tune_callbacks=[_trial_logger(events_path)])
    trials = results.ordered_trials
    if not trials:
        raise SystemExit("No trial reported a score. See trial_events.jsonl and the Ray log.")
    best_params = trials[0].parameters
    print(f"Best trial (val {metric}={trials[0].metric_score:.4f}): {best_params}")

    best_config = final_training_config(config, best_params)
    with tempfile.TemporaryDirectory() as tmp:
        model = LudwigModel(best_config, backend="local", logging_level=30)
        model.train(training_set=train, validation_set=val, output_directory=tmp,
                    random_seed=RANDOM_STATE, skip_save_processed_input=True)
        val_proba = positive_proba(model.predict(dataset=val, output_directory=tmp)[0])
        y_val = val[TARGET].to_numpy()
        threshold, val_f1 = best_f1_threshold(y_val, val_proba)  # validation only
        val_metrics = evaluate(y_val, val_proba, threshold)
        test_proba = positive_proba(model.predict(dataset=test, output_directory=tmp)[0])  # every choice is made by now
        test_metrics = evaluate(test[TARGET].to_numpy(), test_proba, threshold)
        model.save(str(output_dir / "best_model"))
    np.save(output_dir / "test_proba.npy", test_proba)

    stats = json.loads((output_dir / "hyperopt" / "hyperopt_statistics.json").read_text())
    events = summarise_events(read_events(events_path), recorder.phases)
    measured = {"searcher_n_startup_jobs": recorder.n_startup_jobs,
                "random_suggestions": recorder.phases.count("random startup"),
                "adaptive_suggestions": recorder.phases.count("adaptive TPE")}
    print(f"Measured during the search: {measured}")
    rows = trial_rows(stats, events)
    summary = {"budget": budget, "measured": measured, "objective": f"validation {metric}",
               "secondary": "validation roc_auc",
               "trials_completed": sum(1 for e in events.values() if e["status"] == "COMPLETED"),
               "trials_failed": [k for k, e in events.items() if e["status"] != "COMPLETED"],
               "trials_retried": {k: e["n_retries"] for k, e in events.items() if e["n_retries"]},
               "best_params": best_params, "threshold": threshold, "val_f1": val_f1,
               "val_pr_auc": val_metrics["pr_auc"], "val_roc_auc": val_metrics["roc_auc"],
               "test": test_metrics, "n_train": len(train), "n_validation": len(val), "n_test": len(test),
               "trials": [{k: v for k, v in row.items() if k != "curves"} for row in rows]}
    (output_dir / "search_results.json").write_text(json.dumps(summary, indent=2, default=float))
    (output_dir / "best_config.yaml").write_text(yaml.safe_dump(best_config, sort_keys=False))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="ludwig_search_v2"):
        mlflow.set_tags({"automl": "ludwig", "experiment_track": "ludwig_v2", "objective": f"validation {metric}"})
        mlflow.log_params({"model": "ludwig_ecd", "n_features": len(FEATURES), "n_train": len(train),
                           "n_trials": len(trials), "n_initial_points": budget["n_initial_points"],
                           "searcher_n_startup_jobs": measured["searcher_n_startup_jobs"],
                           "random_suggestions": measured["random_suggestions"],
                           "adaptive_suggestions": measured["adaptive_suggestions"],
                           **{f"best.{k}": v for k, v in best_params.items()}})
        mlflow.log_metrics({f"trial_val_{metric}": trials[0].metric_score, "val_f1": val_f1,
                            "val_pr_auc": val_metrics["pr_auc"], "val_roc_auc": val_metrics["roc_auc"],
                            "threshold": threshold, "n_trials_failed": len(summary["trials_failed"]),
                            "n_trials_retried": len(summary["trials_retried"]),
                            **{f"test_{k}": v for k, v in test_metrics.items()}})
        mlflow.log_artifact(str(output_dir / "best_config.yaml"))
        mlflow.log_artifact(str(output_dir / "search_results.json"))
        if events_path.exists():
            mlflow.log_artifact(str(events_path))
        log_trials(mlflow, rows, metric)

    print(f"Ludwig best: val_f1={val_f1:.4f}  val_pr_auc={val_metrics['pr_auc']:.4f}  "
          f"test_f1={test_metrics['f1']:.4f}  test_pr_auc={test_metrics['pr_auc']:.4f}  threshold={threshold:.3f}  "
          f"(saved to {output_dir})")


if __name__ == "__main__":
    main()
