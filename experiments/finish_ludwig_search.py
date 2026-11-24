"""Finish a Ludwig search whose trials ran but whose summary step was lost.

Ray Tune raises "Trials did not complete" when any trial fails twice, and Ludwig then discards the
whole search result, even though every other trial finished and saved its results to disk. On
Windows this happens when a trial hits a file-lock error ("Access is denied") on both attempts.

This script picks up from the trial folders and does exactly what src/automl_ludwig.py would have
done next, with the same functions:
  1. rank trials by their last reported validation score (Ray's and Ludwig's rule) and take the best.
     A trial whose final attempt errored is excluded. A trial stopped by the search's time budget
     counts, as it does in Ray;
  2. retrain those settings on the same train split, with early stopping disabled as in the trials;
  3. tune the F1 threshold on the validation split only, then score the test set once;
  4. write search_results.json (marked "recovered") and log one MLflow run.

The search itself isn't rerun or changed. Trials that never completed are listed, not used.

Usage (Ludwig environment): python -m experiments.finish_ludwig_search ludwig_results_v2
"""

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.automl_ludwig import (
    CONFIG_PATH,
    EVENTS_FILE,
    final_training_config,
    load_config,
    positive_proba,
    search_budget,
)
from src.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, PROCESSED_DIR, RANDOM_STATE, TARGET
from src.features import FEATURES
from src.ludwig_trials import read_events
from src.train import best_f1_threshold, evaluate


def read_trials(hyperopt_dir: Path) -> list[dict]:
    """Every trial folder's parameters and reported scores, best first (last reported score, as Ray ranks)."""
    trials = []
    for folder in sorted(hyperopt_dir.glob("trial_*")):
        lines = [json.loads(x) for x in (folder / "result.json").read_text().splitlines() if x.strip()] \
            if (folder / "result.json").exists() else []
        scored = [r for r in lines if r.get("metric_score") is not None]
        if not scored:
            continue
        trials.append({"trial_id": folder.name.removeprefix("trial_"),
                       "parameters": json.loads((folder / "params.json").read_text()),
                       "metric_score": float(scored[-1]["metric_score"]),
                       "best_metric_score_seen": max(float(r["metric_score"]) for r in scored),
                       "iterations": len(scored)})
    return sorted(trials, key=lambda t: t["metric_score"], reverse=True)


def main() -> None:
    from ludwig.api import LudwigModel

    import src.ludwig_metrics  # noqa: F401  registers the PR-AUC metric, as src/automl_ludwig.py does

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if (output_dir / "search_results.json").exists():
        raise SystemExit(f"{output_dir} already has search_results.json; nothing to recover.")

    config = load_config(args.config)
    metric = config["hyperopt"]["metric"]
    events = read_events(output_dir / EVENTS_FILE)
    last_event = {e["trial_id"]: e["event"] for e in events}
    completed = sorted(t for t, ev in last_event.items() if ev == "complete")
    failed = sorted(t for t, ev in last_event.items() if ev == "error")
    stopped_by_budget = sorted(t for t, ev in last_event.items() if ev == "start")
    retried = sorted({e["trial_id"] for e in events if e["event"] == "recover"} & set(completed))
    trials = [t for t in read_trials(output_dir / "hyperopt") if t["trial_id"] not in failed]
    best = trials[0]
    print(f"{len(trials)} eligible trials: completed {len(completed)} (of which retried {len(retried)}), "
          f"stopped by the time budget {stopped_by_budget}; failed and excluded {failed}")
    print(f"Best trial {best['trial_id']} (val {metric}={best['metric_score']:.4f}): {best['parameters']}")

    columns = [*FEATURES, TARGET]
    train_full = pd.read_parquet(PROCESSED_DIR / "train.parquet")[columns]
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")[columns]
    train, val = train_test_split(train_full, test_size=0.2, stratify=train_full[TARGET], random_state=RANDOM_STATE)
    constant = [f for f in FEATURES if train[f].nunique() <= 1]
    if constant:
        config["input_features"] = [f for f in config["input_features"] if f["name"] not in constant]

    best_config = final_training_config(config, best["parameters"])
    with tempfile.TemporaryDirectory() as tmp:
        model = LudwigModel(best_config, backend="local", logging_level=30)
        model.train(training_set=train, validation_set=val, output_directory=tmp,
                    random_seed=RANDOM_STATE, skip_save_processed_input=True)
        val_proba = positive_proba(model.predict(dataset=val, output_directory=tmp)[0])
        y_val = val[TARGET].to_numpy()
        threshold, val_f1 = best_f1_threshold(y_val, val_proba)  # validation only
        val_metrics = evaluate(y_val, val_proba, threshold)
        test_proba = positive_proba(model.predict(dataset=test, output_directory=tmp)[0])  # every choice is made
        test_metrics = evaluate(test[TARGET].to_numpy(), test_proba, threshold)
        model.save(str(output_dir / "best_model"))
    np.save(output_dir / "test_proba.npy", test_proba)

    summary = {"recovered": "Ray raised 'Trials did not complete' after the search; this summary was rebuilt "
                            "from the trial folders by experiments/finish_ludwig_search.py",
               "budget": search_budget(config), "objective": f"validation {metric}",
               "trials_ranked": len(trials), "trials_completed": len(completed), "trials_retried": retried,
               "trials_stopped_by_time_budget": stopped_by_budget, "trials_failed_excluded": failed,
               "best_trial": best["trial_id"], "best_params": best["parameters"], "threshold": threshold,
               "val_f1": val_f1, "val_pr_auc": val_metrics["pr_auc"], "val_roc_auc": val_metrics["roc_auc"],
               "test": test_metrics, "n_train": len(train), "n_validation": len(val), "n_test": len(test),
               "trials": [{"rank": i + 1, **t} for i, t in enumerate(trials)]}
    (output_dir / "search_results.json").write_text(json.dumps(summary, indent=2, default=float))
    (output_dir / "best_config.yaml").write_text(yaml.safe_dump(best_config, sort_keys=False))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="ludwig_search_v2"):
        mlflow.set_tags({"automl": "ludwig", "experiment_track": "ludwig_v2", "objective": f"validation {metric}",
                         "recovered_from_trial_folders": "true"})
        mlflow.log_params({"model": "ludwig_ecd", "n_features": len(FEATURES), "n_train": len(train),
                           "n_trials": len(trials), "n_initial_points": summary["budget"]["n_initial_points"],
                           **{f"best.{k}": v for k, v in best["parameters"].items()}})
        mlflow.log_metrics({f"trial_val_{metric}": best["metric_score"], "val_f1": val_f1,
                            "val_pr_auc": val_metrics["pr_auc"], "val_roc_auc": val_metrics["roc_auc"],
                            "threshold": threshold, "n_trials_failed": len(failed),
                            "n_trials_retried": len(retried), **{f"test_{k}": v for k, v in test_metrics.items()}})
        mlflow.log_artifact(str(output_dir / "best_config.yaml"))
        mlflow.log_artifact(str(output_dir / "search_results.json"))

    print(f"Ludwig best: val_f1={val_f1:.4f}  val_pr_auc={val_metrics['pr_auc']:.4f}  "
          f"test_f1={test_metrics['f1']:.4f}  test_pr_auc={test_metrics['pr_auc']:.4f}  threshold={threshold:.3f}")


if __name__ == "__main__":
    main()
