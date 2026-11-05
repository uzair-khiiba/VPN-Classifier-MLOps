"""Make the first Ludwig search's 20 trials searchable in MLflow, from its saved files. Nothing is rerun.

They're logged to experiment zainab_ludwig_trials in the separate model-packages database, so the main experiment
history doesn't change. Safe to repeat: an existing parent run stops it.

Usage: python -m experiments.backfill_ludwig_trials
"""

import json
import os
import re
from collections import Counter

from experiments.common import REPORT_DIR
from experiments.log_models import PACKAGE_ARTIFACTS, PACKAGE_DB
from src.ludwig_trials import RETRIED_TAG, log_trials, trial_rows

STATS_DIR = "ludwig_results/hyperopt"
EXPERIMENT = "zainab_ludwig_trials"
PARENT_NAME = "ludwig_automl_v1_trials"
N_INITIAL_POINTS = 20  # the default that applied to this search, so every trial was in the random stage


def event_summary(stats_dir, log_text: str) -> dict[str, dict]:
    """Trial ids and settings from each trial's params.json, retries from the search log."""
    errored = Counter(re.findall(r"Trial task failed for trial (trial_\w+)", log_text))
    info = {}
    for folder in sorted(stats_dir.glob("trial_*")):
        params = json.loads((folder / "params.json").read_text())
        info[folder.name] = {"status": "COMPLETED", "errors": errored[folder.name], RETRIED_TAG: errored[folder.name],
                             "sampling_phase": "random startup", "config": params}
    return info


def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    from src.config import ROOT

    stats_dir = ROOT / STATS_DIR
    stats = json.loads((stats_dir / "hyperopt_statistics.json").read_text())
    log_text = (REPORT_DIR / "logs" / "ludwig_full.log").read_text(encoding="utf-8", errors="replace")
    rows = trial_rows(stats, event_summary(stats_dir, log_text))
    unmatched = [r["rank"] for r in rows if "trial_id" not in r]
    if unmatched:
        raise RuntimeError(f"trials at ranks {unmatched} couldn't be matched to their folders")

    mlflow.set_tracking_uri(f"sqlite:///{PACKAGE_DB.as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(PACKAGE_ARTIFACTS / "trials").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    found = mlflow.search_runs(experiment_names=[EXPERIMENT], output_format="list",
                               filter_string=f"tags.mlflow.runName='{PARENT_NAME}'")
    if found:
        print(f"{PARENT_NAME} is already logged ({found[0].info.run_id}). Nothing to do.")
        return
    with mlflow.start_run(run_name=PARENT_NAME):
        mlflow.set_tags({"automl": "ludwig", "source": "ludwig_results/hyperopt, logged after the search",
                         "objective": "validation roc_auc"})
        mlflow.log_params({"n_trials": len(rows), "n_initial_points": N_INITIAL_POINTS, "adaptive_trials": 0})
        mlflow.log_metrics({"best_trial_val_roc_auc": rows[0]["score"],
                            "n_trials_retried": sum(1 for r in rows if r.get(RETRIED_TAG))})
        count = log_trials(mlflow, rows, "roc_auc")
    print(f"Logged {count} trials under {PARENT_NAME}: best {rows[0]['trial_id']} {rows[0]['score']:.6f}; "
          f"retried: {[r['trial_id'] for r in rows if r.get(RETRIED_TAG)]}")


if __name__ == "__main__":
    main()
