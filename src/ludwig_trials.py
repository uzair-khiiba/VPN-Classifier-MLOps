"""Turn a finished Ludwig search into searchable MLflow runs: one parent, with one nested run per trial.

Needs no Ludwig install, so it also runs in the main environment. Each trial records its settings, its score, its
per-epoch validation curves, and how many times Ray had to retry it.
"""

import json
from pathlib import Path

RETRIED_TAG = "n_retries"


def read_events(path: Path) -> list[dict]:
    """The trial events written during a search, one JSON object per line. Missing file = no events."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarise_events(events: list[dict], phases: list[str] | None = None) -> dict[str, dict]:
    """Per trial: start order, sampling phase (as measured during the search), errors, retries and final status."""
    trials: dict[str, dict] = {}
    for event in events:
        trial = trials.setdefault(event["trial_id"], {"status": "UNKNOWN", "errors": 0, RETRIED_TAG: 0,
                                                     "config": event.get("config", {})})
        kind = event["event"]
        if kind == "start" and "start_order" not in trial:
            trial["start_order"] = sum(1 for t in trials.values() if "start_order" in t) + 1
        elif kind == "error":
            trial["errors"] += 1
            trial["status"] = "ERROR"
        elif kind == "recover":  # Ray reports a failure it will retry as this, without a separate error event
            trial["errors"] += 1
            trial[RETRIED_TAG] += 1
            trial["status"] = "RETRYING"
        elif kind == "complete":
            trial["status"] = "COMPLETED"
    for trial in trials.values():
        order = trial.get("start_order")
        if phases and order is not None and order <= len(phases):
            trial["sampling_phase"] = phases[order - 1]
    return trials


def _same_params(config: dict, parameters: dict) -> bool:
    for key, value in parameters.items():
        have = config.get(key)
        if have is None or (isinstance(value, float) and abs(float(have) - value) > 1e-9) or \
                (not isinstance(value, float) and have != value):
            return False
    return True


def trial_rows(stats: dict, event_summary: dict[str, dict] | None = None) -> list[dict]:
    """Rows ordered best first, joining Ludwig's results with the event log by their sampled settings."""
    rows = []
    for rank, result in enumerate(stats["hyperopt_results"], start=1):
        parameters = result["parameters"]
        row = {"rank": rank, "score": float(result["metric_score"]), "parameters": parameters,
               "curves": result.get("training_stats", {}).get("validation", {}).get("label", {})}
        for trial_id, info in (event_summary or {}).items():
            if _same_params(info.get("config", {}), parameters):
                row.update(trial_id=trial_id, **{k: v for k, v in info.items() if k != "config"})
                break
        rows.append(row)
    return rows


def log_trials(mlflow_module, rows: list[dict], metric_name: str) -> int:
    """Log each trial as a run nested under the active run. Returns how many were logged."""
    for row in rows:
        name = row.get("trial_id") or f"trial_rank_{row['rank']:02d}"
        with mlflow_module.start_run(run_name=name, nested=True):
            mlflow_module.log_params({key: str(value)[:500] for key, value in row["parameters"].items()})
            mlflow_module.set_tags({"rank": row["rank"], "status": row.get("status", "COMPLETED"),
                                    RETRIED_TAG: row.get(RETRIED_TAG, 0), "errors": row.get("errors", 0),
                                    "sampling_phase": row.get("sampling_phase", "not recorded")})
            mlflow_module.log_metric(f"trial_val_{metric_name}", row["score"])
            for key, values in row["curves"].items():
                for step, value in enumerate(values):
                    mlflow_module.log_metric(f"val_{key}", float(value), step=step)
            mlflow_module.log_metric("epochs_run", max((len(v) for v in row["curves"].values()), default=0))
    return len(rows)
