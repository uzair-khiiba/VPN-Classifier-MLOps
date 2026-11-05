"""Read saved evaluation results for display. Never loads a model.

Every function returns None or an empty list when its file is absent, because reports/ and data/
are not always present (a fresh clone has only models/). Pages then say what is unavailable.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "reports" / "experiments" / "results"
FIGURES = ROOT / "reports" / "experiments" / "figures"
COMPARISON_METRICS = ("test_f1", "test_precision", "test_recall", "test_pr_auc", "threshold")
MATCH_TOLERANCE = 1e-3


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def model_meta() -> dict | None:
    return read_json(MODELS / "model_meta.json")


def feature_importance() -> dict | None:
    return read_json(MODELS / "feature_importance.json")


def group_importance() -> list[dict]:
    data = read_json(RESULTS / "explainability.json")
    return data.get("groups", []) if isinstance(data, dict) else []


def dataset_stats() -> dict | None:
    """Saved dataset facts from data preparation, not live data."""
    data = read_json(ROOT / "data" / "processed" / "prepare_report.json")
    if not isinstance(data, dict) or "n_ips" not in data:
        return None
    return {key: data.get(key) for key in ("n_rows", "n_ips", "positive_rate", "n_train", "n_test", "time_range")}


def _records(experiment: str) -> list[dict]:
    data = read_json(RESULTS / f"{experiment}.json")
    return data if isinstance(data, list) else []


def experimental_model() -> dict | None:
    """The experiment track's selected run. It is a comparison only: the API does not serve it."""
    selections = read_json(RESULTS / "selections.json")
    if not isinstance(selections, dict):
        return None
    run_name = selections.get("overall", {}).get("selected_run")
    for experiment in ("exp_c", "exp_c2", "exp_b"):
        for record in _records(experiment):
            if record.get("run_name") == run_name and "test_f1" in record.get("metrics", {}):
                return record
    return None


def confusion_matrix(metrics: dict) -> dict | None:
    if all(f"test_{key}" in metrics for key in ("tp", "fp", "fn", "tn")):
        return {key: int(metrics[f"test_{key}"]) for key in ("tp", "fp", "fn", "tn")}
    return None


def served_confusion_matrix(model: str, test_f1: float | None) -> dict | None:
    """Counts from the baseline reproduction, used only if that run scored the same as the served model."""
    if test_f1 is None:
        return None
    for record in _records("baseline"):
        metrics = record.get("metrics", {})
        if record.get("model") == model and abs(metrics.get("test_f1", -1) - test_f1) < MATCH_TOLERANCE:
            return confusion_matrix(metrics)
    return None


def comparison_rows(served_model: str | None) -> list[dict]:
    """Test results of every model with a saved record, labelled by its role."""
    rows = []
    meta = model_meta()
    for name, metrics in (meta or {}).get("all_runs", {}).items():
        role = "Served by the API" if name == served_model else "Baseline candidate"
        rows.append({"model": name, "role": role, **{key: metrics.get(key) for key in COMPARISON_METRICS}})
    selected = experimental_model()
    if selected:
        rows.append({"model": selected["name"], "role": "Experimental, not served",
                     **{key: selected["metrics"].get(key) for key in COMPARISON_METRICS}})
    for record in _records("exp_f"):
        rows.append({"model": record.get("name"), "role": "Experiment, not served",
                     **{key: record.get("metrics", {}).get(key) for key in COMPARISON_METRICS}})
    return sorted(rows, key=lambda row: -(row["test_f1"] or 0))


def figure(name: str) -> Path | None:
    path = FIGURES / name
    return path if path.exists() else None
