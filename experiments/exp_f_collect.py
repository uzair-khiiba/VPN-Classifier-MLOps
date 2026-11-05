"""Experiment F, step 2: turn the Ludwig MLflow run into a result record, with the paired comparison.

The saved model's predictions must reproduce the test F1 that src.automl_ludwig logged, otherwise
this stops. It also records how the search itself went: trials, failures and search time.

Usage: python -m experiments.exp_f_collect
"""

import json
import re
from pathlib import Path

import mlflow
import numpy as np

from experiments.common import (
    ARTIFACT_DIR,
    EXPERIMENT_NAME,
    REPORT_DIR,
    RESULTS_DIR,
    bootstrap_f1_difference,
    connect_tracking,
    load_records,
    load_split,
    save_records,
    test_summary,
)
from src.config import ROOT, TARGET


def main() -> None:
    connect_tracking()
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], output_format="list",
                              filter_string="tags.mlflow.runName='ludwig_automl'")
    if len(runs) != 1:
        raise RuntimeError(f"expected one ludwig_automl run, found {len(runs)}")
    run = runs[0]
    logged = run.data.metrics

    _, test = load_split()
    y = test[TARGET].to_numpy()
    proba = np.load(ARTIFACT_DIR / "test_proba_ludwig.npy")
    metrics = test_summary(y, proba, logged["threshold"])
    print(f"test F1: logged {logged['test_f1']:.6f}, rescored {metrics['test_f1']:.6f}")
    if abs(metrics["test_f1"] - logged["test_f1"]) > 1e-3:
        raise RuntimeError("the saved Ludwig model does not reproduce the logged test F1")

    base = next(r for r in load_records("exp_e") if r["name"] == "hgb_slow")
    base_proba = np.load(ARTIFACT_DIR / f"test_proba_{base['test_proba_key']}.npy")
    versus = bootstrap_f1_difference(y, proba >= logged["threshold"], base_proba >= base["metrics"]["threshold"])
    print(f"vs baseline: {versus['diff']:+.4f} [{versus['diff_low']:+.3f}, {versus['diff_high']:+.3f}]")
    if versus["diff_low"] > 0:
        label = "better"
    else:
        label = "worse" if versus["diff_high"] < 0 else "no clear change"

    log = (REPORT_DIR / "logs" / "ludwig_full.log").read_text(encoding="utf-8", errors="replace")
    search = re.findall(r"Total running time: (.+)", log)[-1].strip()
    best_model = ROOT / "ludwig_results" / "best_model"
    size_kb = sum(p.stat().st_size for p in Path(best_model).rglob("*") if p.is_file()) / 1024
    trials_path = mlflow.MlflowClient().download_artifacts(run.info.run_id, "ludwig_trials.json")
    trials = json.loads(Path(trials_path).read_text())
    scores = [t["score"] for t in trials["trials"]]

    save_records("exp_f", [{
        "run_name": "ludwig_automl", "name": "ludwig_automl", "experiment": "exp_f", "verdict": label,
        "change": "Ludwig neural net, best of 20 trials", "params": dict(run.data.params),
        "metrics": {**logged, **metrics, "model_size_kb": size_kb},
    }])
    comparison_path = RESULTS_DIR / "test_comparison.json"
    comparison = json.loads(comparison_path.read_text())
    comparison["vs_baseline"]["exp_f/ludwig_automl"] = versus
    comparison_path.write_text(json.dumps(comparison, indent=2))
    (RESULTS_DIR / "exp_f_search.json").write_text(json.dumps({
        "search_running_time": search, "best_model_size_kb": size_kb, "n_trials": len(scores),
        "best_val_roc_auc": max(scores), "worst_val_roc_auc": min(scores),
        "n_trials_with_task_failures": log.count("Trial task failed"), "vs_baseline": versus}, indent=2))
    print(f"search time {search}; model {size_kb:.0f} KB; verdict: {label}")


if __name__ == "__main__":
    main()
