"""Check that MLflow holds exactly what the JSON results say, value by value. Writes nothing to MLflow.

Usage: python -m experiments.verify_mlflow_sync
"""

import json
import sys

import mlflow

from experiments.common import EXPERIMENT_NAME, REPORT_DIR, RESULTS_DIR, connect_tracking, load_records

TRACKS = ["baseline", "exp_a", "exp_b", "exp_c", "exp_c2", "exp_d", "exp_e", "exp_f", "summary"]


def main() -> None:
    uri = connect_tracking()
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], output_format="list", max_results=1000)
    by_name: dict[str, list] = {}
    for run in runs:
        by_name.setdefault(run.info.run_name, []).append(run)
    problems, compared, expected = [], 0, set()
    for track in TRACKS:
        for rec in load_records(track):
            name = rec["run_name"]
            expected.add(name)
            matches = by_name.get(name, [])
            if len(matches) != 1:
                problems.append(f"{name}: found {len(matches)} runs")
                continue
            logged = matches[0].data.metrics
            for key, value in rec["metrics"].items():
                if value is None:
                    continue
                compared += 1
                if key not in logged or abs(logged[key] - float(value)) > 1e-9 * max(1, abs(float(value))):
                    problems.append(f"{name}: {key} is {logged.get(key)} in MLflow, {value} in JSON")
    unexpected = sorted(set(by_name) - expected)
    if unexpected:
        problems.append(f"runs with no JSON record: {unexpected}")
    summary_run = by_name["summary_selected_model_and_figures"][0]
    artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(summary_run.info.run_id)]
    wanted = len(load_records("summary")[0]["artifacts"])
    if len(artifacts) != wanted:
        problems.append(f"summary run has {len(artifacts)} artifacts, expected {wanted}")
    all_experiments = [e.name for e in mlflow.search_experiments()]
    report = {"tracking_uri": uri, "experiments_visible": all_experiments, "experiment": EXPERIMENT_NAME,
              "runs_in_mlflow": len(runs), "runs_expected": len(expected), "metric_values_compared": compared,
              "summary_artifacts": len(artifacts), "problems": problems}
    (RESULTS_DIR / "mlflow_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:3000])
    print(f"see also {REPORT_DIR / 'comparison_tables.md'}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
