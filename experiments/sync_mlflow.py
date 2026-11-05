"""Log every saved result to MLflow. Safe to re-run: runs that already exist are skipped.

Usage:
    python -m experiments.sync_mlflow                                # local database
    EXPERIMENT_TRACKING=dagshub python -m experiments.sync_mlflow    # DagsHub (see final report)
"""

from experiments.common import RESULTS_DIR, connect_tracking, load_records, log_records


def main() -> None:
    print("tracking:", connect_tracking())
    for path in sorted(RESULTS_DIR.glob("*.json")):
        records = load_records(path.stem)
        if isinstance(records, list) and records and "run_name" in records[0]:
            logged, skipped = log_records(records, path.stem)
            print(f"{path.stem}: {logged} logged, {skipped} skipped", flush=True)


if __name__ == "__main__":
    main()
