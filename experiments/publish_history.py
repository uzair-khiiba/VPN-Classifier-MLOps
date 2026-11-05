"""Copy finished MLflow runs from the local databases to a remote server, without rerunning anything.

Local databases stay the source of truth. Each one is read through a temporary copy, so they can't be changed.

Every copied run carries the tag mirror.source_run_id. That makes the copy safe to repeat:
  * a run that's already complete and matches is skipped, so no duplicates are created
  * a run that stopped part-way is resumed: only its missing values and files are sent
  * one run failing doesn't stop the others, and every failure is reported

Usage (after `dagshub login` in your own terminal):
    python -m experiments.publish_history --plan        # what would be sent; contacts nothing
    python -m experiments.publish_history               # copy, then verify every value on the server
    python -m experiments.publish_history --verify-only # only compare local and remote

The report is written to reports/experiments/results/dagshub_sync_report.json.
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

from experiments.common import ARTIFACT_DIR, RESULTS_DIR, connect_dagshub
from src.config import ROOT

SOURCE_TAG = "mirror.source_run_id"
COMPLETE_TAG = "mirror.complete"
ORIGIN_TAG = "mirror.source_database"
BATCH_METRICS, BATCH_PARAMS, BATCH_TAGS = 1000, 100, 100
ATTEMPTS = 4
RETRY_SECONDS = 20
VALUE_TOLERANCE = 1e-9
REPORT_PATH = RESULTS_DIR / "dagshub_sync_report.json"
# The server can't use the name zainab_experiments: looking it up returns errors, and DagsHub support would have to
# clear it. So those runs live under this name on DagsHub, and keep their original run names.
MAIN_REMOTE_NAME = os.getenv("DAGSHUB_MAIN_EXPERIMENT", "zainab_experiments_history")


@dataclass(frozen=True)
class Source:
    database: Path
    experiment: str
    remote_name: str


SOURCES = [
    Source(ARTIFACT_DIR / "zainab_mlflow.db", "zainab_experiments", MAIN_REMOTE_NAME),
    Source(ARTIFACT_DIR / "zainab_mlflow.db", "zainab_ludwig_check", "zainab_ludwig_check"),
    Source(ARTIFACT_DIR / "model_packages_mlflow.db", "zainab_ludwig_trials", "zainab_ludwig_trials"),
    Source(ARTIFACT_DIR / "baseline" / "baseline_repro_mlflow.db", "zainab_baseline_repro_local",
           "zainab_baseline_repro_local"),
]
# Absolute paths reveal this machine's folders and mean nothing elsewhere. Anything else is copied unchanged.
LOCAL_PATH_TAGS = {"mlflow.source.name", "mlflow.source.git.repoURL"}
LOCAL_PATH = re.compile(r"^[A-Za-z]:[\\/]|^/(home|Users)/")
URL_CREDENTIALS = re.compile(r"(?<=//)[^/@]+@")


def clean_tags(tags: dict[str, str]) -> dict[str, str]:
    """Tags to send: MLflow's internal bookkeeping is excluded and absolute local paths are shortened."""
    out = {}
    for key, value in tags.items():
        if key == "mlflow.log-model.history" or key.startswith(("mirror.", "model_logged.")):
            continue
        if key in LOCAL_PATH_TAGS and LOCAL_PATH.match(value):
            value = Path(value.replace("\\", "/")).name
        value = URL_CREDENTIALS.sub("", value)  # a Git remote URL can embed a login
        out[key] = value
    return out


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def list_all_runs(client: MlflowClient, experiment_id: str) -> list:
    runs, token = [], None
    while True:
        page = client.search_runs([experiment_id], max_results=500, page_token=token)
        runs.extend(page)
        token = page.token
        if not token:
            return runs


def list_files(client: MlflowClient, run_id: str, path: str | None = None) -> dict[str, int | None]:
    """Every artifact of a run: relative path -> size."""
    files = {}
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            files.update(list_files(client, run_id, item.path))
        else:
            files[item.path] = item.file_size
    return files


def metric_history(client: MlflowClient, run) -> list[Metric]:
    history = []
    for key in run.data.metrics:
        history.extend(client.get_metric_history(run.info.run_id, key))
    return history


def _retry(action, description: str):
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return action()
        except Exception as exc:
            if attempt == ATTEMPTS:
                raise
            print(f"    retry {attempt}/{ATTEMPTS - 1} after {type(exc).__name__} while {description}", flush=True)
            time.sleep(RETRY_SECONDS * attempt)


def copy_run(source: MlflowClient, target: MlflowClient, run, experiment_id: str, existing, database: str,
             id_map: dict[str, str]) -> str:
    """Create or resume one run on the target, and send only what it's missing."""
    info = run.info
    wanted_tags = clean_tags(run.data.tags)
    wanted_tags[SOURCE_TAG] = info.run_id
    wanted_tags[ORIGIN_TAG] = database
    parent = wanted_tags.get("mlflow.parentRunId")
    if parent:  # a nested run points at its parent's id on the server, not the local one
        wanted_tags["mlflow.parentRunId"] = id_map.get(parent, parent)
    if existing is None:
        created = _retry(lambda: target.create_run(experiment_id, start_time=info.start_time,
                                                   run_name=info.run_name), f"creating {info.run_name}")
        run_id = created.info.run_id
        have_params, have_tags = {}, {}
    else:
        run_id = existing.info.run_id
        have_params, have_tags = existing.data.params, existing.data.tags

    params = [Param(k, v) for k, v in run.data.params.items() if k not in have_params]
    tags = [RunTag(k, v) for k, v in wanted_tags.items() if have_tags.get(k) != v]
    history = metric_history(source, run)
    if existing is not None:
        history = [m for m in history if m.key not in existing.data.metrics]
    for batch in _chunks(params, BATCH_PARAMS):
        _retry(lambda b=batch: target.log_batch(run_id, params=b), f"logging params of {info.run_name}")
    for batch in _chunks(tags, BATCH_TAGS):
        _retry(lambda b=batch: target.log_batch(run_id, tags=b), f"logging tags of {info.run_name}")
    for batch in _chunks(history, BATCH_METRICS):
        _retry(lambda b=batch: target.log_batch(run_id, metrics=b), f"logging metrics of {info.run_name}")

    wanted_files = list_files(source, info.run_id)
    remote_files = list_files(target, run_id) if existing is not None and wanted_files else {}
    if wanted_files:
        with tempfile.TemporaryDirectory() as tmp:
            for path, size in wanted_files.items():
                if path in remote_files and remote_files[path] == size:
                    continue
                local = source.download_artifacts(info.run_id, path, tmp)
                folder = str(Path(path).parent).replace("\\", "/")
                _retry(lambda p=local, f=folder: target.log_artifact(run_id, p, None if f == "." else f),
                       f"uploading {path}")
    _retry(lambda: target.set_terminated(run_id, info.status, info.end_time), f"closing {info.run_name}")
    _retry(lambda: target.set_tag(run_id, COMPLETE_TAG, "true"), f"marking {info.run_name} complete")
    return run_id


def compare_run(source: MlflowClient, target: MlflowClient, run, remote) -> list[str]:
    """Every difference between a local run and its remote copy. An empty list means they match."""
    problems = []
    if remote.info.run_name != run.info.run_name:
        problems.append(f"run name is {remote.info.run_name}")
    if remote.info.status != run.info.status:
        problems.append(f"status is {remote.info.status}")
    if remote.data.tags.get(COMPLETE_TAG) != "true":
        problems.append("not marked complete")
    if remote.data.params != run.data.params:
        missing = sorted(set(run.data.params) - set(remote.data.params))
        changed = sorted(k for k, v in run.data.params.items()
                         if k in remote.data.params and remote.data.params[k] != v)
        problems.append(f"params differ: missing {missing}, changed {changed}")
    for key, value in run.data.metrics.items():
        got = remote.data.metrics.get(key)
        if got is None or abs(got - value) > VALUE_TOLERANCE * max(1, abs(value)):
            problems.append(f"metric {key} is {got}, local {value}")
    for key, value in clean_tags(run.data.tags).items():
        if key != "mlflow.parentRunId" and remote.data.tags.get(key) != value:
            problems.append(f"tag {key} differs")
    wanted = list_files(source, run.info.run_id)
    if wanted:
        got = list_files(target, remote.info.run_id)
        for path, size in wanted.items():
            if path not in got:
                problems.append(f"artifact {path} is missing")
            elif size is not None and got[path] is not None and got[path] != size:
                problems.append(f"artifact {path} is {got[path]} bytes, local {size}")
    return problems


def _source_client(database: Path, work_dir: Path) -> MlflowClient:
    """A client on a temporary copy. The stored artifact locations still point at the original files, read-only."""
    copy = work_dir / f"{database.stem}_{abs(hash(str(database)))}.db"
    if not copy.exists():
        shutil.copy2(database, copy)
    return MlflowClient(tracking_uri=f"sqlite:///{copy.as_posix()}")


def publish_experiment(source: MlflowClient, target: MlflowClient, experiment: str, remote_name: str, database: str,
                       verify_only: bool = False) -> dict:
    local = source.get_experiment_by_name(experiment)
    if local is None:
        return {"experiment": experiment, "remote_name": remote_name, "skipped": "not in this database",
                "local_runs": 0, "uploaded": 0, "already_there": 0, "verified": 0, "failed": [], "duplicates": []}
    runs = sorted(list_all_runs(source, local.experiment_id), key=lambda r: "mlflow.parentRunId" in r.data.tags)
    remote = _retry(lambda: target.get_experiment_by_name(remote_name), f"finding {remote_name}")
    if remote is None:
        if verify_only:
            experiment_id = None
        else:
            experiment_id = _retry(lambda: target.create_experiment(remote_name), f"creating {remote_name}")
    else:
        experiment_id = remote.experiment_id

    remote_runs = _retry(lambda: list_all_runs(target, experiment_id), "listing remote runs") if experiment_id else []
    by_source: dict[str, list] = {}
    for item in remote_runs:
        if SOURCE_TAG in item.data.tags:
            by_source.setdefault(item.data.tags[SOURCE_TAG], []).append(item)
    duplicates = sorted(k for k, v in by_source.items() if len(v) > 1)
    id_map = {k: v[0].info.run_id for k, v in by_source.items()}

    uploaded, already, failed = 0, 0, []
    for index, run in enumerate(runs, start=1):
        name = run.info.run_name
        existing = by_source.get(run.info.run_id, [None])[0]
        try:
            if verify_only:
                continue
            if existing is not None and not compare_run(source, target, run, existing):
                already += 1
                continue
            started = time.perf_counter()
            id_map[run.info.run_id] = copy_run(source, target, run, experiment_id, existing, database, id_map)
            uploaded += 1
            print(f"  [{index}/{len(runs)}] {name}: sent in {time.perf_counter() - started:.0f}s", flush=True)
        except Exception as exc:
            failed.append({"run": name, "run_id": run.info.run_id, "error": f"{type(exc).__name__}: {exc}"[:400]})
            print(f"  [{index}/{len(runs)}] {name}: FAILED {type(exc).__name__}", flush=True)

    # Verify from a fresh listing, so the result reflects what the server actually holds.
    remote_runs = _retry(lambda: list_all_runs(target, experiment_id), "listing remote runs") if experiment_id else []
    by_source = {}
    for item in remote_runs:
        if SOURCE_TAG in item.data.tags:
            by_source.setdefault(item.data.tags[SOURCE_TAG], []).append(item)
    verified, mismatches = 0, []
    for run in runs:
        matches = by_source.get(run.info.run_id, [])
        problems = compare_run(source, target, run, matches[0]) if matches else ["missing on the server"]
        if len(matches) > 1:
            problems.append(f"{len(matches)} copies")
        if problems:
            mismatches.append({"run": run.info.run_name, "problems": problems[:10]})
        else:
            verified += 1
    return {"experiment": experiment, "remote_name": remote_name, "remote_experiment_id": experiment_id,
            "local_runs": len(runs), "uploaded": uploaded, "already_there": already, "verified": verified,
            "remote_runs_visible": len(remote_runs), "failed": failed, "mismatches": mismatches,
            "duplicates": sorted(k for k, v in by_source.items() if len(v) > 1) or duplicates}


def plan(work_dir: Path) -> list[dict]:
    rows = []
    for item in SOURCES:
        client = _source_client(item.database, work_dir)
        local = client.get_experiment_by_name(item.experiment)
        runs = list_all_runs(client, local.experiment_id) if local else []
        with_files = sum(1 for r in runs if list_files(client, r.info.run_id))
        rows.append({"database": item.database.relative_to(ROOT).as_posix(), "experiment": item.experiment,
                     "remote_name": item.remote_name, "runs": len(runs), "runs_with_artifacts": with_files,
                     "metric_values": sum(len(r.data.metrics) for r in runs),
                     "params": sum(len(r.data.params) for r in runs)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", action="store_true", help="show what would be sent, contacting nothing")
    parser.add_argument("--verify-only", action="store_true", help="compare local and remote, sending nothing")
    args = parser.parse_args()
    work_dir = Path(tempfile.mkdtemp(prefix="mlflow_read_copy_"))
    try:
        if args.plan:
            print(json.dumps(plan(work_dir), indent=2))
            return
        uri = connect_dagshub()
        target = MlflowClient(tracking_uri=uri)
        results = []
        for item in SOURCES:
            print(f"{item.experiment} -> {item.remote_name}", flush=True)
            source = _source_client(item.database, work_dir)
            results.append(publish_experiment(source, target, item.experiment, item.remote_name,
                                              item.database.name, args.verify_only))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    report = {"tracking_uri": uri, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "experiments": results,
              "total_local_runs": sum(r["local_runs"] for r in results),
              "total_verified": sum(r["verified"] for r in results),
              "total_failed": sum(len(r["failed"]) for r in results)}
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    for row in results:
        print({k: (len(v) if isinstance(v, list) else v) for k, v in row.items()})
    print(f"Verified {report['total_verified']} of {report['total_local_runs']} runs. Report: {REPORT_PATH}")
    if report["total_verified"] != report["total_local_runs"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
