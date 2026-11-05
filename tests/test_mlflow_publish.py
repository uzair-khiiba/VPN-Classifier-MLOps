"""Copying history, logging models and promoting them, tested on temporary local MLflow stores."""

import hashlib
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier

from experiments import log_models, promote_model
from experiments import publish_history as publish
from src.config import ROOT
from src.features import FEATURES
from src.ludwig_trials import log_trials, read_events, summarise_events, trial_rows
from src.model_io import load_bundle, save_bundle


def store(tmp_path, name: str) -> MlflowClient:
    folder = tmp_path / name
    folder.mkdir()
    return MlflowClient(tracking_uri=f"sqlite:///{(folder / 'mlflow.db').as_posix()}")


def add_run(client: MlflowClient, tmp_path, experiment_id: str, name: str, metric: float = 0.5, file: str | None = None,
            parent: str | None = None) -> str:
    tags = {"experiment_track": "exp_a", "mlflow.source.name": "C:\\Users\\someone\\project\\train.py"}
    if parent:
        tags["mlflow.parentRunId"] = parent
    run = client.create_run(experiment_id, run_name=name, tags=tags, start_time=1000)
    run_id = run.info.run_id
    client.log_batch(run_id, metrics=[Metric("cv_f1", metric, 111, 0), Metric("curve", 1.0, 111, 0),
                                      Metric("curve", 2.0, 112, 1)],
                     params=[Param("max_iter", "400")], tags=[RunTag("owner", "zainab")])
    if file:
        path = tmp_path / file
        path.write_text("contents")
        client.log_artifact(run_id, str(path), "figures")
    client.set_terminated(run_id, "FINISHED", 2000)
    return run_id


@pytest.fixture
def stores(tmp_path):
    source, target = store(tmp_path, "local"), store(tmp_path, "remote")
    location = (tmp_path / "local_artifacts").as_uri()
    experiment = source.create_experiment("exp", artifact_location=location)
    (tmp_path / "remote_artifacts").mkdir()
    target.create_experiment("exp_remote", artifact_location=(tmp_path / "remote_artifacts").as_uri())
    return source, target, experiment


def remote_runs(target: MlflowClient, name: str = "exp_remote") -> list:
    return publish.list_all_runs(target, target.get_experiment_by_name(name).experiment_id)


def test_copy_preserves_everything_and_verifies(stores, tmp_path):
    source, target, experiment = stores
    run_id = add_run(source, tmp_path, experiment, "run_a", file="plot.txt")
    add_run(source, tmp_path, experiment, "run_b", metric=0.7)
    result = publish.publish_experiment(source, target, "exp", "exp_remote", "local.db")
    assert (result["uploaded"], result["verified"], result["failed"], result["mismatches"]) == (2, 2, [], [])

    copy = next(r for r in remote_runs(target) if r.info.run_name == "run_a")
    assert copy.data.tags[publish.SOURCE_TAG] == run_id and copy.info.status == "FINISHED"
    assert copy.info.start_time == 1000 and copy.info.end_time == 2000
    assert copy.data.metrics["cv_f1"] == 0.5 and copy.data.params == {"max_iter": "400"}
    assert copy.data.tags["owner"] == "zainab" and copy.data.tags["experiment_track"] == "exp_a"
    assert [m.value for m in target.get_metric_history(copy.info.run_id, "curve")] == [1.0, 2.0]
    assert publish.list_files(target, copy.info.run_id) == publish.list_files(source, run_id)


def test_url_logins_are_removed_before_publishing():
    tags = publish.clean_tags({"mlflow.source.git.repoURL": "https://user:secret@github.com/o/r.git", "a": "b"})
    assert tags == {"mlflow.source.git.repoURL": "https://github.com/o/r.git", "a": "b"}


def test_local_paths_are_not_published(stores, tmp_path):
    source, target, experiment = stores
    add_run(source, tmp_path, experiment, "run_a")
    publish.publish_experiment(source, target, "exp", "exp_remote", "local.db")
    assert remote_runs(target)[0].data.tags["mlflow.source.name"] == "train.py"


def test_repeating_creates_no_duplicates_and_sends_nothing(stores, tmp_path):
    source, target, experiment = stores
    add_run(source, tmp_path, experiment, "run_a", file="plot.txt")
    add_run(source, tmp_path, experiment, "same_name")
    add_run(source, tmp_path, experiment, "same_name")
    publish.publish_experiment(source, target, "exp", "exp_remote", "local.db")
    again = publish.publish_experiment(source, target, "exp", "exp_remote", "local.db")
    assert (again["uploaded"], again["already_there"], again["verified"], again["duplicates"]) == (0, 3, 3, [])
    assert len(remote_runs(target)) == 3


def test_an_interrupted_run_is_resumed_not_duplicated(stores, tmp_path):
    source, target, experiment = stores
    run_id = add_run(source, tmp_path, experiment, "run_a", file="plot.txt")
    remote_id = target.get_experiment_by_name("exp_remote").experiment_id
    partial = target.create_run(remote_id, run_name="run_a", tags={publish.SOURCE_TAG: run_id})
    target.log_batch(partial.info.run_id, metrics=[Metric("cv_f1", 0.5, 111, 0)])
    result = publish.publish_experiment(source, target, "exp", "exp_remote", "local.db")
    assert result["verified"] == 1 and len(remote_runs(target)) == 1
    finished = target.get_run(partial.info.run_id)
    assert finished.data.params == {"max_iter": "400"} and finished.data.tags[publish.COMPLETE_TAG] == "true"
    assert list(publish.list_files(target, partial.info.run_id)) == ["figures/plot.txt"]


def test_a_failure_is_reported_and_the_others_continue(stores, tmp_path, monkeypatch):
    source, target, experiment = stores
    add_run(source, tmp_path, experiment, "bad")
    add_run(source, tmp_path, experiment, "good")
    original = publish.copy_run
    monkeypatch.setattr(publish, "RETRY_SECONDS", 0)

    def flaky(src, dst, run, *args):
        if run.info.run_name == "bad":
            raise ConnectionError("server down")
        return original(src, dst, run, *args)

    monkeypatch.setattr(publish, "copy_run", flaky)
    result = publish.publish_experiment(source, target, "exp", "exp_remote", "local.db")
    assert [f["run"] for f in result["failed"]] == ["bad"] and result["verified"] == 1
    assert [m["run"] for m in result["mismatches"]] == ["bad"]


def test_changed_remote_value_fails_verification(stores, tmp_path):
    source, target, experiment = stores
    add_run(source, tmp_path, experiment, "run_a")
    publish.publish_experiment(source, target, "exp", "exp_remote", "local.db")
    copy = remote_runs(target)[0]
    target.log_batch(copy.info.run_id, metrics=[Metric("cv_f1", 0.9, 999, 0)])
    result = publish.publish_experiment(source, target, "exp", "exp_remote", "local.db", verify_only=True)
    assert result["verified"] == 0 and "cv_f1" in result["mismatches"][0]["problems"][0]


def test_nested_runs_point_at_the_copied_parent(stores, tmp_path):
    source, target, experiment = stores
    parent = add_run(source, tmp_path, experiment, "parent")
    add_run(source, tmp_path, experiment, "child", parent=parent)
    publish.publish_experiment(source, target, "exp", "exp_remote", "local.db")
    runs = {r.info.run_name: r for r in remote_runs(target)}
    assert runs["child"].data.tags["mlflow.parentRunId"] == runs["parent"].info.run_id


def test_every_source_experiment_keeps_its_own_identity():
    names = {source.remote_name for source in publish.SOURCES}
    assert {"zainab_ludwig_check", "zainab_baseline_repro_local", "zainab_ludwig_trials"} <= names
    assert len(names) == len(publish.SOURCES)


@pytest.fixture
def small_bundle(tmp_path):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((300, len(FEATURES))) * 10, columns=FEATURES)
    y = (X["asn_unique_ips"] > 5).astype(int).to_numpy()
    model = HistGradientBoostingClassifier(max_iter=10, random_state=0).fit(X, y)
    bundle = {"model": model, "features": FEATURES, "threshold": 0.4, "reference": X.median().to_dict(),
              "metadata": {"model_name": "small", "version": "v1", "n_train": 300, "metrics": {"cv_f1": 0.7}}}
    return bundle, save_bundle(bundle, tmp_path / "small.joblib"), X, y


def test_sklearn_model_reloads_with_equal_predictions(small_bundle, tmp_path):
    bundle, path, X, y = small_bundle
    uri = f"sqlite:///{(tmp_path / 'models.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment("models", artifact_location=(tmp_path / "models_artifacts").as_uri())
    run = client.create_run(experiment, run_name="original", start_time=5)
    client.set_terminated(run.info.run_id, "FINISHED", 9)

    entry = log_models.log_sklearn(mlflow, client, run.info.run_id, "candidate", bundle, path, X)
    check = log_models.check_sklearn(mlflow, client, entry, bundle, X, y)
    assert check["predictions_equal"] and check["bundle_features_match"] and check["bundle_threshold_matches"]
    assert entry["method"] == "mlflow.sklearn.log_model" and entry["threshold"] == 0.4

    after = client.get_run(run.info.run_id)
    assert after.info.status == "FINISHED" and after.info.end_time == 9
    card_path = client.download_artifacts(run.info.run_id, "model_info/model_card.json", str(tmp_path))
    card = json.loads(Path(card_path).read_text())
    assert card["features"] == FEATURES and card["threshold"] == 0.4 and card["status"] == "candidate, not deployed"
    assert card["algorithm"] == "HistGradientBoostingClassifier" and "scikit-learn" in card["environment"]


def test_promotion_changes_nothing_without_approval(small_bundle, tmp_path):
    _, candidate, _, _ = small_bundle
    live = tmp_path / "live" / "model.joblib"
    live.parent.mkdir()
    live.write_bytes(candidate.read_bytes())
    before = hashlib.sha256(live.read_bytes()).hexdigest()
    result = promote_model.promote(candidate, live)
    assert result["promoted"] is False and hashlib.sha256(live.read_bytes()).hexdigest() == before
    assert not (live.parent / "promotion_log.jsonl").exists()


def test_promotion_needs_a_reason(small_bundle, tmp_path):
    _, candidate, _, _ = small_bundle
    with pytest.raises(ValueError, match="reason"):
        promote_model.promote(candidate, tmp_path / "model.joblib", approve=True)


def test_promotion_refuses_a_lower_cv_score_whatever_its_test_score(small_bundle, tmp_path):
    bundle, _, _, _ = small_bundle
    live = save_bundle(bundle, tmp_path / "live" / "model.joblib")
    worse = {**bundle, "metadata": {**bundle["metadata"], "metrics": {"cv_f1": 0.5, "test_f1": 0.99}}}
    path = save_bundle(worse, tmp_path / "worse.joblib")
    with pytest.raises(ValueError, match="checks failed"):
        promote_model.promote(path, live, approve=True, reason="higher test score")


def test_approved_promotion_archives_the_old_model_and_logs_it(small_bundle, tmp_path):
    bundle, _, _, _ = small_bundle
    live = save_bundle(bundle, tmp_path / "live" / "model.joblib")
    better = {**bundle, "metadata": {**bundle["metadata"], "model_name": "better", "version": "v2",
                                     "metrics": {"cv_f1": 0.8}}}
    path = save_bundle(better, tmp_path / "better.joblib")
    result = promote_model.promote(path, live, approve=True, reason="approved in review")
    assert result["promoted"] and load_bundle(live)["metadata"]["model_name"] == "better"
    assert load_bundle(live.parent / "archive" / "small_v1.joblib")["metadata"]["version"] == "v1"
    log = json.loads((live.parent / "promotion_log.jsonl").read_text())
    assert log["reason"] == "approved in review" and log["from"] == "small" and log["to"] == "better"


def test_deployed_model_has_not_changed():
    """A real change must go through experiments.promote_model, which records it, and then update this hash."""
    if (ROOT / "models" / "promotion_log.jsonl").exists():
        pytest.skip("a promotion has been recorded")
    digest = hashlib.sha256((ROOT / "models" / "model.joblib").read_bytes()).hexdigest()
    assert digest == "f253f1394aacd0b4bd872231fd1b4653208772887e331389417cbb069d97810b"
    assert load_bundle(ROOT / "models" / "model.joblib")["metadata"]["version"] == "20261019-173911"


def test_trial_events_record_retries_and_phases(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [("a", "start"), ("b", "start"), ("b", "error"), ("b", "recover"), ("c", "start"), ("a", "complete"),
              ("b", "complete"), ("c", "error")]
    path.write_text("\n".join(json.dumps({"event": e, "trial_id": t, "config": {"lr": ord(t)}}) for t, e in events))
    summary = summarise_events(read_events(path), phases=["random startup", "random startup", "adaptive TPE"])
    assert summary["a"]["status"] == "COMPLETED" and summary["a"]["sampling_phase"] == "random startup"
    assert summary["b"]["n_retries"] == 1 and summary["b"]["status"] == "COMPLETED" and summary["b"]["errors"] == 2
    assert summary["c"]["status"] == "ERROR" and summary["c"]["sampling_phase"] == "adaptive TPE"
    assert read_events(tmp_path / "missing.jsonl") == []


def test_trials_become_searchable_nested_runs(tmp_path):
    stats = {"hyperopt_results": [
        {"metric_score": 0.9, "parameters": {"trainer.learning_rate": 0.01},
         "training_stats": {"validation": {"label": {"pr_auc": [0.5, 0.9], "roc_auc": [0.8, 0.95]}}}},
        {"metric_score": 0.7, "parameters": {"trainer.learning_rate": 0.002},
         "training_stats": {"validation": {"label": {"pr_auc": [0.7]}}}}]}
    events = {"t1": {"status": "COMPLETED", "n_retries": 1, "errors": 1, "sampling_phase": "adaptive TPE",
                     "config": {"trainer.learning_rate": 0.01}}}
    rows = trial_rows(stats, events)
    assert rows[0]["trial_id"] == "t1" and "trial_id" not in rows[1]

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'trials.db').as_posix()}")
    mlflow.set_experiment("trials")
    with mlflow.start_run(run_name="parent") as parent:
        assert log_trials(mlflow, rows, "pr_auc") == 2
    children = mlflow.search_runs(experiment_names=["trials"], output_format="list",
                                  filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'")
    assert len(children) == 2
    first = next(c for c in children if c.info.run_name == "t1")
    assert first.data.metrics["trial_val_pr_auc"] == 0.9 and first.data.tags["n_retries"] == "1"
    assert first.data.tags["sampling_phase"] == "adaptive TPE" and first.data.metrics["epochs_run"] == 2
