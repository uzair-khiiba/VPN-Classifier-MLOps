"""Ludwig AutoML, the simple way: Ludwig designs the model and its search by itself, then runs a random search.

Unlike src/automl_ludwig.py, where we wrote the network and its search space by hand, here we write no model
config at all. Ludwig's AutoML (ludwig.automl.create_auto_config) looks at the data and decides:
  * the type of every feature (number, category or binary),
  * the model architecture (for this data it picks TabNet, an attention-based network for tables),
  * which hyperparameters to tune and over what ranges,
  * the search algorithm: random search (Ray Tune's variant generator), with weak trials stopped early.

Steps:
  1. Split the training IPs into train/validation (stratified). The test set is never given to AutoML.
  2. AutoML generates the config. It is saved as auto_config.yaml, the record of what Ludwig chose.
  3. Run that search on train, scoring each trial on validation, within the time limit.
  4. Retrain the best trial, tune the F1 threshold on validation, and score the test set once.
  5. Log one MLflow run, `ludwig_auto_train`, next to the scikit-learn runs in mlflow.db.

Usage (needs the Ludwig environment: make setup-automl):
    python -m src.automl_simple                      # 10-minute search on all training IPs
    python -m src.automl_simple --time-limit-min 3   # quicker demo
"""

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, PROCESSED_DIR, RANDOM_STATE, ROOT, TARGET
from src.features import FEATURES

OUTPUT_DIR = ROOT / "ludwig_results_auto"
RUN_NAME = "ludwig_auto_train"


def describe_auto_config(config: dict) -> dict:
    """What Ludwig's AutoML decided, in plain terms. Needs no Ludwig install."""
    hyperopt = config.get("hyperopt", {})
    executor = hyperopt.get("executor", {})
    types: dict[str, list[str]] = {}
    for feature in config.get("input_features", []):
        types.setdefault(feature["type"], []).append(feature["name"])
    return {
        "architecture": config.get("combiner", {}).get("type"),
        "search_algorithm": hyperopt.get("search_alg", {}).get("type"),
        "early_stopping_scheduler": executor.get("scheduler", {}).get("type"),
        "max_trials": executor.get("num_samples"),
        "time_budget_s": executor.get("time_budget_s"),
        "objective": f"{hyperopt.get('split', 'validation')} {hyperopt.get('metric')}",
        "tuned_hyperparameters": sorted(hyperopt.get("parameters", {})),
        "feature_types": {kind: sorted(names) for kind, names in sorted(types.items())},
    }


def main() -> None:
    # Ludwig is imported here, so this file (and its tests) load without the Ludwig environment.
    from ludwig.api import LudwigModel
    from ludwig.automl import create_auto_config
    from ludwig.hyperopt.execution import RayTuneExecutor
    from ludwig.hyperopt.run import hyperopt

    from src.automl_ludwig import final_training_config, positive_proba
    from src.train import best_f1_threshold, evaluate

    # Ludwig 0.11 bug (see src/automl_ludwig.py): re-evaluating trials the scheduler stopped early crashes the
    # whole run. The best trial is retrained and evaluated below instead, so that step is skipped.
    RayTuneExecutor._evaluate_best_model = staticmethod(lambda *args, **kwargs: None)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--time-limit-min", type=float, default=10, help="total search time (default 10 minutes)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"{output_dir} already has results. Delete it or pass a new --output-dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    time_limit_s = int(args.time_limit_min * 60)

    columns = [*FEATURES, TARGET]
    train_full = pd.read_parquet(PROCESSED_DIR / "train.parquet")[columns]
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")[columns]
    train, val = train_test_split(train_full, test_size=0.2, stratify=train_full[TARGET], random_state=RANDOM_STATE)
    print(f"AutoML data: {len(train):,} train / {len(val):,} validation / {len(test):,} test IPs (test not searched)")

    # 2. Ludwig designs everything: feature types, architecture, search space and search algorithm.
    config = create_auto_config(dataset=train, target=TARGET, time_limit_s=time_limit_s, random_seed=RANDOM_STATE)
    choices = describe_auto_config(config)
    (output_dir / "auto_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print("Ludwig AutoML chose:\n" + json.dumps(choices, indent=2))

    # 3. Run the search AutoML designed.
    results = hyperopt(config, training_set=train, validation_set=val, output_directory=str(output_dir),
                       experiment_name="auto", backend="local", random_seed=RANDOM_STATE,
                       skip_save_processed_input=True)
    trials = results.ordered_trials
    if not trials:
        raise SystemExit("No trial finished within the time limit. Try a larger --time-limit-min.")
    best_params = trials[0].parameters
    print(f"Best of {len(trials)} trials ({choices['objective']} = {trials[0].metric_score:.4f}): "
          f"{best_params}")

    # 4. Retrain the winner, choose the threshold on validation only, then score the test set once.
    best_config = final_training_config(config, best_params)
    with tempfile.TemporaryDirectory() as tmp:
        model = LudwigModel(best_config, backend="local", logging_level=30)
        model.train(training_set=train, validation_set=val, output_directory=tmp, random_seed=RANDOM_STATE,
                    skip_save_processed_input=True)
        val_proba = positive_proba(model.predict(dataset=val, output_directory=tmp)[0])
        threshold, val_f1 = best_f1_threshold(val[TARGET].to_numpy(), val_proba)
        val_metrics = evaluate(val[TARGET].to_numpy(), val_proba, threshold)
        test_proba = positive_proba(model.predict(dataset=test, output_directory=tmp)[0])
        test_metrics = evaluate(test[TARGET].to_numpy(), test_proba, threshold)
        model.save(str(output_dir / "best_model"))

    summary = {"automl_choices": choices, "trials_finished": len(trials), "best_params": best_params,
               "best_trial_score": trials[0].metric_score, "threshold": threshold, "val_f1": val_f1,
               "val_roc_auc": val_metrics["roc_auc"], "test": test_metrics,
               "n_train": len(train), "n_validation": len(val), "n_test": len(test), "time_limit_s": time_limit_s}
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    (output_dir / "best_config.yaml").write_text(yaml.safe_dump(best_config, sort_keys=False))

    # 5. One MLflow run, in the same experiment as src/train.py, so it sits next to the scikit-learn models.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tags({"automl": "ludwig_auto_config", "architecture": choices["architecture"],
                         "search_algorithm": choices["search_algorithm"]})
        mlflow.log_params({"model": f"ludwig_{choices['architecture']}", "n_features": len(FEATURES),
                           "n_train": len(train), "time_limit_s": time_limit_s, "trials_finished": len(trials),
                           "max_trials": choices["max_trials"], "objective": choices["objective"],
                           **{f"best.{k}": v for k, v in best_params.items()}})
        mlflow.log_metrics({"val_f1": val_f1, "val_roc_auc": val_metrics["roc_auc"], "threshold": threshold,
                            **{f"test_{k}": v for k, v in test_metrics.items()}})
        for name in ("auto_config.yaml", "best_config.yaml", "results.json"):
            mlflow.log_artifact(str(output_dir / name))

    print(f"AutoML result: test F1 {test_metrics['f1']:.4f}, precision {test_metrics['precision']:.3f}, "
          f"recall {test_metrics['recall']:.3f}, threshold {threshold:.3f}. "
          f"Logged to MLflow as '{RUN_NAME}'; files in {output_dir}")


if __name__ == "__main__":
    main()
