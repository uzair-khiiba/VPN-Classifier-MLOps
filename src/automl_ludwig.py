"""AutoML with Ludwig: hyperparameter search over a tabular neural network.

Uses configs/ludwig_automl.yaml on the same per-IP features and the same held-out
test IPs as src/train.py, and logs the best trial to the same MLflow experiment, so
the Ludwig model can be compared with the scikit-learn runs in the MLflow UI.

Steps:
  1. Split the training IPs into train/validation (stratified); the test set is untouched.
  2. Ludwig hyperopt (Ray Tune + TPE) searches the config's `hyperopt.parameters`,
     scoring each trial on validation ROC-AUC.
  3. Retrain the best trial's config, tune the F1 threshold on validation, and report
     test metrics with the same functions src/train.py uses.

Usage (needs the Ludwig environment, see requirements-ludwig.txt):
    python -m src.automl_ludwig                      # full search from the config
    python -m src.automl_ludwig --num-samples 4      # quicker search
"""

import argparse
import copy
import shutil
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, PROCESSED_DIR, RANDOM_STATE, ROOT, TARGET
from src.features import FEATURES
from src.train import best_f1_threshold, evaluate

CONFIG_PATH = ROOT / "configs" / "ludwig_automl.yaml"
OUTPUT_DIR = ROOT / "ludwig_results"


def load_config(path: Path = CONFIG_PATH) -> dict:
    config = yaml.safe_load(path.read_text())
    names = [f["name"] for f in config["input_features"]]
    missing, extra = set(FEATURES) - set(names), set(names) - set(FEATURES)
    if missing or extra or len(names) != len(set(names)):
        raise ValueError(f"{path} input_features out of sync with src/features.py "
                         f"(missing={sorted(missing)}, extra={sorted(extra)}, or duplicates)")
    return config


def apply_params(config: dict, params: dict) -> dict:
    """Write hyperopt-style dotted params (e.g. 'trainer.learning_rate',
    'label.loss.positive_class_weight') into a copy of the config."""
    config = copy.deepcopy(config)
    config.pop("hyperopt", None)
    features = {f["name"]: f for f in config["input_features"] + config["output_features"]}
    for key, value in params.items():
        head, *rest = key.split(".")
        node = features[head] if head in features else config.setdefault(head, {})
        for part in rest[:-1]:
            node = node.setdefault(part, {})
        node[rest[-1]] = value.item() if isinstance(value, np.generic) else value
    return config


def positive_proba(predictions: pd.DataFrame) -> np.ndarray:
    """Probability of label=True from a Ludwig predictions frame (column names vary by version)."""
    if hasattr(predictions, "compute"):  # Dask frame from Ludwig's Ray backend
        predictions = predictions.compute()
    for col in (f"{TARGET}_probabilities_True", f"{TARGET}_probabilities_1"):
        if col in predictions:
            return predictions[col].to_numpy(dtype=float)
    if f"{TARGET}_probabilities" in predictions:
        return np.array([p[-1] for p in predictions[f"{TARGET}_probabilities"]], dtype=float)
    prob, pred = predictions[f"{TARGET}_probability"], predictions[f"{TARGET}_predictions"].astype(bool)
    return np.where(pred, prob, 1 - prob).astype(float)


def main() -> None:
    from ludwig.api import LudwigModel  # imported here: Ludwig lives in its own environment
    from ludwig.hyperopt.run import hyperopt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--num-samples", type=int, default=None, help="override hyperopt.executor.num_samples")
    parser.add_argument("--time-budget-s", type=int, default=None, help="override hyperopt.executor.time_budget_s")
    args = parser.parse_args()

    config = load_config(args.config)
    executor = config["hyperopt"]["executor"]
    if args.num_samples:
        executor["num_samples"] = args.num_samples
    if args.time_budget_s:
        executor["time_budget_s"] = args.time_budget_s

    columns = [*FEATURES, TARGET]
    train_full = pd.read_parquet(PROCESSED_DIR / "train.parquet")[columns]
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")[columns]
    train, val = train_test_split(train_full, test_size=0.2, stratify=train_full[TARGET], random_state=RANDOM_STATE)

    # Ludwig refuses to z-score a constant column, and it carries no signal anyway.
    constant = [f for f in FEATURES if train[f].nunique() <= 1]
    if constant:
        print(f"Dropping constant features for Ludwig: {constant}")
        config["input_features"] = [f for f in config["input_features"] if f["name"] not in constant]
    print(f"Ludwig AutoML: {len(train):,} train / {len(val):,} validation / {len(test):,} test IPs, "
          f"{executor['num_samples']} trials")

    OUTPUT_DIR.mkdir(exist_ok=True)
    results = hyperopt(config, training_set=train, validation_set=val, output_directory=str(OUTPUT_DIR),
                       experiment_name="hyperopt", backend="local", random_seed=RANDOM_STATE,
                       skip_save_processed_input=True)
    trials = results.ordered_trials
    best_params = trials[0].parameters
    print(f"Best trial (val {config['hyperopt']['metric']}={trials[0].metric_score:.4f}): {best_params}")

    best_config = apply_params(config, best_params)
    with tempfile.TemporaryDirectory() as tmp:
        model = LudwigModel(best_config, backend="local", logging_level=30)
        model.train(training_set=train, validation_set=val, output_directory=tmp,
                    random_seed=RANDOM_STATE, skip_save_processed_input=True)
        val_proba = positive_proba(model.predict(dataset=val, output_directory=tmp)[0])
        test_proba = positive_proba(model.predict(dataset=test, output_directory=tmp)[0])

        threshold, val_f1 = best_f1_threshold(val[TARGET].to_numpy(), val_proba)
        test_metrics = evaluate(test[TARGET].to_numpy(), test_proba, threshold)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="ludwig_automl"):
            mlflow.set_tag("automl", "ludwig")
            mlflow.log_params({"model": "ludwig_ecd", "n_features": len(FEATURES), "n_train": len(train),
                               "n_trials": len(trials), **{f"best.{k}": v for k, v in best_params.items()}})
            mlflow.log_metrics({"val_f1": val_f1, f"val_{config['hyperopt']['metric']}": trials[0].metric_score,
                                "threshold": threshold, **{f"test_{k}": v for k, v in test_metrics.items()}})
            mlflow.log_dict(best_config, "ludwig_best_config.yaml")
            mlflow.log_dict({"trials": [{"score": t.metric_score, **t.parameters} for t in trials]},
                            "ludwig_trials.json")
            model_dir = OUTPUT_DIR / "best_model"
            shutil.rmtree(model_dir, ignore_errors=True)
            model.save(str(model_dir))

    print(f"Ludwig best: val_f1={val_f1:.4f}  test_f1={test_metrics['f1']:.4f}  "
          f"test_pr_auc={test_metrics['pr_auc']:.4f}  threshold={threshold:.3f}  (model saved to {model_dir})")


if __name__ == "__main__":
    main()
