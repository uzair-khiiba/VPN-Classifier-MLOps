"""The simple AutoML runner: it loads without Ludwig, and reports what Ludwig's AutoML chose."""

import ast
from pathlib import Path

from src import automl_simple

AUTO_CONFIG = {  # the shape of what ludwig.automl.create_auto_config returns
    "input_features": [{"name": "n_signals", "type": "number"}, {"name": "n_attack_types", "type": "category"},
                       {"name": "has_shodan", "type": "binary"}, {"name": "asn_unique_ips", "type": "number"}],
    "output_features": [{"name": "label", "type": "binary"}],
    "combiner": {"type": "tabnet"},
    "hyperopt": {"metric": "roc_auc", "split": "validation", "goal": "maximize",
                 "parameters": {"trainer.learning_rate": {}, "combiner.size": {}},
                 "search_alg": {"type": "variant_generator", "random_state_seed": 42},
                 "executor": {"type": "ray", "num_samples": 10, "time_budget_s": 300,
                              "scheduler": {"type": "async_hyperband"}}},
}


def test_describes_what_automl_chose():
    choices = automl_simple.describe_auto_config(AUTO_CONFIG)
    assert choices["architecture"] == "tabnet"
    assert choices["search_algorithm"] == "variant_generator"  # random search
    assert choices["early_stopping_scheduler"] == "async_hyperband"
    assert choices["max_trials"] == 10 and choices["time_budget_s"] == 300
    assert choices["objective"] == "validation roc_auc"
    assert choices["tuned_hyperparameters"] == ["combiner.size", "trainer.learning_rate"]
    assert choices["feature_types"] == {"binary": ["has_shodan"], "category": ["n_attack_types"],
                                        "number": ["asn_unique_ips", "n_signals"]}


def test_missing_sections_do_not_crash():
    assert automl_simple.describe_auto_config({})["architecture"] is None


def test_ludwig_is_only_imported_inside_main():
    """The API and the test suite run without Ludwig installed, so no module-level import may need it."""
    tree = ast.parse(Path(automl_simple.__file__).read_text(encoding="utf-8"))
    top_level = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
    names = [alias.name for node in top_level for alias in node.names] + \
            [node.module for node in top_level if isinstance(node, ast.ImportFrom) and node.module]
    assert not [name for name in names if name.startswith("ludwig")]
