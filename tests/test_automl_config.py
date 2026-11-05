"""Checks on the Ludwig AutoML config that do not need Ludwig installed."""

import pytest
import yaml

from src.automl_ludwig import (
    CONFIG_PATH,
    FIRST_SEARCH_DIR,
    OUTPUT_DIR,
    apply_params,
    final_training_config,
    load_config,
    positive_proba,
    search_budget,
)
from src.config import ROOT
from src.features import FEATURES


def test_config_features_match_pipeline():
    config = load_config()
    assert sorted(f["name"] for f in config["input_features"]) == sorted(FEATURES)
    assert config["output_features"][0] == {**config["output_features"][0], "name": "label", "type": "binary"}


def test_config_rejects_out_of_sync_features(tmp_path):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["input_features"].pop()
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="out of sync"):
        load_config(path)


def test_hyperopt_params_point_at_existing_sections():
    config = load_config()
    names = {f["name"] for f in config["input_features"] + config["output_features"]}
    for key in config["hyperopt"]["parameters"]:
        head = key.split(".")[0]
        assert head in names or head in config, key


def test_apply_params_writes_nested_and_feature_params():
    config = apply_params(load_config(), {"trainer.learning_rate": 0.005, "label.loss.positive_class_weight": 20})
    assert "hyperopt" not in config
    assert config["trainer"]["learning_rate"] == 0.005
    assert config["output_features"][0]["loss"]["positive_class_weight"] == 20


def test_positive_proba_handles_column_variants():
    import pandas as pd

    assert positive_proba(pd.DataFrame({"label_probabilities_True": [0.2, 0.9]})).tolist() == [0.2, 0.9]
    assert positive_proba(pd.DataFrame({"label_probabilities": [[0.8, 0.2]]})).tolist() == [0.2]
    framed = pd.DataFrame({"label_probability": [0.7, 0.9], "label_predictions": [False, True]})
    assert positive_proba(framed).tolist() == pytest.approx([0.3, 0.9])


FIRST_CONFIG = ROOT / "configs" / "ludwig_automl.yaml"


def test_first_search_never_reached_the_adaptive_stage():
    assert search_budget(load_config(FIRST_CONFIG))["adaptive_trials"] == 0


def test_new_search_has_room_for_adaptive_trials():
    budget = search_budget(load_config())
    assert budget == {"num_samples": 30, "n_initial_points": 10, "random_trials": 10, "adaptive_trials": 20}


def test_new_search_optimises_pr_auc_and_changes_only_the_search_settings():
    new, old = load_config(), load_config(FIRST_CONFIG)
    assert new["hyperopt"]["metric"] == new["trainer"]["validation_metric"] == "pr_auc"
    for section in ("input_features", "output_features", "combiner"):
        assert new[section] == old[section], section
    assert new["hyperopt"]["parameters"] == old["hyperopt"]["parameters"]


def test_final_training_matches_how_trials_ran():
    final = final_training_config(load_config(), {"trainer.learning_rate": 0.01, "combiner.dropout": 0.2})
    assert final["trainer"]["early_stop"] == -1 and final["trainer"]["learning_rate"] == 0.01
    assert final["combiner"]["dropout"] == 0.2 and "hyperopt" not in final


def test_new_results_go_to_a_separate_folder():
    assert OUTPUT_DIR != FIRST_SEARCH_DIR and OUTPUT_DIR.name == "ludwig_results_v2"
