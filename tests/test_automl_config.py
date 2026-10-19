"""Checks on the Ludwig AutoML config that do not need Ludwig installed."""

import pytest
import yaml

from src.automl_ludwig import CONFIG_PATH, apply_params, load_config, positive_proba
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
