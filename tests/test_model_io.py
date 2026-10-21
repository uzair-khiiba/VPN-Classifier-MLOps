import json

import joblib
import pandas as pd
import pytest

from src.config import MODEL_PATH, ROOT
from src.features import FEATURES
from src.model_io import ModelLoadError, load_bundle, save_bundle


def test_load_bundle_roundtrip(model_path):
    bundle = load_bundle(model_path)
    assert bundle["threshold"] == 0.5
    assert len(bundle["features"]) == len(bundle["reference"])
    assert hasattr(bundle["model"], "predict_proba")


def test_load_bundle_missing_file(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_bundle(tmp_path / "nope.joblib")


def test_load_bundle_rejects_non_bundle(tmp_path):
    path = tmp_path / "bad.joblib"
    joblib.dump({"model": "x"}, path)
    with pytest.raises(ModelLoadError, match="not a valid model bundle"):
        load_bundle(path)


def test_load_bundle_rejects_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"not a pickle")
    with pytest.raises(ModelLoadError, match="could not load"):
        load_bundle(path)


def test_load_bundle_rejects_bad_threshold(model_path, tmp_path):
    bundle = load_bundle(model_path)
    bundle["threshold"] = 1.5
    path = save_bundle(bundle, tmp_path / "bad_threshold.joblib")
    with pytest.raises(ModelLoadError, match="invalid threshold"):
        load_bundle(path)


def test_save_bundle_requires_all_keys(tmp_path):
    with pytest.raises(ValueError, match="missing keys"):
        save_bundle({"model": object()}, tmp_path / "m.joblib")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="no trained model in models/")
def test_committed_model_loads_and_predicts_examples():
    """The model the API serves (models/model.joblib) loads and gets both example IPs right."""
    bundle = load_bundle(MODEL_PATH)
    assert bundle["features"] == FEATURES
    for name, expected in (("vpn_ip.json", True), ("regular_ip.json", False)):
        row = json.loads((ROOT / "examples" / name).read_text())
        proba = bundle["model"].predict_proba(pd.DataFrame([row])[bundle["features"]])[0, 1]
        assert bool(proba >= bundle["threshold"]) is expected, name
