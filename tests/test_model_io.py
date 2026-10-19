import joblib
import pytest

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
