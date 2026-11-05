"""The IP-level export must carry the training values unchanged, in the API's feature order."""

import numpy as np
import openpyxl
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import app as app_module
from src.config import ID_COLUMN, ROOT, TARGET
from src.export_features import FULL_NAME, ID_COLUMNS, SAMPLE_NAME, build_table, export
from src.feature_docs import FEATURE_DOCS, POPULATION_FEATURES
from src.features import FEATURES
from src.model_io import load_bundle

EXPORT_DIR = ROOT / "data" / "exports"


def make_split(n_train: int = 200, n_test: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.random((n_train + n_test, len(FEATURES))), columns=FEATURES)
    frame[ID_COLUMN] = rng.permutation(n_train + n_test)
    frame[TARGET] = (np.arange(n_train + n_test) % 4 == 0).astype(int)
    return frame.iloc[:n_train].reset_index(drop=True), frame.iloc[n_train:].reset_index(drop=True)


def test_feature_docs_cover_every_feature_with_a_known_kind():
    assert set(FEATURE_DOCS) == set(FEATURES)
    assert {doc["kind"] for doc in FEATURE_DOCS.values()} == {"count", "fraction", "binary", "duration", "rate"}
    assert POPULATION_FEATURES == ["asn_unique_ips", "jarm_shared", "ja3s_shared", "headers_hash_shared"]


def test_table_keeps_feature_order_and_values():
    train, test = make_split()
    table = build_table(train, test)
    assert table.columns.tolist() == [*ID_COLUMNS, *FEATURES]
    merged = pd.concat([train, test]).sort_values(ID_COLUMN, ignore_index=True)
    assert (table[FEATURES].to_numpy() == merged[FEATURES].to_numpy()).all()
    assert (table[TARGET].to_numpy() == merged[TARGET].to_numpy()).all()


def test_table_assigns_splits_like_training():
    train, test = make_split()
    table = build_table(train, test)
    assert table["split"].value_counts().to_dict() == {"train": 200, "test": 50}
    assert table["ludwig_split"].value_counts().to_dict() == {"train": 160, "test": 50, "validation": 40}
    assert table.loc[table["split"] == "test", "cv_fold"].isna().all()
    assert table.loc[table["split"] == "train", "cv_fold"].between(0, 4).all()


def test_table_rejects_an_ip_in_both_splits():
    train, test = make_split()
    test.loc[0, ID_COLUMN] = train.loc[0, ID_COLUMN]
    with pytest.raises(ValueError, match="both train and test"):
        build_table(train, test)


def test_small_table_is_refused_as_probably_synthetic(tmp_path, model_path):
    train, test = make_split()
    train.to_parquet(tmp_path / "train.parquet")
    test.to_parquet(tmp_path / "test.parquet")
    with pytest.raises(RuntimeError, match="synthetic"):
        export(tmp_path, tmp_path / "out", model_path)


def test_workbook_is_filterable_and_explained(tmp_path, model_path):
    train, test = make_split()
    train.to_parquet(tmp_path / "train.parquet")
    test.to_parquet(tmp_path / "test.parquet")
    out = tmp_path / "out"
    manifest = export(tmp_path, out, model_path, allow_small=True)
    assert manifest["rows"] == 250

    book = openpyxl.load_workbook(out / f"{FULL_NAME}.xlsx")
    assert book.sheetnames == ["Instructions", "IP_features", "Feature_guide"]
    sheet = book["IP_features"]
    assert sheet.freeze_panes == "B2" and sheet.auto_filter.ref
    assert [cell.value for cell in sheet[1]] == [*ID_COLUMNS, *FEATURES]
    guide = [row[1].value for row in book["Feature_guide"].iter_rows(min_row=2, max_row=29)]
    assert guide == FEATURES
    text = " ".join(str(row[0].value) for row in book["Instructions"].iter_rows())
    assert "split = test" in text and "can't be classified from its address" in text

    exact = pd.read_parquet(out / f"{FULL_NAME}.parquet")
    excel = pd.read_excel(out / f"{FULL_NAME}.xlsx", sheet_name="IP_features")
    assert np.allclose(excel[FEATURES].to_numpy(), exact[FEATURES].to_numpy(), rtol=1e-12)
    assert (out / f"{SAMPLE_NAME}.xlsx").exists()


real_export = pytest.mark.skipif(not (EXPORT_DIR / f"{FULL_NAME}.parquet").exists(),
                                 reason="the export isn't generated")


@real_export
def test_real_export_rows_match_training_tables_and_real_size():
    table = pd.read_parquet(EXPORT_DIR / f"{FULL_NAME}.parquet")
    assert len(table) == 148259 and table[ID_COLUMN].is_unique
    assert table.columns[len(ID_COLUMNS):].tolist() == FEATURES
    processed = pd.concat([pd.read_parquet(ROOT / "data" / "processed" / f"{name}.parquet")
                           for name in ("train", "test")]).sort_values(ID_COLUMN, ignore_index=True)
    assert (table[FEATURES].to_numpy() == processed[FEATURES].to_numpy()).all()


@real_export
def test_real_export_row_gets_the_same_answer_through_the_api():
    table = pd.read_parquet(EXPORT_DIR / f"{FULL_NAME}.parquet")
    rows = table[table["split"] == "test"].groupby(TARGET).head(3)
    bundle = load_bundle(ROOT / "models" / "model.joblib")
    app_module.load_model(ROOT / "models" / "model.joblib")
    try:
        client = TestClient(app_module.app)
        for _, row in rows.iterrows():
            answer = client.post("/predict", json={f: float(row[f]) for f in FEATURES})
            assert answer.status_code == 200
            direct = bundle["model"].predict_proba(pd.DataFrame([row[FEATURES].astype(float)]))[0, 1]
            assert answer.json()["probability"] == round(float(direct), 4)
    finally:
        app_module.STATE.update(bundle=None, error=None)
