"""Export the per-IP training table so real records can be picked in Excel and tried on the website.

Nothing is recalculated. Feature values are read from data/processed/*.parquet, the exact tables written
by src.prepare_data and used for training, so they can't drift from what the model saw.

Writes to data/exports/ (git-ignored):
    ip_level_aggregated_features.xlsx      every IP, with Instructions and Feature_guide sheets
    ip_level_aggregated_features.parquet   the same rows at full precision (the website loads this)
    ip_level_aggregated_features.csv       the same rows at full precision
    ip_level_sample_for_testing.xlsx       60 held-out IPs: 15 per outcome of the served model
    ip_level_sample_for_testing.csv        the same sample at full precision
    export_manifest.json                   row counts and hashes of the source tables

Usage:  python -m src.export_features
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import ID_COLUMN, MODEL_PATH, PROCESSED_DIR, RANDOM_STATE, ROOT, TARGET
from src.feature_docs import BINARY, COUNT, FEATURE_DOCS, FRACTION, KIND_HELP
from src.features import FEATURES
from src.model_io import load_bundle

EXPORT_DIR = ROOT / "data" / "exports"
FULL_NAME = "ip_level_aggregated_features"
SAMPLE_NAME = "ip_level_sample_for_testing"
MIN_REAL_IPS = 100_000  # the committed synthetic sample has only 300 IPs
ID_COLUMNS = [ID_COLUMN, TARGET, "label_name", "split", "ludwig_split", "cv_fold"]
SAMPLE_COLUMNS = [ID_COLUMN, TARGET, "label_name", "split", "served_score", "served_prediction", "served_outcome"]
OUTCOMES = {(1, 1): "true positive", (0, 1): "false alarm", (1, 0): "missed", (0, 0): "true negative"}
N_PER_OUTCOME = 15


def build_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """One row per IP, in the model's feature order, with where each IP went during training.

    The fold and validation assignments repeat the calls in src/train.py and src/automl_ludwig.py.
    """
    for name, df in (("train", train), ("test", test)):
        missing = [c for c in [ID_COLUMN, *FEATURES, TARGET] if c not in df]
        if missing:
            raise ValueError(f"{name} table is missing columns: {missing}")
    if set(train[ID_COLUMN]) & set(test[ID_COLUMN]):
        raise ValueError("an IP appears in both train and test")

    y = train[TARGET].to_numpy()
    cv_fold = np.zeros(len(train), dtype=int)
    splitter = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (_, val) in enumerate(splitter.split(train[FEATURES], y)):
        cv_fold[val] = fold
    _, ludwig_val = train_test_split(np.arange(len(train)), test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    ludwig = np.full(len(train), "train", dtype=object)
    ludwig[ludwig_val] = "validation"

    first = train[[ID_COLUMN, *FEATURES, TARGET]].assign(split="train", ludwig_split=ludwig, cv_fold=cv_fold)
    second = test[[ID_COLUMN, *FEATURES, TARGET]].assign(split="test", ludwig_split="test", cv_fold=-1)
    table = pd.concat([first, second], ignore_index=True)
    table["cv_fold"] = table["cv_fold"].astype("Int64").where(table["split"] == "train")
    table["label_name"] = np.where(table[TARGET] == 1, "VPN/proxy", "Not VPN/proxy")
    return table[[*ID_COLUMNS, *FEATURES]].sort_values(ID_COLUMN, ignore_index=True)


def build_sample(table: pd.DataFrame, bundle: dict, per_outcome: int = N_PER_OUTCOME) -> pd.DataFrame:
    """Held-out IPs covering every outcome of the served model, so testing isn't limited to easy cases."""
    test = table[table["split"] == "test"].reset_index(drop=True)
    score = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]
    pred = (score >= bundle["threshold"]).astype(int)
    test = test.assign(served_score=score, served_prediction=pred,
                       served_outcome=[OUTCOMES[(int(y), int(p))] for y, p in zip(test[TARGET], pred, strict=True)])
    chosen = [group.sample(min(per_outcome, len(group)), random_state=RANDOM_STATE)
              for _, group in test.groupby("served_outcome")]
    sample = pd.concat(chosen).sort_values(["served_outcome", "served_score"], ignore_index=True)
    sample = sample.assign(api_json=[json.dumps({f: float(row[f]) for f in FEATURES}) for _, row in sample.iterrows()])
    return sample[[*SAMPLE_COLUMNS, *FEATURES, "api_json"]]


def feature_guide(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position, name in enumerate(FEATURES, start=1):
        doc, values = FEATURE_DOCS[name], table[name]
        rows.append({"position": position, "feature": name, "kind": doc["kind"], "unit": doc["unit"],
                     "meaning": doc["meaning"], "how_to_read_it": doc["interpret"],
                     "uses_whole_population": "yes" if doc.get("population") else "no",
                     "min": float(values.min()), "median": float(values.median()),
                     "p99": float(values.quantile(0.99)), "max": float(values.max())})
    return pd.DataFrame(rows)


def _instructions(sample: bool, n_rows: int, model: str) -> list[str]:
    lines = [
        "CrowdSec VPN/proxy classifier: IP-level feature export" + (" (small sample)" if sample else ""),
        "",
        "WHAT THIS IS",
        f"{n_rows:,} rows, one per attacker IP. Values are read unchanged from data/processed/train.parquet and "
        "test.parquet, the tables used to train the model.",
        "This is real CrowdSec data, not the synthetic sample. attacker_ip_enum is the dataset's anonymised ID, "
        "so a row can't be mapped back to a real IP address. Raw attack signals aren't included.",
        "",
        "HOW TO TEST AN IP ON THE WEBSITE",
        "1. Start the API and dashboard, then open the IP Detection page.",
        "2. Most exact: enter the attacker_ip_enum under 'Load a real record'. It reads every value at full precision.",
        "3. Or select the row's 28 feature cells here, copy them, and use 'Paste a row from Excel'.",
        "4. Or type the values into the form by hand, using the Feature_guide sheet.",
        "Cells display rounded values, and Excel copies what it displays, so a pasted score can differ slightly from "
        "the exact one.",
        "",
        "CHOOSING ROWS",
        "Prefer rows where split = test. The model never trained on them, so they show realistic behaviour.",
        "Rows where split = train were learned from, so their scores look better than the model really is.",
        "Filter on label = 1 for VPN/proxy IPs. They are rare, about 1.95% of rows.",
        "",
        "COLUMNS",
        "label: the dataset's answer: 1 = VPN/proxy, 0 = not. It is not sent to the model.",
        "split: train or test, from src/prepare_data.py (stratified 80/20, random_state 42).",
        "ludwig_split: train/validation/test as used by the Ludwig search. The scikit-learn models have no fixed "
        "validation set.",
        "cv_fold: which of the 5 cross-validation folds this IP validated in. Empty for test rows.",
        "Then the 28 features, in the exact order the API expects. See the Feature_guide sheet.",
        "",
        "LIMITS",
        "The model scores these features, not an IP address: an arbitrary IP can't be classified from its address.",
        "A new IP needs its signals aggregated by src/features.py, and Shodan data, to get these values.",
        "asn_unique_ips and the three *_shared features were counted across the whole dataset, so they can't be "
        "worked out from a single IP.",
        "Feature values describe behaviour. They don't prove why an IP was labelled VPN/proxy.",
        "",
        "REGENERATE",
        r"From the project root: .venv\Scripts\python -m src.export_features",
    ]
    if sample:
        at = lines.index("COLUMNS")
        lines[at:at] = [
            "THIS SAMPLE",
            f"Held-out test IPs only, up to {N_PER_OUTCOME} per outcome of the served model ({model}).",
            "served_outcome compares the model's answer with the label: true positive, false alarm, missed, "
            "true negative.",
            "api_json holds the 28 values at full precision, ready to paste into POST /predict at /docs.",
            "",
        ]
    return lines


def _number_format(workbook, name: str):
    kind = FEATURE_DOCS[name]["kind"]
    if kind in (COUNT, BINARY):
        return workbook.add_format({"num_format": "#,##0"})
    return workbook.add_format({"num_format": "0.000000" if kind == FRACTION else "#,##0.000000"})


def write_workbook(table: pd.DataFrame, guide: pd.DataFrame, path: Path, sample: bool = False, model: str = "") -> None:
    import xlsxwriter

    workbook = xlsxwriter.Workbook(str(path))
    header = workbook.add_format({"bold": True, "bg_color": "#1F3A5F", "font_color": "#FFFFFF", "text_wrap": True,
                                  "valign": "top", "border": 1})
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    title = workbook.add_format({"bold": True, "font_size": 14})
    section = workbook.add_format({"bold": True, "font_color": "#1F3A5F"})

    notes = workbook.add_worksheet("Instructions")
    notes.set_column(0, 0, 120, wrap)
    for row, line in enumerate(_instructions(sample, len(table), model)):
        notes.write_string(row, 0, line, title if row == 0 else section if line.isupper() else wrap)

    sheet = workbook.add_worksheet("IP_features")
    formats = {name: _number_format(workbook, name) for name in FEATURES}
    score = workbook.add_format({"num_format": "0.000000"})
    columns = table.columns.tolist()
    for col, name in enumerate(columns):
        sheet.write_string(0, col, name, header)
        width = 60 if name == "api_json" else max(11, min(len(name) + 2, 24))
        sheet.set_column(col, col, width, formats.get(name) or (score if name == "served_score" else None))
        values = table[name]
        if name in FEATURES or name == "served_score":
            sheet.write_column(1, col, values.astype(float).tolist())
        elif name == "cv_fold":
            sheet.write_column(1, col, [None if pd.isna(v) else int(v) for v in values])
        elif pd.api.types.is_integer_dtype(values):
            sheet.write_column(1, col, values.astype(int).tolist())
        else:
            sheet.write_column(1, col, values.astype(str).tolist())
    sheet.set_row(0, 45)
    sheet.freeze_panes(1, 1)
    sheet.autofilter(0, 0, len(table), len(columns) - 1)

    sheet = workbook.add_worksheet("Feature_guide")
    plain = workbook.add_format({"num_format": "#,##0.####"})
    widths = {"position": 8, "feature": 28, "kind": 10, "unit": 26, "meaning": 60, "how_to_read_it": 70,
              "uses_whole_population": 12}
    for col, name in enumerate(guide.columns):
        sheet.write_string(0, col, name, header)
        is_text = not pd.api.types.is_numeric_dtype(guide[name])
        sheet.set_column(col, col, widths.get(name, 13), wrap if is_text else plain)
        sheet.write_column(1, col, guide[name].tolist())
    sheet.freeze_panes(1, 2)
    sheet.autofilter(0, 0, len(guide), len(guide.columns) - 1)
    row = len(guide) + 2
    sheet.write_string(row, 1, "Kinds of value", section)
    for offset, (kind, text) in enumerate(KIND_HELP.items(), start=1):
        sheet.write_string(row + offset, 1, kind)
        sheet.write_string(row + offset, 4, text)
    workbook.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(processed_dir: Path = PROCESSED_DIR, out_dir: Path = EXPORT_DIR, model_path: Path = MODEL_PATH,
           allow_small: bool = False) -> dict:
    train_path, test_path = processed_dir / "train.parquet", processed_dir / "test.parquet"
    train, test = pd.read_parquet(train_path), pd.read_parquet(test_path)
    if len(train) + len(test) < MIN_REAL_IPS and not allow_small:
        raise RuntimeError(f"{processed_dir} holds only {len(train) + len(test)} IPs: this looks like the synthetic "
                           "sample, not the real data. Run src.prepare_data on the real data first.")
    table = build_table(train, test)
    guide = feature_guide(table)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"{FULL_NAME}.parquet", index=False)
    table.to_csv(out_dir / f"{FULL_NAME}.csv", index=False)
    write_workbook(table, guide, out_dir / f"{FULL_NAME}.xlsx")

    bundle = load_bundle(model_path)
    meta = bundle["metadata"]
    model = f"{meta['model_name']}-{meta['version']}"
    sample = build_sample(table, bundle)
    sample.drop(columns="api_json").to_csv(out_dir / f"{SAMPLE_NAME}.csv", index=False)
    write_workbook(sample, guide, out_dir / f"{SAMPLE_NAME}.xlsx", sample=True, model=model)

    manifest = {"rows": len(table), "columns": table.columns.tolist(), "features": FEATURES,
                "positive_rate": float(table[TARGET].mean()), "rows_per_split": table["split"].value_counts().to_dict(),
                "rows_per_ludwig_split": table["ludwig_split"].value_counts().to_dict(), "sample_rows": len(sample),
                "sample_outcomes": sample["served_outcome"].value_counts().to_dict(), "sample_scored_by": model,
                "source_sha256": {"train.parquet": _sha256(train_path), "test.parquet": _sha256(test_path)}}
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--allow-small", action="store_true", help="allow a small table (for tests only)")
    args = parser.parse_args()
    manifest = export(args.processed_dir, args.out_dir, allow_small=args.allow_small)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("columns", "features")}, indent=2))
    for path in sorted(args.out_dir.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.1f} MB  {path.name}")


if __name__ == "__main__":
    main()
