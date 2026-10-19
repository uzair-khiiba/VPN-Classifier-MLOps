"""Data preparation: validate raw signals, build per-IP features, split train/test.

Usage:
    python -m src.prepare_data                 # full data in data/raw/
    python -m src.prepare_data --sample        # small committed sample in data/sample/
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ID_COLUMN,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DIR,
    SAMPLE_DIR,
    SHODAN_FILE,
    SIGNALS_FILE,
    TARGET,
    TEST_SIZE,
)
from src.features import FEATURES, build_features, connect

REQUIRED_SIGNAL_COLUMNS = {
    ID_COLUMN,
    "attack_time",
    "attack_type",
    "attacker_as_num",
    "watcher_as_num",
    "watcher_country",
    "watcher_uuid_enum",
    TARGET,
}


class DataValidationError(ValueError):
    """Raised when the raw data fails a validation check."""


def validate_signals(signals_path: Path) -> dict:
    """Check schema, missing values, label values and label consistency per IP.

    Returns a small report dict; raises DataValidationError on a hard failure.
    """
    con = connect()
    try:
        src = f"read_parquet('{signals_path}')"
        columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}
        missing = REQUIRED_SIGNAL_COLUMNS - columns
        if missing:
            raise DataValidationError(f"missing columns: {sorted(missing)}")

        null_counts = con.execute(
            "SELECT " + ", ".join(f"count(*) - count({c}) AS {c}" for c in sorted(REQUIRED_SIGNAL_COLUMNS))
            + f", count(*) AS n_rows FROM {src}"
        ).df().iloc[0].to_dict()
        n_rows = int(null_counts.pop("n_rows"))
        if n_rows == 0:
            raise DataValidationError("signals file is empty")
        for col in (ID_COLUMN, "attack_time", "attack_type", TARGET):
            if null_counts[col]:
                raise DataValidationError(f"{null_counts[col]} null values in required column {col}")

        bad_labels = con.execute(f"SELECT count(*) FROM {src} WHERE {TARGET} NOT IN (0, 1)").fetchone()[0]
        if bad_labels:
            raise DataValidationError(f"{bad_labels} rows have a label other than 0/1")

        inconsistent = con.execute(
            f"SELECT count(*) FROM (SELECT {ID_COLUMN} FROM {src} GROUP BY {ID_COLUMN} "
            f"HAVING count(DISTINCT {TARGET}) > 1)"
        ).fetchone()[0]
        if inconsistent:
            raise DataValidationError(f"{inconsistent} IPs have conflicting labels")

        # Cast in SQL: returning TIMESTAMPTZ values to Python would require pytz.
        t_min, t_max, ordered = con.execute(
            f"SELECT min(attack_time)::VARCHAR, max(attack_time)::VARCHAR, min(attack_time) <= max(attack_time) "
            f"FROM {src}"
        ).fetchone()
        if not ordered:
            raise DataValidationError("invalid attack_time range")
    finally:
        con.close()

    return {
        "n_rows": n_rows,
        "null_counts": {k: int(v) for k, v in null_counts.items()},
        "time_range": [t_min, t_max],
    }


def validate_features(feats: pd.DataFrame) -> None:
    """Sanity checks on the per-IP table: one row per IP, no NaNs, no negative values."""
    if feats[ID_COLUMN].duplicated().any():
        raise DataValidationError("duplicate IPs in feature table")
    if feats[FEATURES].isna().any().any():
        bad = feats[FEATURES].columns[feats[FEATURES].isna().any()].tolist()
        raise DataValidationError(f"NaN values in features: {bad}")
    if (feats[FEATURES] < 0).any().any():
        bad = feats[FEATURES].columns[(feats[FEATURES] < 0).any()].tolist()
        raise DataValidationError(f"negative values in features: {bad}")


def split_by_ip(feats: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split. The table has one row per IP, so no IP lands in both sets."""
    train, test = train_test_split(
        feats, test_size=test_size, stratify=feats[TARGET], random_state=RANDOM_STATE
    )
    assert not set(train[ID_COLUMN]) & set(test[ID_COLUMN]), "IP leakage between train and test"
    return train.reset_index(drop=True), test.reset_index(drop=True)


def prepare(raw_dir: Path, out_dir: Path) -> dict:
    signals_path = raw_dir / SIGNALS_FILE
    shodan_path = raw_dir / SHODAN_FILE
    if not signals_path.exists():
        raise FileNotFoundError(f"{signals_path} not found; see README.md for how to download the data")

    print(f"Validating {signals_path} ...")
    report = validate_signals(signals_path)
    print(f"  {report['n_rows']:,} signals, time range {report['time_range']}")

    shodan = pd.read_csv(shodan_path) if shodan_path.exists() else None
    if shodan is None:
        print(f"  {shodan_path} not found; Shodan features will be 0")

    print("Building per-IP features ...")
    feats = build_features(signals_path, shodan)
    validate_features(feats)

    train, test = split_by_ip(feats)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    report.update(
        n_ips=len(feats),
        positive_rate=round(float(feats[TARGET].mean()), 4),
        n_train=len(train),
        n_test=len(test),
        features=FEATURES,
    )
    (out_dir / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(f"  {len(feats):,} IPs ({report['positive_rate']:.2%} VPN/proxy) -> "
          f"{len(train):,} train / {len(test):,} test, written to {out_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", action="store_true", help="use the small sample in data/sample/")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    raw_dir = args.raw_dir or (SAMPLE_DIR if args.sample else RAW_DIR)
    prepare(raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
