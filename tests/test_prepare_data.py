import pandas as pd
import pytest

from src.features import FEATURES, build_shodan_features, build_signal_features, parse_shodan_info
from src.prepare_data import DataValidationError, prepare, split_by_ip, validate_features, validate_signals
from tests.conftest import make_shodan, make_signals


def _write(df, tmp_path):
    path = tmp_path / "train.parq"
    df.to_parquet(path, index=False)
    return path


def test_validate_signals_accepts_good_data(raw_dir):
    report = validate_signals(raw_dir / "train.parq")
    assert report["n_rows"] > 0
    assert report["null_counts"]["attacker_ip_enum"] == 0


def test_pipeline_handles_timezone_aware_timestamps(tmp_path):
    # The Kaggle file stores attack_time as TIMESTAMP WITH TIME ZONE.
    df = make_signals()
    df["attack_time"] = df["attack_time"].dt.tz_localize("Asia/Karachi")
    path = _write(df, tmp_path)
    report = validate_signals(path)
    assert report["time_range"][0] < report["time_range"][1]
    assert len(build_signal_features(path)) == df["attacker_ip_enum"].nunique()


def test_validate_signals_rejects_missing_column(tmp_path):
    path = _write(make_signals().drop(columns=["attack_type"]), tmp_path)
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_signals(path)


def test_validate_signals_rejects_invalid_label(tmp_path):
    df = make_signals()
    df.loc[0, "label"] = 2
    with pytest.raises(DataValidationError, match="label other than 0/1"):
        validate_signals(_write(df, tmp_path))


def test_validate_signals_rejects_null_ip(tmp_path):
    df = make_signals()
    df["attacker_ip_enum"] = df["attacker_ip_enum"].astype("Int64")
    df.loc[0, "attacker_ip_enum"] = pd.NA
    with pytest.raises(DataValidationError, match="null values"):
        validate_signals(_write(df, tmp_path))


def test_validate_signals_rejects_conflicting_labels(tmp_path):
    df = make_signals()
    first_ip_rows = df.index[df["attacker_ip_enum"] == 1]
    df.loc[first_ip_rows[0], "label"] = 1 - df.loc[first_ip_rows[0], "label"]
    with pytest.raises(DataValidationError, match="conflicting labels"):
        validate_signals(_write(df, tmp_path))


def test_signal_features_match_hand_computed_values(tmp_path):
    t0 = pd.Timestamp("2023-09-01 00:00")
    df = pd.DataFrame({
        "attacker_ip_enum": [7, 7, 7],
        "attack_time": [t0, t0 + pd.Timedelta(minutes=10), t0 + pd.Timedelta(hours=26)],
        "attack_type": ["http:scan", "http:scan", "ssh:bruteforce"],
        "attacker_as_num": [1.0, 1.0, 1.0],
        "watcher_as_num": [10.0, 11.0, 11.0],
        "watcher_country": ["FR", "FR", "DE"],
        "watcher_uuid_enum": [1, 2, 2],
        "label": [1, 1, 1],
    })
    row = build_signal_features(_write(df, tmp_path)).iloc[0]
    assert row["n_signals"] == 3
    assert row["n_watchers"] == 2
    assert row["n_watcher_countries"] == 2
    assert row["n_active_days"] == 2
    assert row["active_hours"] == pytest.approx(26)
    assert row["share_http_scan"] == pytest.approx(2 / 3)
    assert row["max_signals_per_hour"] == 2
    assert row["mean_gap_s"] == pytest.approx((600 + 25 * 3600 + 50 * 60) / 2)
    assert row["asn_unique_ips"] == 1
    assert row["label"] == 1


@pytest.mark.parametrize("raw", ["{}", "", None, "not a dict", "[1, 2]"])
def test_parse_shodan_info_handles_empty_and_bad_input(raw):
    assert parse_shodan_info(raw) == {}


def test_shodan_features_count_ports_and_shared_fingerprints():
    feats = build_shodan_features(make_shodan([0, 1, 2, 3])).set_index("attacker_ip_enum")
    assert feats.loc[1, "n_tcp_ports"] == 1 and feats.loc[1, "n_udp_ports"] == 1
    assert feats.loc[0, "jarm_shared"] == 1  # IP 1 shares J1
    assert feats.loc[0, "ja3s_shared"] == 0  # nobody else has A1
    assert feats.loc[2, "no_print_with_ports"] == 1
    assert feats.loc[3, "has_shodan"] == 0


def test_validate_features_rejects_nan_and_duplicates():
    good = pd.DataFrame([[i, *[1.0] * len(FEATURES), 0] for i in range(3)],
                        columns=["attacker_ip_enum", *FEATURES, "label"])
    validate_features(good)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_features(pd.concat([good, good.iloc[[0]]]))
    bad = good.copy()
    bad.loc[0, FEATURES[0]] = float("nan")
    with pytest.raises(DataValidationError, match="NaN"):
        validate_features(bad)


def test_split_has_no_ip_overlap_and_keeps_both_classes():
    feats = pd.DataFrame({"attacker_ip_enum": range(100), "label": [1] * 20 + [0] * 80})
    train, test = split_by_ip(feats)
    assert not set(train["attacker_ip_enum"]) & set(test["attacker_ip_enum"])
    assert len(test) == 20 and test["label"].sum() == 4  # stratified


def test_prepare_writes_train_and_test(raw_dir, tmp_path):
    out = tmp_path / "processed"
    report = prepare(raw_dir, out)
    train = pd.read_parquet(out / "train.parquet")
    test = pd.read_parquet(out / "test.parquet")
    assert len(train) + len(test) == report["n_ips"] == 40
    assert list(train.columns) == ["attacker_ip_enum", *FEATURES, "label"]
    assert (out / "prepare_report.json").exists()
