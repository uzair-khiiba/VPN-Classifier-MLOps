"""Phase 2: independent checks of the prepared tables, plus leakage screening.

Usage:
    python -m experiments.verify_prepared_data
"""

import json

import numpy as np
import pandas as pd

from experiments.common import REPORT_DIR, load_split, split_fingerprint
from src.config import ID_COLUMN, RAW_DIR, SHODAN_FILE, SIGNALS_FILE, TARGET
from src.features import FEATURES, SHODAN_FEATURES, build_shodan_features, connect, parse_shodan_info


def main() -> None:
    train, test = load_split()
    both = pd.concat([train, test], ignore_index=True)
    report: dict = {"split_fingerprint": split_fingerprint(train, test)}

    report["rows"] = {"train": len(train), "test": len(test), "total": len(both)}
    report["duplicate_ips"] = int(both[ID_COLUMN].duplicated().sum())
    report["ip_overlap"] = len(set(train[ID_COLUMN]) & set(test[ID_COLUMN]))
    report["class_distribution"] = {
        name: {"positives": int(df[TARGET].sum()), "negatives": int((df[TARGET] == 0).sum()),
               "positive_rate": float(df[TARGET].mean())}
        for name, df in (("train", train), ("test", test), ("total", both))}
    expected = {ID_COLUMN, TARGET, *FEATURES}
    report["unexpected_columns"] = sorted(set(both.columns) - expected)
    report["missing_columns"] = sorted(expected - set(both.columns))
    report["n_features"] = len(FEATURES)
    report["features_with_nulls"] = both[FEATURES].columns[both[FEATURES].isna().any()].tolist()
    report["features_with_negative"] = both[FEATURES].columns[(both[FEATURES] < 0).any()].tolist()
    report["constant_features"] = [f for f in FEATURES if both[f].nunique() <= 1]
    report["id_or_label_in_features"] = sorted({ID_COLUMN, TARGET} & set(FEATURES))
    report["n_signals_sum"] = int(both["n_signals"].sum())

    # Screening on TRAIN ONLY: how well does each single feature rank the label by itself?
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = train[TARGET].to_numpy()
    screen = []
    for f in FEATURES:
        auc = roc_auc_score(y, train[f])
        screen.append({"feature": f, "single_feature_auc": round(max(auc, 1 - auc), 4),
                       "single_feature_pr_auc": round(average_precision_score(y, train[f]), 4),
                       "corr_with_label": round(float(np.corrcoef(train[f], y)[0, 1]), 4),
                       "median_vpn": float(train.loc[y == 1, f].median()),
                       "median_other": float(train.loc[y == 0, f].median())})
    screen.sort(key=lambda r: -r["single_feature_auc"])
    report["single_feature_screen"] = screen

    report["shodan_by_class_train"] = {
        f: {"positive_rate_nonzero_vpn": round(float((train.loc[y == 1, f] > 0).mean()), 4),
            "positive_rate_nonzero_other": round(float((train.loc[y == 0, f] > 0).mean()), 4)}
        for f in SHODAN_FEATURES}
    has = train["has_shodan"] == 1
    report["label_rate_by_has_shodan_train"] = {"has_shodan=1": round(float(y[has].mean()), 4),
                                                "has_shodan=0": round(float(y[~has].mean()), 4),
                                                "share_of_ips_with_shodan": round(float(has.mean()), 4)}

    # Do the "crowding" features use test IPs? Recompute both from TRAIN IPs only and compare on train rows.
    shodan = pd.read_csv(RAW_DIR / SHODAN_FILE)
    parsed = shodan["shodan_info"]
    report["shodan_cell_values"] = {"literal_empty_dict": int((parsed.str.strip() == "{}").sum()),
                                    "parses_to_empty": int((parsed.map(parse_shodan_info).map(len) == 0).sum())}
    train_only = build_shodan_features(shodan[shodan[ID_COLUMN].isin(set(train[ID_COLUMN]))])
    merged = train[[ID_COLUMN, "jarm_shared", "ja3s_shared", "headers_hash_shared"]].merge(
        train_only, on=ID_COLUMN, suffixes=("_all", "_trainonly"))
    report["shared_counts_change_when_test_ips_removed"] = {
        f: {"train_rows_changed": round(float((merged[f + "_all"] != merged[f + "_trainonly"]).mean()), 4),
            "median_ratio_when_nonzero": round(float(
                (merged[f + "_trainonly"] / merged[f + "_all"].where(merged[f + "_all"] > 0)).median()), 4)}
        for f in ("jarm_shared", "ja3s_shared", "headers_hash_shared")}

    con = connect()
    con.register("train_ips", train[[ID_COLUMN]])
    asn = con.execute(f"""
        WITH s AS (SELECT DISTINCT {ID_COLUMN}, attacker_as_num FROM read_parquet('{RAW_DIR / SIGNALS_FILE}')
                   WHERE {ID_COLUMN} IN (SELECT {ID_COLUMN} FROM train_ips)),
             c AS (SELECT attacker_as_num, count(DISTINCT {ID_COLUMN}) AS n FROM s GROUP BY 1)
        SELECT {ID_COLUMN}, max(c.n) AS asn_trainonly FROM s JOIN c USING (attacker_as_num) GROUP BY 1
    """).df()
    con.close()
    m = train[[ID_COLUMN, "asn_unique_ips"]].merge(asn, on=ID_COLUMN, how="left").fillna(0)
    report["asn_unique_ips_when_test_ips_removed"] = {
        "train_rows_changed": round(float((m["asn_unique_ips"] != m["asn_trainonly"]).mean()), 4),
        "median_ratio_when_nonzero": round(float(
            (m["asn_trainonly"] / m["asn_unique_ips"].where(m["asn_unique_ips"] > 0)).median()), 4)}

    (REPORT_DIR / "prepared_data_checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "single_feature_screen"}, indent=1))
    print(pd.DataFrame(screen).to_string(index=False))


if __name__ == "__main__":
    main()
