"""Features for a single IP at request time, and the reference tables that make them match training.

Two of the 28 features compare an IP with every other IP in the data, so they can't be computed
from one IP's signals alone:

    asn_unique_ips   how many distinct attacker IPs attack from the same network (ASN)
    *_shared         how many other IPs share this IP's Shodan JARM / JA3S / header fingerprints

`build_reference` counts these once over the full training data and saves them. For a new IP,
`features_from_signals` runs the same aggregation code as training (src/features.py) on its signals
and Shodan ports, then looks those two kinds of features up in the reference. For an IP that was not
in the data, the result equals what the training pipeline would have produced had the IP been in it.

Build the reference and the test-IP lookup table with:
    python -m src.online_features            # from data/raw/ and data/processed/
    python -m src.online_features --sample   # from the synthetic sample
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import (
    FEATURE_REFERENCE_PATH,
    ID_COLUMN,
    IP_LOOKUP_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    SAMPLE_DIR,
    SHODAN_FILE,
    SIGNALS_FILE,
    TARGET,
)
from src.features import (
    FEATURES,
    FINGERPRINT_KINDS,
    build_signal_features,
    connect,
    parse_shodan_info,
    shodan_flags,
    summarize_shodan_ports,
)

# The raw signal columns the feature code reads. Names and meanings follow the Kaggle train.parq.
SIGNAL_COLUMNS = ["attack_time", "attack_type", "attacker_as_num", "watcher_uuid_enum", "watcher_as_num",
                  "watcher_country"]


def _asn_key(asn) -> str:
    return str(int(asn))


def _fingerprint_key(value) -> str:
    # Case and spaces are ignored. The real fingerprints are all lowercase with no two differing only by case,
    # so this changes no count; it only lets "ABC123" typed by a user match "abc123".
    return str(value).strip().lower()


def build_reference(signals: Path | pd.DataFrame, shodan: pd.DataFrame | None) -> dict:
    """Per-ASN IP counts and per-fingerprint IP counts over the given data."""
    con = connect()
    try:
        if isinstance(signals, pd.DataFrame):
            con.register("signals_input", signals)
            relation = "signals_input"
        else:
            relation = f"read_parquet('{signals}')"
        rows = con.execute(f"""
            SELECT attacker_as_num, count(DISTINCT {ID_COLUMN}) FROM {relation}
            WHERE attacker_as_num IS NOT NULL GROUP BY attacker_as_num
        """).fetchall()
        n_ips = con.execute(f"SELECT count(DISTINCT {ID_COLUMN}) FROM {relation}").fetchone()[0]
    finally:
        con.close()

    counts = {kind: Counter() for kind in FINGERPRINT_KINDS}
    n_shodan_ips = 0
    if shodan is not None:
        shodan = shodan.drop_duplicates(subset=ID_COLUMN)  # as in build_shodan_features
        n_shodan_ips = len(shodan)
        for raw in shodan["shodan_info"]:
            prints = summarize_shodan_ports(parse_shodan_info(raw))["prints"]
            for kind in FINGERPRINT_KINDS:
                counts[kind].update(_fingerprint_key(fp) for fp in prints[kind])
    return {
        "n_ips": int(n_ips),
        "n_shodan_ips": int(n_shodan_ips),
        "asn_ips": {_asn_key(asn): int(n) for asn, n in rows},
        "fingerprints": {kind: dict(counter) for kind, counter in counts.items()},
    }


def features_from_signals(signals: pd.DataFrame, shodan_ports: dict | None, reference: dict) -> dict[str, float]:
    """All 28 model features for ONE new IP, from its raw signals and its Shodan ports.

    `signals` has the SIGNAL_COLUMNS, one row per signal, with attack_time as a UTC timestamp.
    `shodan_ports` has the shape of one shodan_info cell: {"443/tcp": {"jarm": ..., "ja3s": ...,
    "headers_hash": ...}, ...}. Empty or None means Shodan has no record of the IP.
    """
    if signals.empty:
        raise ValueError("at least one signal is needed")
    frame = signals[SIGNAL_COLUMNS].copy()
    frame[ID_COLUMN] = 0
    row = build_signal_features(frame, with_label=False).iloc[0].to_dict()

    # In training the IP's own ASN count includes the IP itself; this IP is not in the reference, so add 1.
    asns = {_asn_key(asn) for asn in frame["attacker_as_num"].dropna()}
    row["asn_unique_ips"] = max((reference["asn_ips"].get(asn, 0) + 1 for asn in asns), default=0)

    summary = summarize_shodan_ports(shodan_ports or {})
    row.update(n_tcp_ports=summary["n_tcp_ports"], n_udp_ports=summary["n_udp_ports"],
               **shodan_flags(summary["n_tcp_ports"], summary["n_udp_ports"], summary["prints"]))
    for kind in FINGERPRINT_KINDS:
        known = reference["fingerprints"][kind]
        # Every IP in the reference is "another IP", so no self-count to remove here.
        row[f"{kind}_shared"] = sum(known.get(_fingerprint_key(fp), 0) for fp in summary["prints"][kind])
    return {name: float(row[name]) for name in FEATURES}


def build_ip_lookup(test: pd.DataFrame) -> pd.DataFrame:
    """The held-out test IPs with their features and true label, for lookups by IP ID."""
    return test[[ID_COLUMN, *FEATURES, TARGET]].sort_values(ID_COLUMN).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", action="store_true", help="use the small sample in data/sample/")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    raw_dir = args.raw_dir or (SAMPLE_DIR if args.sample else RAW_DIR)

    shodan_path = raw_dir / SHODAN_FILE
    shodan = pd.read_csv(shodan_path) if shodan_path.exists() else None
    reference = build_reference(raw_dir / SIGNALS_FILE, shodan)
    FEATURE_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_REFERENCE_PATH.write_text(json.dumps(reference))
    print(f"Reference: {len(reference['asn_ips']):,} ASNs, "
          f"{sum(len(v) for v in reference['fingerprints'].values()):,} fingerprints -> {FEATURE_REFERENCE_PATH}")

    lookup = build_ip_lookup(pd.read_parquet(args.processed_dir / "test.parquet"))
    lookup.to_parquet(IP_LOOKUP_PATH, index=False)
    print(f"Lookup: {len(lookup):,} test IPs -> {IP_LOOKUP_PATH}")


if __name__ == "__main__":
    main()
