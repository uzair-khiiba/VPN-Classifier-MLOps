"""Per-IP feature engineering.

The raw data has one row per attack signal (~61M rows), but the label belongs to the
attacker IP. Every IP is therefore collapsed into a single row of behavioural
features. DuckDB does the aggregation so the full file never has to fit in memory.

Feature ideas follow the 2nd-place Kaggle solution (crowdsec-vpn-classification-2nd-place.ipynb):
attack volume and timing, how many watchers/ASNs an IP touches, how crowded its ASN
is, and how many other IPs share its Shodan TLS/HTTP fingerprints.
"""

import ast
import os
import re
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

from src.config import ID_COLUMN, TARGET

# Attack types that get their own "share of this IP's signals" feature: every type
# seen from more than ~2,000 training IPs. Rarer types are left out to keep the API input small.
ATTACK_TYPES = [
    "http:spam",
    "http:scan",
    "http:exploit",
    "http:bruteforce",
    "http:crawl",
    "ssh:bruteforce",
    "pop3/imap:bruteforce",
    "tcp:scan",
]


def share_name(attack_type: str) -> str:
    return "share_" + re.sub(r"[^a-z0-9]+", "_", attack_type.lower())


SIGNAL_FEATURES = [
    "n_signals",
    "n_attack_types",
    "n_watchers",
    "n_watcher_as",
    "n_watcher_countries",
    "n_attacker_as",
    "active_hours",
    "n_active_days",
    "signals_per_day",
    "max_signals_per_hour",
    "mean_gap_s",
    "median_gap_s",
    "asn_unique_ips",
] + [share_name(t) for t in ATTACK_TYPES]

SHODAN_FEATURES = [
    "has_shodan",
    "n_tcp_ports",
    "n_udp_ports",
    "jarm_shared",
    "ja3s_shared",
    "headers_hash_shared",
    "no_print_with_ports",
]

FEATURES = SIGNAL_FEATURES + SHODAN_FEATURES


def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB connection pinned to UTC (day/hour buckets must not depend on the machine's
    time zone) and capped in memory so the 61M-row aggregation spills to disk instead."""
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"SET memory_limit = '{os.getenv('DUCKDB_MEMORY_LIMIT', '6GB')}'")
    return con


def _share_columns() -> str:
    return ",\n".join(
        f"avg(CASE WHEN attack_type = '{t}' THEN 1.0 ELSE 0.0 END) AS {share_name(t)}"
        for t in ATTACK_TYPES
    )


def build_signal_features(signals_path: Path, with_label: bool = True) -> pd.DataFrame:
    """Aggregate raw signals into one row of features per attacker IP."""
    label_col = f", max({TARGET})::INT AS {TARGET}" if with_label else ""
    query = f"""
    WITH s AS (
        SELECT * FROM read_parquet('{signals_path}')
    ),
    gaps AS (
        SELECT {ID_COLUMN},
               epoch(attack_time) - epoch(lag(attack_time) OVER (
                   PARTITION BY {ID_COLUMN} ORDER BY attack_time)) AS gap
        FROM s
    ),
    gap_stats AS (
        SELECT {ID_COLUMN}, avg(gap) AS mean_gap_s, median(gap) AS median_gap_s
        FROM gaps GROUP BY {ID_COLUMN}
    ),
    hourly AS (
        SELECT {ID_COLUMN}, max(c) AS max_signals_per_hour
        FROM (SELECT {ID_COLUMN}, count(*) AS c FROM s
              GROUP BY {ID_COLUMN}, date_trunc('hour', attack_time))
        GROUP BY {ID_COLUMN}
    ),
    -- How many distinct IPs attack from each ASN: hosting/VPN ASNs tend to be crowded.
    asn_size AS (
        SELECT attacker_as_num, count(DISTINCT {ID_COLUMN}) AS asn_ips
        FROM s GROUP BY attacker_as_num
    ),
    ip_asn AS (
        SELECT {ID_COLUMN}, max(a.asn_ips) AS asn_unique_ips
        FROM (SELECT DISTINCT {ID_COLUMN}, attacker_as_num FROM s) d
        JOIN asn_size a USING (attacker_as_num)
        GROUP BY {ID_COLUMN}
    ),
    base AS (
        SELECT {ID_COLUMN},
               count(*) AS n_signals,
               count(DISTINCT attack_type) AS n_attack_types,
               count(DISTINCT watcher_uuid_enum) AS n_watchers,
               count(DISTINCT watcher_as_num) AS n_watcher_as,
               count(DISTINCT watcher_country) AS n_watcher_countries,
               count(DISTINCT attacker_as_num) AS n_attacker_as,
               (epoch(max(attack_time)) - epoch(min(attack_time))) / 3600.0 AS active_hours,
               count(DISTINCT date_trunc('day', attack_time)) AS n_active_days,
               {_share_columns()}
               {label_col}
        FROM s GROUP BY {ID_COLUMN}
    )
    SELECT base.*,
           n_signals / n_active_days AS signals_per_day,
           coalesce(h.max_signals_per_hour, 0) AS max_signals_per_hour,
           coalesce(g.mean_gap_s, 0) AS mean_gap_s,
           coalesce(g.median_gap_s, 0) AS median_gap_s,
           coalesce(a.asn_unique_ips, 0) AS asn_unique_ips
    FROM base
    LEFT JOIN hourly h USING ({ID_COLUMN})
    LEFT JOIN gap_stats g USING ({ID_COLUMN})
    LEFT JOIN ip_asn a USING ({ID_COLUMN})
    ORDER BY {ID_COLUMN}
    """
    con = connect()
    try:
        return con.execute(query).df()
    finally:
        con.close()


def parse_shodan_info(raw: str) -> dict:
    """Parse one shodan_info cell, e.g. "{'80/tcp': {'jarm': None, ...}}"."""
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_shodan_features(shodan: pd.DataFrame) -> pd.DataFrame:
    """Port counts and fingerprint sharing per IP from the Shodan metadata file."""
    rows = []
    for ip, raw in zip(shodan[ID_COLUMN], shodan["shodan_info"], strict=True):
        ports = parse_shodan_info(raw)
        prints = {"jarm": set(), "ja3s": set(), "headers_hash": set()}
        for info in ports.values():
            if isinstance(info, dict):
                for kind in prints:
                    value = info.get(kind)
                    if value not in (None, "", 0):
                        prints[kind].add(value)
        rows.append(
            {
                ID_COLUMN: ip,
                "n_tcp_ports": sum(p.endswith("/tcp") for p in ports),
                "n_udp_ports": sum(p.endswith("/udp") for p in ports),
                **{f"_{k}": v for k, v in prints.items()},
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=ID_COLUMN)

    # For each fingerprint type, count how many *other* IPs share this IP's fingerprints.
    for kind in ("jarm", "ja3s", "headers_hash"):
        counts = Counter(fp for fps in df[f"_{kind}"] for fp in fps)
        df[f"{kind}_shared"] = [sum(counts[fp] - 1 for fp in fps) for fps in df[f"_{kind}"]]

    no_prints = df[["_jarm", "_ja3s", "_headers_hash"]].apply(lambda r: not any(r), axis=1)
    has_ports = (df["n_tcp_ports"] + df["n_udp_ports"]) > 0
    df["no_print_with_ports"] = (no_prints & has_ports).astype(int)
    df["has_shodan"] = (df["n_tcp_ports"] + df["n_udp_ports"] > 0).astype(int)
    return df.drop(columns=["_jarm", "_ja3s", "_headers_hash"])


def build_features(signals_path: Path, shodan: pd.DataFrame | None) -> pd.DataFrame:
    """Full per-IP feature table (plus label): signal features joined with Shodan features."""
    feats = build_signal_features(signals_path)
    if shodan is not None:
        feats = feats.merge(build_shodan_features(shodan), on=ID_COLUMN, how="left")
    for col in SHODAN_FEATURES:
        if col not in feats:
            feats[col] = 0
    feats[SHODAN_FEATURES] = feats[SHODAN_FEATURES].fillna(0)
    feats[FEATURES] = feats[FEATURES].astype("float64")
    return feats[[ID_COLUMN, *FEATURES, TARGET]]
