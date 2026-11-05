"""Phase 1: read-only inspection of the real raw files in data/raw/.

Writes nothing except reports/experiments/raw_data_inspection.json.

Usage:
    python -m experiments.inspect_raw_data
"""

import json
import time

import pandas as pd

from src.config import ID_COLUMN, RAW_DIR, ROOT, SHODAN_FILE, SIGNALS_FILE, TARGET
from src.features import ATTACK_TYPES, connect, parse_shodan_info
from src.prepare_data import REQUIRED_SIGNAL_COLUMNS

OUT = ROOT / "reports" / "experiments" / "raw_data_inspection.json"


def main() -> None:
    started = time.time()
    signals = RAW_DIR / SIGNALS_FILE
    src = f"read_parquet('{signals}')"
    con = connect()
    report: dict = {"signals_file": str(signals), "size_bytes": signals.stat().st_size}

    schema = con.execute(f"DESCRIBE SELECT * FROM {src}").df()
    report["columns"] = dict(zip(schema["column_name"], schema["column_type"], strict=True))
    report["missing_required"] = sorted(REQUIRED_SIGNAL_COLUMNS - set(schema["column_name"]))
    print(schema[["column_name", "column_type"]].to_string(index=False), flush=True)

    columns = list(schema["column_name"])
    nulls = con.execute(
        "SELECT count(*) AS n_rows, " + ", ".join(f'count(*) - count("{c}") AS "{c}"' for c in columns) + f" FROM {src}"
    ).df().iloc[0].to_dict()
    report["n_rows"] = int(nulls.pop("n_rows"))
    report["null_counts"] = {k: int(v) for k, v in nulls.items()}
    print(f"\nrows: {report['n_rows']:,}\nnulls: {report['null_counts']}", flush=True)

    per_ip = con.execute(
        f"SELECT {ID_COLUMN} AS ip, count(*) AS n, count(DISTINCT {TARGET}) AS n_labels, max({TARGET})::INT AS y "
        f"FROM {src} GROUP BY 1"
    ).df()
    report["n_ips"] = len(per_ip)
    report["n_positive_ips"] = int(per_ip["y"].sum())
    report["positive_rate_ips"] = float(per_ip["y"].mean())
    report["ips_with_conflicting_labels"] = int((per_ip["n_labels"] > 1).sum())
    report["signals_per_ip"] = {k: float(v) for k, v in per_ip["n"].describe().items()}
    report["positive_rate_signals"] = float(per_ip.loc[per_ip["y"] == 1, "n"].sum() / per_ip["n"].sum())
    print(f"IPs: {len(per_ip):,}, positive: {report['n_positive_ips']:,} ({report['positive_rate_ips']:.4%}), "
          f"conflicting: {report['ips_with_conflicting_labels']}", flush=True)
    print(f"share of SIGNALS from positive IPs: {report['positive_rate_signals']:.4%}", flush=True)
    print(per_ip["n"].describe().to_string(), flush=True)

    t_min, t_max = con.execute(
        f"SELECT min(attack_time)::VARCHAR, max(attack_time)::VARCHAR FROM {src}"
    ).fetchone()
    report["time_range"] = [t_min, t_max]
    print(f"time range: {t_min} .. {t_max}", flush=True)

    types = con.execute(
        f"SELECT attack_type, count(*) AS n_signals, count(DISTINCT {ID_COLUMN}) AS n_ips FROM {src} "
        "GROUP BY 1 ORDER BY n_ips DESC"
    ).df()
    report["n_attack_types"] = len(types)
    report["attack_types_top"] = types.head(15).to_dict(orient="records")
    report["expected_attack_types_missing"] = sorted(set(ATTACK_TYPES) - set(types["attack_type"]))
    print(types.head(15).to_string(index=False), flush=True)

    multi_asn = con.execute(
        f"SELECT count(*) FROM (SELECT {ID_COLUMN} FROM {src} GROUP BY 1 HAVING count(DISTINCT attacker_as_num) > 1)"
    ).fetchone()[0]
    report["ips_with_multiple_asns"] = int(multi_asn)
    print(f"IPs seen in more than one attacker ASN: {multi_asn:,}", flush=True)
    con.close()

    shodan = pd.read_csv(RAW_DIR / SHODAN_FILE)
    parsed = shodan["shodan_info"].map(parse_shodan_info)
    ids = set(shodan[ID_COLUMN])
    train_ids = set(per_ip["ip"])
    positive_ids = set(per_ip.loc[per_ip["y"] == 1, "ip"])
    report["shodan"] = {
        "columns": {c: str(t) for c, t in shodan.dtypes.items()},
        "n_rows": len(shodan),
        "n_unique_ips": len(ids),
        "null_counts": {c: int(v) for c, v in shodan.isna().sum().items()},
        "n_unparseable_or_empty": int((parsed.map(len) == 0).sum()),
        "ips_in_signals": len(ids & train_ids),
        "ips_not_in_signals": len(ids - train_ids),
        "signal_ips_with_shodan_row": len(train_ids & ids),
        "positive_ips_with_shodan_row": len(positive_ids & ids),
        "negative_ips_with_shodan_row": len((train_ids - positive_ids) & ids),
    }
    print(json.dumps(report["shodan"], indent=2), flush=True)
    print("example shodan_info:", str(shodan["shodan_info"].iloc[0])[:300], flush=True)

    report["elapsed_s"] = round(time.time() - started, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT} in {report['elapsed_s']} s")


if __name__ == "__main__":
    main()
