"""Experiment D: feature ablation, Shodan dependence, and the effect of population counts.

Only the feature set changes. Part 2 ("inductive") rebuilds the four population-count features
so each one only counts IPs from the training part of the fold: the validation IPs are treated
as new arrivals, the way a deployed service would see them.

Usage: python -m experiments.exp_d_features
"""

import time
from collections import Counter

import numpy as np
import pandas as pd

from experiments.common import ARTIFACT_DIR, folds, hgb, load_split, save_records
from experiments.runner import make_record, run_cv
from src.config import ID_COLUMN, RAW_DIR, SHODAN_FILE, SIGNALS_FILE, TARGET
from src.features import FEATURES, SHODAN_FEATURES, connect, parse_shodan_info

FINGERPRINTS = {"jarm": "jarm_shared", "ja3s": "ja3s_shared", "headers_hash": "headers_hash_shared"}
POPULATION = [*FINGERPRINTS.values(), "asn_unique_ips"]

ABLATIONS = {
    "all_features": [],
    "no_shodan": SHODAN_FEATURES,
    "no_has_shodan": ["has_shodan"],
    "no_n_tcp_ports": ["n_tcp_ports"],
    "no_headers_hash_shared": ["headers_hash_shared"],
    "no_shodan_ports": ["has_shodan", "n_tcp_ports", "n_udp_ports", "no_print_with_ports"],
    "no_fingerprint_counts": list(FINGERPRINTS.values()),
    "no_asn_unique_ips": ["asn_unique_ips"],
    "no_population_counts": POPULATION,
}


def load_lookups() -> tuple[dict, pd.DataFrame]:
    """Per-IP fingerprint sets from Shodan, and (IP, ASN) pairs from the signals."""
    shodan = pd.read_csv(RAW_DIR / SHODAN_FILE)
    prints = {}
    for ip, raw in zip(shodan[ID_COLUMN], shodan["shodan_info"], strict=True):
        found = {kind: set() for kind in FINGERPRINTS}
        for info in parse_shodan_info(raw).values():
            if isinstance(info, dict):
                for kind in FINGERPRINTS:
                    value = info.get(kind)
                    if value not in (None, "", 0):
                        found[kind].add(value)
        prints[ip] = found
    con = connect()
    pairs = con.execute(f"SELECT DISTINCT {ID_COLUMN}, attacker_as_num FROM read_parquet('{RAW_DIR / SIGNALS_FILE}') "
                        "WHERE attacker_as_num IS NOT NULL").df()
    con.close()
    return prints, pairs


def inductive_counts(fit_ids, other_ids, prints: dict, pairs: pd.DataFrame,
                     aligned: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Population features counted over fit_ids only.

    Default: fit rows count the OTHER fit IPs, as in src/features.py. A new row counts all fit IPs sharing
    its fingerprint. For ASN size, the row itself is counted, also as in src/features.py.

    aligned=True: every IP reads one frozen table, with no self-adjustment, so a fit row and a new
    row with the same fingerprint get the same value.
    """
    tally = {kind: Counter(fp for ip in fit_ids for fp in prints[ip][kind]) for kind in FINGERPRINTS}
    fit_set = set(fit_ids.tolist())
    asn_size = pairs[pairs[ID_COLUMN].isin(fit_set)].groupby("attacker_as_num")[ID_COLUMN].nunique()
    best = pairs.assign(n=pairs["attacker_as_num"].map(asn_size).fillna(0)).groupby(ID_COLUMN)["n"].max()

    def table(ids, is_fit):
        own = 1 if is_fit and not aligned else 0
        rows = {col: [sum(tally[kind][fp] - own for fp in prints[ip][kind]) for ip in ids]
                for kind, col in FINGERPRINTS.items()}
        known = pd.Series(ids).map(best)
        # Unknown ASN stays 0, as in the pipeline. Unless aligned, a new IP adds itself to its ASN's count.
        rows["asn_unique_ips"] = (known if is_fit or aligned else known + 1).fillna(0).to_numpy()
        return pd.DataFrame(rows, index=ids, dtype=float)

    return table(fit_ids, True), table(other_ids, False)


def inductive_frame(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURES].copy()
    for col in POPULATION:
        out[col] = table.loc[df[ID_COLUMN], col].to_numpy()
    return out


def run_inductive(train: pd.DataFrame, prints: dict, pairs: pd.DataFrame, aligned: bool = False) -> dict:
    name = "inductive_aligned" if aligned else "inductive_counts"
    run_name = f"exp_d_{name}"
    change = "population counts from training-fold IPs" + (", one frozen lookup table" if aligned else "")
    path = ARTIFACT_DIR / f"oof_{run_name}.npz"
    y = train[TARGET].to_numpy()
    if path.exists():
        cached = np.load(path)
        oof, n_iters, seconds = cached["oof"], cached["n_iters"].tolist(), cached["fit_seconds"].tolist()
    else:
        print(f"  {run_name}: {change}", flush=True)
        oof, n_iters, seconds = np.zeros(len(y)), [], []
        ids = train[ID_COLUMN].to_numpy()
        for i, (fit, val) in enumerate(folds(y)):
            fit_table, val_table = inductive_counts(ids[fit], ids[val], prints, pairs, aligned)
            model = hgb()
            started = time.perf_counter()
            model.fit(inductive_frame(train.iloc[fit], fit_table), y[fit])
            seconds.append(time.perf_counter() - started)
            oof[val] = model.predict_proba(inductive_frame(train.iloc[val], val_table))[:, 1]
            n_iters.append(int(model.n_iter_))
            print(f"    fold {i + 1}/5: {seconds[-1]:.0f}s", flush=True)
        np.savez(path, oof=oof, n_iters=np.array(n_iters), fit_seconds=np.array(seconds))
    return make_record(run_name, name, "exp_d", {"kind": "inductive", "aligned": aligned}, y, oof, n_iters, seconds,
                       change)


def main() -> None:
    train, _ = load_split()
    records = []
    for name, removed in ABLATIONS.items():
        feats = [f for f in FEATURES if f not in removed]
        spec = {"kind": "hgb", "overrides": {}, "features": feats}
        cache = "hgb_slow_base" if not removed else None
        records.append(run_cv(name, "exp_d", spec, train, f"removed: {removed or 'nothing'}", cache_as=cache))
        records[-1]["removed"] = removed
        save_records("exp_d", records)
    spec = {"kind": "hgb", "overrides": {}, "features": SHODAN_FEATURES}
    records.append(run_cv("only_shodan", "exp_d", spec, train, "only the 7 Shodan features"))
    save_records("exp_d", records)
    prints, pairs = load_lookups()
    records.append(run_inductive(train, prints, pairs))
    save_records("exp_d", records)


if __name__ == "__main__":
    main()
