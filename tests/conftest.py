"""Shared fixtures: tiny hand-made datasets so tests never need the full Kaggle data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features import FEATURES


def make_signals(n_ips: int = 40, seed: int = 0) -> pd.DataFrame:
    """Synthetic raw signals with the same columns as the Kaggle train.parq."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2023-09-01")
    attack_types = ["http:scan", "ssh:bruteforce", "tcp:scan", "http:exploit"]
    for ip in range(n_ips):
        label = int(ip % 4 == 0)  # 25% positive
        for k in range(int(rng.integers(2, 12))):
            rows.append({
                "attacker_ip_enum": ip,
                "attack_time": start + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 10))),
                "attack_type": attack_types[(ip + k) % len(attack_types)],
                "attacker_as_num": float(100 + ip % 5),
                "attacker_as_name": f"AS{100 + ip % 5}",
                "attacker_country": "US",
                "watcher_as_num": float(rng.integers(1, 6)),
                "watcher_as_name": "WATCHER",
                "watcher_country": ["FR", "DE", "US"][k % 3],
                "watcher_uuid_enum": int(rng.integers(0, 8)),
                "label": label,
            })
    return pd.DataFrame(rows)


def make_shodan(ips) -> pd.DataFrame:
    infos = {
        0: "{'443/tcp': {'headers_hash': 11, 'jarm': 'J1', 'ja3s': 'A1'}}",
        1: "{'443/tcp': {'headers_hash': 11, 'jarm': 'J1', 'ja3s': None}, "
           "'53/udp': {'headers_hash': None, 'jarm': None, 'ja3s': None}}",
        2: "{'22/tcp': {'headers_hash': None, 'jarm': None, 'ja3s': None}}",
    }
    return pd.DataFrame({"shodan_info": [infos.get(ip, "{}") for ip in ips], "attacker_ip_enum": list(ips)})


@pytest.fixture
def raw_dir(tmp_path):
    """A raw-data directory laid out like data/raw/."""
    signals = make_signals()
    signals.to_parquet(tmp_path / "train.parq", index=False)
    make_shodan(sorted(signals["attacker_ip_enum"].unique())).to_csv(tmp_path / "shodan_df_hashed.csv", index=False)
    return tmp_path


@pytest.fixture
def model_path(tmp_path):
    """A small but valid model bundle trained on random features."""
    from src.model_io import save_bundle

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((200, len(FEATURES))) * 10, columns=FEATURES)
    y = (X["asn_unique_ips"] + rng.normal(0, 1, 200) > 5).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    bundle = {
        "model": model,
        "features": FEATURES,
        "threshold": 0.5,
        "reference": X.median().to_dict(),
        "metadata": {"model_name": "test-logreg", "version": "test"},
    }
    return save_bundle(bundle, tmp_path / "model.joblib")
