"""Phase 5: explain the committed model with the project's own methods.

Uses functions from src.explain and the running API's /explain endpoint. src.explain.main() is
not called, because it would overwrite models/feature_importance.json and docs/feature_importance.png.
No SHAP is used. Everything here reads the model without changing it.

Usage: python -m experiments.exp_explain
"""

import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.metrics import f1_score

from experiments.common import RESULTS_DIR, load_split
from src.app import app
from src.config import MODEL_PATH, MODELS_DIR, RANDOM_STATE, TARGET
from src.explain import permutation_f1_importance
from src.features import ATTACK_TYPES, SHODAN_FEATURES, share_name
from src.model_io import load_bundle

GROUPS = {
    "All Shodan features": SHODAN_FEATURES,
    "Attack mix (shares)": [share_name(t) for t in ATTACK_TYPES],
    "Timing": ["active_hours", "n_active_days", "signals_per_day", "max_signals_per_hour", "mean_gap_s",
               "median_gap_s"],
    "Breadth": ["n_signals", "n_attack_types", "n_watchers", "n_watcher_as", "n_watcher_countries", "n_attacker_as"],
    "ASN crowding": ["asn_unique_ips"],
}
WHAT_IF = {
    "has_shodan": [0, 1],
    "n_tcp_ports": [0, 1, 2, 5, 20],
    "headers_hash_shared": [0, 10, 100, 690, 5000],
    "asn_unique_ips": [1, 50, 154, 1000, 5000],
    "n_active_days": [1, 3, 9, 20, 30],
}


def group_importance(bundle: dict, test: pd.DataFrame, y: np.ndarray, n_repeats: int = 5) -> list[dict]:
    """Shuffle a whole group together: correlated features cannot cover for each other."""
    X = test[bundle["features"]]
    threshold = bundle["threshold"]
    base = f1_score(y, bundle["model"].predict_proba(X)[:, 1] >= threshold)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for group, cols in GROUPS.items():
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[cols] = X[cols].to_numpy()[order]
            drops.append(base - f1_score(y, bundle["model"].predict_proba(shuffled)[:, 1] >= threshold))
        rows.append({"group": group, "n_features": len(cols), "f1_drop": float(np.mean(drops)),
                     "f1_drop_std": float(np.std(drops))})
    return sorted(rows, key=lambda r: -r["f1_drop"])


def main() -> None:
    _, test = load_split()
    bundle = load_bundle(MODEL_PATH)
    features, threshold = bundle["features"], bundle["threshold"]
    y = test[TARGET].to_numpy()
    proba = bundle["model"].predict_proba(test[features])[:, 1]
    pred = proba >= threshold

    importance = permutation_f1_importance(bundle, test, y, n_repeats=5)
    documented = json.loads((MODELS_DIR / "feature_importance.json").read_text())["importance"]
    documented = {r["feature"]: r["f1_drop"] for r in documented}
    importance["documented_f1_drop"] = importance["feature"].map(documented)
    print(importance.head(10).to_string(index=False), flush=True)

    groups = group_importance(bundle, test, y)
    print(pd.DataFrame(groups).to_string(index=False), flush=True)

    outcome = np.select([pred & (y == 1), pred & (y == 0), ~pred & (y == 1)], ["TP", "FP", "FN"], "TN")
    profile = {}
    for name in ("TP", "FP", "FN", "TN"):
        part = test[outcome == name]
        profile[name] = {"n": len(part), "has_shodan_rate": float(part["has_shodan"].mean()),
                         **{f"median_{f}": float(part[f].median())
                            for f in ("n_tcp_ports", "headers_hash_shared", "asn_unique_ips", "n_active_days")}}
    print(json.dumps(profile, indent=1), flush=True)

    # Per-IP explanations from the real API endpoint: the most confident case of each outcome.
    cases = {}
    with TestClient(app) as client:
        for name, pick in (("TP", np.argmax), ("FP", np.argmax), ("FN", np.argmin), ("TN", np.argmin)):
            positions = np.flatnonzero(outcome == name)
            position = positions[pick(proba[positions])]
            body = test.iloc[position][features].astype(float).to_dict()
            response = client.post("/explain?top_k=5", json=body)
            response.raise_for_status()
            cases[name] = {"true_label": int(y[position]), **response.json()}
            top = [(c["feature"], c["value"], c["effect"]) for c in cases[name]["contributions"][:3]]
            print(f"{name}: p={cases[name]['probability']:.4f} top={top}", flush=True)

        # One feature at a time, on the median true positive.
        positions = np.flatnonzero(outcome == "TP")
        anchor = positions[np.argsort(proba[positions])[len(positions) // 2]]
        row = test.iloc[anchor][features].astype(float).to_dict()
        what_if = {"anchor_probability": float(proba[anchor]), "anchor_values": {f: row[f] for f in WHAT_IF},
                   "sweeps": {}}
        for feature, values in WHAT_IF.items():
            sweep = []
            for value in values:
                result = client.post("/predict", json={**row, feature: float(value)})
                result.raise_for_status()
                sweep.append({"value": value, "probability": result.json()["probability"],
                              "is_vpn_proxy": result.json()["is_vpn_proxy"]})
            what_if["sweeps"][feature] = sweep
            print(feature, [(s["value"], s["probability"]) for s in sweep], flush=True)

    out = {"model": bundle["metadata"]["model_name"], "version": bundle["metadata"]["version"],
           "threshold": threshold, "permutation": importance.round(4).to_dict(orient="records"), "groups": groups,
           "outcome_profile": profile, "cases": cases, "what_if": what_if}
    (RESULTS_DIR / "explainability.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"saved {RESULTS_DIR / 'explainability.json'}", flush=True)


if __name__ == "__main__":
    main()
