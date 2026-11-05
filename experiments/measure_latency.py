"""Clean prediction timing for the baseline and the selected model. Run it with the CPU otherwise idle.

Usage: python -m experiments.measure_latency
"""

import json
import time

import joblib
import numpy as np

from experiments.common import ARTIFACT_DIR, RESULTS_DIR, load_split, model_size_kb
from src.config import MODEL_PATH
from src.features import FEATURES


def timed(function, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples))


def main() -> None:
    _, test = load_split()
    X = test[FEATURES]
    one = X.iloc[[0]]
    models = {"baseline (committed hgb_slow)": joblib.load(MODEL_PATH)["model"],
              "selected (400 trees, no early stopping)":
                  joblib.load(ARTIFACT_DIR / "selected_models" / "exp_c_trees_400_no_early_stop.joblib")}
    out = {}
    for name, model in models.items():
        out[name] = {"trees": int(model.n_iter_), "model_size_kb": model_size_kb(model),
                     "one_ip_ms": timed(lambda m=model: m.predict_proba(one), 50) * 1000,
                     "all_29652_ips_s": timed(lambda m=model: m.predict_proba(X), 5)}
        print(name, out[name], flush=True)
    (RESULTS_DIR / "latency.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
