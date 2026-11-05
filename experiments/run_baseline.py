"""Phase 3: reproduce the baseline by calling the UNMODIFIED src.train.main().

Environment variables are set before importing src.*, so src.train writes its model and
metadata to reports/experiments/artifacts/baseline/ and its own MLflow runs to a separate
local database. models/model.joblib and the DagsHub baseline experiment are not touched.
`run_candidate` is wrapped from the outside, only to record time, size and confusion matrix.

Usage:
    python -m experiments.run_baseline
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "experiments" / "artifacts" / "baseline"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["MODELS_DIR"] = str(OUT)
os.environ["MODEL_PATH"] = str(OUT / "model.joblib")
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{OUT / 'baseline_repro_mlflow.db'}"
os.environ["MLFLOW_EXPERIMENT"] = "zainab_baseline_repro_local"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import src.config as config  # noqa: E402
import src.train as train_module  # noqa: E402
from experiments.common import (  # noqa: E402
    RESULTS_DIR,
    counts,
    environment,
    load_split,
    model_size_kb,
    save_records,
    split_fingerprint,
)
from src.config import TARGET  # noqa: E402
from src.features import FEATURES  # noqa: E402

assert config.MODEL_PATH == OUT / "model.joblib", "refusing to run: the served model path was not redirected"


def main() -> None:
    train, test = load_split()
    y_test = test[TARGET].to_numpy()
    records: list[dict] = []
    original = train_module.run_candidate

    def timed(name, *args, **kwargs):
        started = time.perf_counter()
        result = original(name, *args, **kwargs)
        seconds = time.perf_counter() - started
        model = result["model"]
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        started = time.perf_counter()
        proba = model.predict_proba(test[FEATURES])[:, 1]
        predict_ms_per_ip = (time.perf_counter() - started) / len(test) * 1000
        m = result["metrics"]
        records.append({
            "run_name": f"baseline_repro_{name}",
            "experiment": "baseline", "model": name, "change": "unmodified src.train candidate",
            "params": {k: v for k, v in est.get_params().items() if isinstance(v, (int, float, str, bool, type(None)))},
            "metrics": {**m, **{f"test_{k}": v for k, v in counts(y_test, proba, result["threshold"]).items()},
                        "cv_plus_fit_seconds": seconds, "model_size_kb": model_size_kb(model),
                        "predict_ms_per_ip": predict_ms_per_ip,
                        "n_iter_trained": float(np.max(est.n_iter_)) if hasattr(est, "n_iter_") else None},
        })
        print(f"    {name}: {seconds:.0f}s total", flush=True)
        return result

    train_module.run_candidate = timed
    sys.argv = ["src.train"]
    train_module.main()
    save_records("baseline", records)
    meta = {"split_fingerprint": split_fingerprint(train, test), "environment": environment(),
            "n_features": len(FEATURES), "models_dir": str(OUT)}
    (RESULTS_DIR / "baseline_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {RESULTS_DIR / 'baseline.json'}")


if __name__ == "__main__":
    main()
