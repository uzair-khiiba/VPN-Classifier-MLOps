"""Experiment F, step 1: predict the test set with Ludwig's saved best model.

Runs in .venv-ludwig. It repeats the prediction src.automl_ludwig already made, so the results can
join the paired comparison. It tunes nothing. Step 2 is experiments.exp_f_collect, run in .venv.

Usage: .venv-ludwig/Scripts/python -m experiments.exp_f_score_ludwig
"""

import numpy as np
import pandas as pd
from ludwig.api import LudwigModel

from src.automl_ludwig import OUTPUT_DIR, positive_proba
from src.config import PROCESSED_DIR, ROOT, TARGET
from src.features import FEATURES


def main() -> None:
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")[[*FEATURES, TARGET]]
    model = LudwigModel.load(str(OUTPUT_DIR / "best_model"), backend="local", logging_level=30)
    proba = positive_proba(model.predict(dataset=test, output_directory=str(OUTPUT_DIR / "scoring"))[0])
    path = ROOT / "reports" / "experiments" / "artifacts" / "test_proba_ludwig.npy"
    np.save(path, proba)
    print(f"saved {path}: {len(proba)} predictions")


if __name__ == "__main__":
    main()
