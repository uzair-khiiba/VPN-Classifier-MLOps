"""Check a downloaded Ludwig model folder: its test predictions must match the saved ones. Runs in .venv-ludwig.

Usage: python -m experiments.check_ludwig_artifact <folder>     (prints one JSON line last)
"""

import json
import sys
import tempfile

import numpy as np
import pandas as pd
from ludwig.api import LudwigModel

from src.automl_ludwig import positive_proba
from src.config import PROCESSED_DIR, ROOT, TARGET
from src.features import FEATURES

TOLERANCE = 1e-5  # Ludwig returns 32-bit floats


def main() -> None:
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")[[*FEATURES, TARGET]]
    saved = np.load(ROOT / "reports" / "experiments" / "artifacts" / "test_proba_ludwig.npy")
    model = LudwigModel.load(sys.argv[1], backend="local", logging_level=30)
    with tempfile.TemporaryDirectory() as tmp:
        proba = positive_proba(model.predict(dataset=test, output_directory=tmp)[0])
    difference = float(np.abs(proba - saved).max())
    print(json.dumps({"rows_compared": len(test), "max_abs_difference": difference,
                      "predictions_equal": difference <= TOLERANCE, "loaded_with": "LudwigModel.load"}))


if __name__ == "__main__":
    main()
