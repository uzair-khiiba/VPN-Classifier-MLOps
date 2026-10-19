"""Central paths and settings shared by the pipeline, the API and the tests."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = Path(os.getenv("RAW_DIR", ROOT / "data" / "raw"))
SAMPLE_DIR = ROOT / "data" / "sample"
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", ROOT / "data" / "processed"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))

SIGNALS_FILE = "train.parq"
SHODAN_FILE = "shodan_df_hashed.csv"

# The model file the API loads; overridable so tests can point at a temporary model.
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.joblib"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "vpn-proxy-classification"

RANDOM_STATE = 42
TEST_SIZE = 0.2

ID_COLUMN = "attacker_ip_enum"
TARGET = "label"
