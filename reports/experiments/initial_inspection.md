# Phase 1: Initial inspection

Inspection date: 2026-11-01. Nothing in `src/`, `models/`, `docs/` or `data/raw/` was changed. Raw numbers are in [raw_data_inspection.json](raw_data_inspection.json), produced by `python -m experiments.inspect_raw_data`.

## Verdict

The dataset is correct and the pipeline can run on it. Four things need care:

1. Running `src.train` or `src.explain` directly would overwrite committed files, so their outputs must be redirected.
2. `hgb_slow` does not train 600 trees as documented. Early stopping stops it at 94.
3. Ludwig cannot run in the current environment.
4. Some features are computed across all IPs, including the test set. This is examined in the data preparation report.

## Environment

| Item | Found | Note |
|---|---|---|
| OS and CPU | Windows 11, i5-8250U, 4 cores / 8 threads | |
| RAM | 19.9 GB total, 7.4 GB free when checked | Data prep is capped at about 6 GB by DuckDB and can spill to disk |
| Disk | 38.8 GB free | Enough |
| Python | 3.12.2 in `.venv` | `pyproject.toml` asks for `>=3.11,<3.12`. The project ran on 3.11 originally, so exact numbers could in principle differ |
| Packages | All 14 pins in `requirements.txt` match | `matplotlib` 3.11.2 is installed but not pinned |
| `make`, `uv` | Not installed | Makefile targets also use Unix paths (`.venv/bin/python`), so we use `python -m ...` directly |
| Tests and lint | 43 passed, ruff clean | Run in `.venv` before any experiment code was added |
| Git remote | GitHub (`uzair-khiiba/VPN-Classifier-MLOps`) | |

## Configured paths (`src/config.py`)

| Setting | Default | Environment override |
|---|---|---|
| Raw data | `data/raw/` | `RAW_DIR` |
| Sample data | `data/sample/` | none |
| Processed data | `data/processed/` | `PROCESSED_DIR` |
| Models | `models/` | `MODELS_DIR` |
| Served model | `models/model.joblib` | `MODEL_PATH` |
| MLflow tracking | `sqlite:///mlflow.db` | `MLFLOW_TRACKING_URI` |
| MLflow experiment | `vpn-proxy-classification` | `MLFLOW_EXPERIMENT` |
| Seed, test share | 42, 0.2 | none |

## The pipeline as the code implements it

**`src/prepare_data.py`** reads `data/raw/train.parq` and `shodan_df_hashed.csv`, and writes `train.parquet`, `test.parquet` and `prepare_report.json` to `data/processed/`. The split is a stratified 80/20 split per IP with seed 42, which matches the documentation.

**`src/features.py`** builds 28 features: 21 from signals and 7 from Shodan.

**`src/train.py`**:

| Question | What the code does |
|---|---|
| Models | `logreg`, `rf`, `rf_shallow`, `hgb`, `hgb_slow` |
| Cross-validation | Stratified 5-fold, shuffled, seed 42, with out-of-fold probabilities pooled across folds |
| Threshold | The value that maximises F1 on the pooled out-of-fold probabilities (`best_f1_threshold`) |
| Imbalance handling | Class weights only (`balanced` or `balanced_subsample`). No resampling |
| Selection | Highest `cv_f1` |
| Metrics | F1, precision, recall, PR-AUC, ROC-AUC and accuracy on test. `f1_score` defaults to `pos_label=1`, so F1 is for the VPN/proxy class |
| Not recorded | Confusion matrix, training time and model size. The experiment wrapper adds these |
| MLflow | One run per candidate, with parameters and metrics only. No model artifact is logged |
| Writes | `models/vpn-clf_<name>_<time>.joblib`, `models/model.joblib`, `models/model_meta.json` |

**`src/explain.py`**'s `main()` always writes `docs/feature_importance.png`, because that path is hard-coded. Experiments import its functions instead of running `main()`.

## Raw data checks

| Check | Result |
|---|---|
| Files present | `train.parq`, `shodan_df_hashed.csv`, plus `test.parq`, `train.csv`, `test.csv`, `samples_submissions.csv` (the pipeline does not use these four) |
| Signals | 61,629,685 rows, as documented |
| Unique IPs | 148,259, as documented |
| Positive IPs | 2,890 (1.9493%), as documented |
| Positive share of signals | 1.68%, so VPN/proxy IPs send slightly fewer signals than their IP share |
| Conflicting labels per IP | 0 |
| Time range | 2023-07-30 00:00:20 to 2023-08-29 12:56:09 UTC |
| Signals per IP | Minimum 1, median 37, mean 416, maximum 1,466,169, so very skewed |
| Required columns | All present |
| Attack types | The 8 types used as features all exist |

**Missing values**

| Column | Nulls | Used by features? |
|---|---|---|
| `watcher_country` | 349,386 | Yes, `n_watcher_countries` ignores nulls and can be 0 |
| `watcher_as_name` | 349,405 | No |
| `attacker_country` | 73,661 | No |
| `attacker_as_name` | 49,512 | No |
| `attacker_as_num` | 462 | Yes, a null ASN does not join, so `asn_unique_ips` falls back to 0 for an IP with no known ASN |
| All other columns | 0 | |

**Shodan file**

| Check | Result |
|---|---|
| Rows and unique IPs | 197,679 of each |
| Columns | `shodan_info` (text), `attacker_ip_enum` (integer), as expected |
| Empty `shodan_info` | 150,083 rows, all the literal text `{}`. No cell failed to parse |
| Signal IPs that have a row | All 148,259, both classes alike, so **having a row does not reveal the label** |
| Rows for IPs not in `train.parq` | 49,420, which are exactly the IPs in Kaggle's unlabelled `test.parq` |

`has_shodan` is defined by having at least one port, not by having a row.

## Differences from the documentation

| Documentation says | What was found | Consequence |
|---|---|---|
| `hgb_slow` uses 600 iterations | `early_stopping='auto'` is active above 10,000 rows. The committed model has `n_iter_ = 94` | `max_iter=600` is only a ceiling. Early stopping also holds back 10% of each training fold |
| Progress image: 39 tests and a CI workflow | 43 tests and no workflow file | The image is outdated |
| `n_attacker_as` etc. | 70 IPs appear in more than one ASN | Harmless, `asn_unique_ips` uses the largest ASN |

## Ludwig feasibility

Ludwig, Ray and PyTorch are not installed. Ludwig 0.11.0 needs Python 3.11, and only 3.12 is on this machine. See Experiment F in the final report for the outcome.

## How the rules are protected

| Risk | Protection |
|---|---|
| Overwriting the model | Baseline runs with `MODELS_DIR` and `MODEL_PATH` pointing at `reports/experiments/artifacts/` |
| Writing to the original experiment | `MLFLOW_EXPERIMENT` is overridden, and experiment code only uses `zainab_experiments` |
| Using synthetic data | `experiments/common.py` refuses any processed table under 100,000 IPs |
| Changing the split | A hash of the test IP list is stored with results |
| Test set influencing choices | Configurations are selected on out-of-fold predictions first, and test scoring happens afterwards |

Protected files were hashed before any work, to verify afterwards:

```
f253f1394aacd0b4bd872231fd1b4653208772887e331389417cbb069d97810b  models/model.joblib
182893eb229b71c0758f06bdb9fe5e96e5c2d19c5b3f8b103d3aef8d8577ffc2  models/model_meta.json
2e1a82ebe300d7acb1b5981e9cf6c84b15c7b800709de1927cef1a6eb90ce190  models/feature_importance.json
ae3a3c8fdbcf1113bd65f4a8fe6c5d821916c798a31ab12d20c381a557454659  docs/feature_importance.png
```
