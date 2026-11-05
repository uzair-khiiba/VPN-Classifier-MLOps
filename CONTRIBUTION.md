# Contributions

**Course:** MLOps, BS Computer Science, Fall 2026
**Project:** VPN/Proxy IP Classification Using CrowdSec Threat Intelligence Data

| Member | ERP ID |
|---|---|
| Zainab Soomro | 28367 |
| Uzair Ahmed Khuhro | 29173 |

The group has two members, which the instructor approved.

## Workflow

- Branches are named `feat/...`, `fix/...` or `infra/...`. Each pull request needs one approval from the other member before it is merged into `main`; we do not push to `main` directly.
- The GitHub history (commits, PRs, reviews, issues) is the record for the table below.

## Milestone 0: Proposal

| Task | Owner |
|---|---|
| Proposal document | Zainab Soomro, Uzair Ahmed Khuhro |
| Proposal slides and presentation | _presenter: TODO_ |

## Areas of ownership

| Area | Owner |
|---|---|
| Preprocessing: data validation, per-IP feature engineering, features for a single IP | Uzair Ahmed Khuhro |
| MLflow: experiment tracking, model training and selection | Uzair Ahmed Khuhro |
| Endpoint serving: the FastAPI service and its endpoints | Uzair Ahmed Khuhro |
| Explainability (xAI): SHAP and permutation importance | Zainab Soomro |
| AutoML: Ludwig hyperparameter search | Zainab Soomro |

## Milestone 1: Model & API (`v1.0-milestone1`)

| Task | Files | Owner |
|---|---|---|
| Data validation and per-IP feature engineering | `src/prepare_data.py`, `src/features.py` | Uzair Ahmed Khuhro |
| Data-preparation tests | `tests/test_prepare_data.py` | Uzair Ahmed Khuhro |
| Model training, MLflow tracking, model selection | `src/train.py`, `src/model_io.py` | Uzair Ahmed Khuhro |
| FastAPI service (`/predict`, `/explain`, `/health`, `/model-info`) | `src/app.py` | Uzair Ahmed Khuhro |
| Model-loading and API tests | `tests/test_model_io.py`, `tests/test_app.py`, `tests/test_api_extended.py` | Uzair Ahmed Khuhro |
| Explainability: global permutation importance | `src/explain.py`, `tests/test_explain.py` | Zainab Soomro |
| Ludwig AutoML search | `configs/ludwig_automl.yaml`, `src/automl_ludwig.py` | Zainab Soomro |
| Makefile, tooling | `Makefile`, `pyproject.toml` | Uzair Ahmed Khuhro |
| README, documentation | `README.md`, `AI_USAGE.md` | Uzair Ahmed Khuhro |
| Milestone 1 presentation | | Uzair Ahmed Khuhro |

## After Milestone 1

| Task | Files | Owner |
|---|---|---|
| Scoring by IP ID (`GET /predict/ip/{ip_id}`, `GET /ips/{ip_id}`) and from raw attack signals (`POST /predict/signals`, `POST /features/from-signals`) | `src/app.py`, `src/online_features.py`, `models/ip_lookup.parquet`, `models/feature_reference.json` | Uzair Ahmed Khuhro |
| Input consistency checks and stricter model-file checks | `src/app.py`, `src/model_io.py` | Uzair Ahmed Khuhro |
| Tests for the new endpoints and single-IP features | `tests/test_online_features.py` | Uzair Ahmed Khuhro |
| SHAP explanations (`POST /explain/shap`, `GET /explain/shap/global`), plots and dashboard view | `src/shap_explain.py`, `src/shap_report.py`, `ui/shap_view.py`, `tests/test_shap.py` | Zainab Soomro |
| Ludwig AutoML v2 (PR-AUC objective, nested trial runs) | `configs/ludwig_automl_v2.yaml`, `src/ludwig_metrics.py`, `src/ludwig_trials.py`, `experiments/backfill_ludwig_trials.py`, `experiments/check_ludwig_artifact.py` | Zainab Soomro |

## Reflections

Each member writes 2–3 lines on what they built and what they learned.

- **Zainab Soomro:** TODO
- **Uzair Ahmed Khuhro:** TODO
