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

## Milestone 1: Model & API (`v1.0-milestone1`)

| Task | Files | Owner |
|---|---|---|
| Data validation and per-IP feature engineering | `src/prepare_data.py`, `src/features.py` | Uzair Ahmed Khuhro |
| Data-preparation tests | `tests/test_prepare_data.py` | Uzair Ahmed Khuhro |
| Model training, MLflow tracking, model selection | `src/train.py`, `src/model_io.py` | Uzair Ahmed Khuhro |
| Ludwig AutoML search | `configs/ludwig_automl.yaml`, `src/automl_ludwig.py` | Uzair Ahmed Khuhro |
| FastAPI service (`/predict`, `/explain`, `/health`) | `src/app.py` | Uzair Ahmed Khuhro |
| Model-loading and API tests | `tests/test_model_io.py`, `tests/test_app.py` | Uzair Ahmed Khuhro |
| Explainability (global permutation importance, `/explain`) | `src/explain.py`, `src/app.py`, `tests/test_explain.py` | Uzair Ahmed Khuhro |
| Makefile, tooling | `Makefile`, `pyproject.toml` | Uzair Ahmed Khuhro |
| README, documentation | `README.md`, `AI_USAGE.md` | Uzair Ahmed Khuhro |
| Milestone 1 presentation | | Uzair Ahmed Khuhro |

### Reflections

Each member writes 2–3 lines on what they built and what they learned.

- **Zainab Soomro:** TODO
- **Uzair Ahmed Khuhro:** TODO
