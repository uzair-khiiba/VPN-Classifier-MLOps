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
| Proposal slides and presentation | **To fill in:** the project record doesn't say who presented |

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

## Milestone 2: Containerization, CI/CD & Deployment (`v2.0-milestone2`)

| Task | Files | Owner |
|---|---|---|
| Dockerfile (two-stage, non-root, `/health` healthcheck), slim runtime requirements | `Dockerfile`, `.dockerignore`, `requirements-api.txt` | Uzair Ahmed Khuhro |
| GitHub Actions CI: lint, test, Docker build + smoke test, push to GHCR | `.github/workflows/ci.yml` | Uzair Ahmed Khuhro |
| Deployment on Render, deploy job, rollback by image tag | `render.yaml`, `.github/workflows/ci.yml`, `RUNBOOK.md` | Uzair Ahmed Khuhro |
| Monitoring: per-request logging, `GET /metrics` counters, build ID in `/health` | `src/monitoring.py`, `src/app.py` | Uzair Ahmed Khuhro |
| Monitoring tests | `tests/test_monitoring.py` | Uzair Ahmed Khuhro |
| Project Focus (AutoML vs manual training): scope, fair-comparison protocol, plan | `README.md` (Project Focus) | Zainab Soomro |
| AutoML groundwork carried into the focus (Ludwig hands-off and guided runners, PR-AUC objective, tests) | `src/automl_simple.py`, `src/automl_ludwig.py`, `tests/test_automl_*.py` | Zainab Soomro |
| README deployment, monitoring and cURL sections; Makefile Docker targets | `README.md`, `Makefile` | Uzair Ahmed Khuhro |
| Dashboard image and CI job (`ui.Dockerfile`, build-dashboard, deploy-dashboard) | `ui.Dockerfile`, `requirements-dashboard.txt`, `.github/workflows/ci.yml`, `render.yaml` | Uzair Ahmed Khuhro |
| Milestone 2 finishing: `/stats` alias, `model_type` in `/model-info`, deploy jobs that report a missing secret as skipped, live checks after deploy, runbook troubleshooting, README status and limitations | `src/app.py`, `.github/workflows/ci.yml`, `tests/test_monitoring.py`, `tests/test_api_extended.py`, `RUNBOOK.md`, `README.md` | Zainab Soomro |

## Reflections

> **Drafts.** These were drafted from the project record (commits, files and this table), not written by the members. Each member must read their own, correct anything that's wrong, and rewrite it in their own words before submission.

- **Zainab Soomro (draft):** I worked on explainability and on making the project's results trustworthy: SHAP explanations in the API and dashboard, the Excel export used for manual testing, publishing our experiment history and models to DagsHub, the approval-only model promotion script, and finishing Milestone 2's monitoring, CI deploy checks and documentation. The main lesson was to check claims against evidence. Our Ludwig search was configured for TPE but turned out to be effectively random, because Ludwig silently ignored a setting, and our dashboard deploy showed as successful while it was being skipped. Next time I'd add those checks as tests from the start.
- **Uzair Ahmed Khuhro (draft):** I built the core pipeline and its serving: data validation and per-IP feature engineering, training with MLflow tracking, the FastAPI service including scoring by IP ID and from raw signals, and in Milestone 2 the Docker image, the GitHub Actions pipeline, the Render deployment and request monitoring. I learned to make each step reproducible and checkable: the image is smoke-tested in CI before it's pushed, and `/health` reports exactly which build is live. Next I'd move monitoring to persistent storage, and add the dashboard's Render service so both parts are hosted.
