# AI Usage

The course policy allows AI coding assistants for boilerplate, debugging and documentation drafting, provided the group understands and has reviewed everything it submits.

## Tools used

| Tool | Used for |
|---|---|
| GPTs | Reviewing the proposal against the course spec; drafting the project structure and documentation; debugging Ludwig installation and configuration issues |


## How it was used, by area

| Area (owner) | What the assistant did | What the team did |
|---|---|---|
| Preprocessing (Uzair) | Drafted `src/prepare_data.py` and `src/features.py`; refactored the feature code so the API reuses it for a single IP (`src/online_features.py`) | Chose the per-IP framing and the features; asked for scoring from an IP ID or from raw attack events instead of typing 28 values, and chose raw events over a short summary form |
| MLflow (Uzair) | Drafted training, tracking and model selection in `src/train.py` | |
| Endpoint serving (Uzair) | Drafted the FastAPI service; added `/predict/ip/{ip_id}`, `/ips/{ip_id}`, `/predict/signals` and `/features/from-signals`; reviewed `src/app.py` and fixed what the review found (input consistency checks, stricter model-file checks, case-insensitive attack types and fingerprints) | Asked for the review and decided which findings to fix |
| Explainability (Zainab) | Helped build permutation importance, SHAP (`src/shap_explain.py`, `src/shap_report.py`) and the dashboard's SHAP view | |
| AutoML (Zainab) | Helped with the Ludwig configs, the PR-AUC metric, trial tracking and Ludwig debugging | |
| Dashboard and docs (both) | Built the three-tab IP Detection page; merged both members' dashboard work; made the light, projector-friendly theme with a colourblind-safe palette; updated `README.md`, the guides and `CONTRIBUTION.md` | Asked for the light theme for projector presentation; set the ownership split |
| Repository maintenance (Uzair) | Resolved the 11 conflicting files when combining Zainab's pull request with Uzair's work; removed temporary and duplicate files | Decided to keep all of Zainab's work, including DagsHub publishing, and chose which leftover files to delete |

## How the assistant's output was checked

- **Tests:** the pytest suite (202 tests) and Ruff lint, run after every change.
- **Features for a single IP:** checked against the training pipeline. They matched exactly on every IP of the synthetic sample and on 500 real held-out IPs rebuilt from the raw signals.
- **Input rules on `/predict`:** checked against all 148,259 real IPs, so they never reject a record the pipeline can produce.
- **Live checks:** the API and dashboard were run with the real model, and each endpoint was called.
- **Chart colours:** checked with a palette validator, for colourblind safety and contrast on the white background.
