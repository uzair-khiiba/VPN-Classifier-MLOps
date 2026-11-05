# VPN/Proxy IP Classifier. Run `make` (or `make help`) to list every command.
#
# The one command you need:
#     make run          prepare the data if needed -> train all models -> log them to MLflow -> run pytest
# Then:
#     make mlflow       see the training runs in the MLflow UI (http://127.0.0.1:5000)
#     make serve        serve the deployed model (http://127.0.0.1:8000/docs)
#
# Training saves each new model, versioned, in models/runs/. It never overwrites the deployed
# models/model.joblib: that changes only through `python -m experiments.promote_model`.
#
# On Windows, make isn't usually installed: docs/OPERATIONS_GUIDE.md lists the same steps for Command Prompt.

ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
LUDWIG_PY ?= .venv-ludwig/Scripts/python
else
PY ?= .venv/bin/python
LUDWIG_PY ?= .venv-ludwig/bin/python
endif

# Where `make train` logs its runs, and what `make mlflow` opens by default.
TRACKING_DB ?= mlflow.db
EXPERIMENTS_DB := reports/experiments/artifacts/zainab_mlflow.db
SAMPLE_FLAG := $(if $(filter 1,$(SAMPLE)),--sample,)

# SAMPLE=1 runs everything on the small synthetic sample in separate folders and a separate
# MLflow experiment, so it never touches the real results. Without the dataset, use `make run SAMPLE=1`.
# The folders are passed only to the commands that use them, never to `make test`, so pytest always
# checks the committed model and lookup data.
ifeq ($(SAMPLE),1)
PROCESSED_DIR := data/processed-sample
SAMPLE_ENV := PROCESSED_DIR=$(PROCESSED_DIR) MODELS_DIR=models/sample MODEL_PATH=models/sample/model.joblib \
	MLFLOW_EXPERIMENT=vpn-proxy-classification-sample
TRAIN_ENV := $(SAMPLE_ENV)
else
PROCESSED_DIR := data/processed
SAMPLE_ENV :=
TRAIN_ENV := MODELS_DIR=models/runs MODEL_PATH=models/runs/latest.joblib
endif
TRAIN_ENV += MLFLOW_TRACKING_URI=sqlite:///$(CURDIR)/$(TRACKING_DB) MLFLOW_DISABLE_AGENT_HINT=1

.DEFAULT_GOAL := help
.PHONY: help run setup setup-ui setup-automl data train test lint mlflow mlflow-experiments serve ui \
	lookup explain export shap comparison log-models publish-plan publish automl all clean clean-generated

help:             ## list the commands
	@echo "Main commands (in order):"
	@grep -E '^(setup|setup-ui|run|data|train|test|mlflow|serve|ui):' $(MAKEFILE_LIST) | sed -E 's/:[^#]*## /\t/' | awk -F'\t' '{printf "  make %-12s %s\n", $$1, $$2}'
	@echo ""
	@echo "Other commands:"
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | grep -vE '^(help|setup|setup-ui|run|data|train|test|mlflow|serve|ui):' | sed -E 's/:[^#]*## /\t/' | awk -F'\t' '{printf "  make %-19s %s\n", $$1, $$2}'
	@echo ""
	@echo "Add SAMPLE=1 to run on the small synthetic sample instead of the real data."

# ---- Main pipeline ------------------------------------------------------------------------

setup:            ## create .venv (Python 3.12) and install the pinned packages
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt

setup-ui:         ## add the dashboard's packages (Streamlit, Altair, SHAP) to .venv
	uv pip install --python $(PY) -r requirements-ui.txt

run:              ## THE ONE COMMAND: prepare data if needed, train + log to MLflow, then run pytest
	@test -f $(PROCESSED_DIR)/train.parquet || $(MAKE) --no-print-directory data
	$(MAKE) --no-print-directory train
	$(MAKE) --no-print-directory test
	@echo ""
	@echo "Done. Training runs are in MLflow ($(TRACKING_DB)): open them with  make mlflow"
	@echo "The new model is in $(if $(filter 1,$(SAMPLE)),models/sample/,models/runs/). The API keeps serving models/model.joblib."

data:             ## validate the raw data and build the per-IP train/test tables
	$(SAMPLE_ENV) $(PY) -m src.prepare_data $(SAMPLE_FLAG)

train:            ## train all candidate models, log each run to MLflow, save the best one (versioned)
	$(TRAIN_ENV) $(PY) -m src.train

test:             ## run the pytest suite
	$(PY) -m pytest

mlflow:           ## open the MLflow UI on the training runs (http://127.0.0.1:5000)
	$(PY) -m mlflow ui --backend-store-uri sqlite:///$(TRACKING_DB) --port 5000

serve:            ## serve the deployed model: API on http://127.0.0.1:8000, docs at /docs
	$(SAMPLE_ENV) $(PY) -m uvicorn src.app:app --host 0.0.0.0 --port 8000

ui:               ## run the dashboard on http://localhost:8501 (start `make serve` first)
	$(PY) -m streamlit run ui/streamlit_app.py

# ---- Other commands -----------------------------------------------------------------------

lint:             ## check the code style with ruff
	$(PY) -m ruff check src tests ui experiments

mlflow-experiments: ## open the MLflow UI on the experiment track's 89 runs
	$(PY) -m mlflow ui --backend-store-uri sqlite:///$(EXPERIMENTS_DB) --port 5000

lookup:           ## rebuild the API's test-IP lookup table and feature reference (models/)
	$(SAMPLE_ENV) $(PY) -m src.online_features $(SAMPLE_FLAG)

explain:          ## permutation importance of the deployed model on the test set
	$(SAMPLE_ENV) $(PY) -m src.explain

export:           ## write the IP-level Excel, Parquet and CSV files to data/exports/
	$(PY) -m src.export_features

shap:             ## SHAP plots and ranking for the deployed model, in reports/figures/shap/
	$(PY) -m src.shap_report

comparison:       ## rebuild the verified model comparison from saved models and predictions
	$(PY) -m experiments.verified_comparison

log-models:       ## log the three models to the local packages database, and check they reload
	$(PY) -m experiments.log_models

publish-plan:     ## show what would be copied to DagsHub, contacting nothing
	$(PY) -m experiments.publish_history --plan

publish:          ## copy local run history to DagsHub (run `dagshub login` first), then verify it
	$(PY) -m experiments.publish_history

setup-automl:     ## separate .venv-ludwig for the optional Ludwig search (PyTorch + Ray, ~2 GB)
	uv venv --python 3.11 .venv-ludwig
	uv pip install --python $(LUDWIG_PY) -r requirements-ludwig.txt

automl:           ## Ludwig hyperparameter search (configs/ludwig_automl_v2.yaml), 1-2 hours. TRIALS=n for fewer
	$(LUDWIG_PY) -m src.automl_ludwig $(if $(TRIALS),--num-samples $(TRIALS),)

all: run          ## same as `make run`

clean:            ## remove only the synthetic-sample outputs, which are quick to rebuild
	rm -rf data/processed-sample models/sample

clean-generated:  ## delete real results and models too. Refuses unless CONFIRM=delete-results
ifeq ($(CONFIRM),delete-results)
	rm -rf data/processed data/processed-sample models/sample models/runs ludwig_results mlruns mlflow.db models/*.joblib models/model_meta.json
else
	@echo "Refusing: this deletes trained models and results. Add CONFIRM=delete-results to really do it."
endif
