# On Windows, make isn't usually installed: README "Windows commands" lists the same steps for Command Prompt.
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
LUDWIG_PY ?= .venv-ludwig/Scripts/python
else
PY ?= .venv/bin/python
LUDWIG_PY ?= .venv-ludwig/bin/python
endif
TRACKING_DB ?= reports/experiments/artifacts/zainab_mlflow.db
SAMPLE_FLAG := $(if $(filter 1,$(SAMPLE)),--sample,)

# SAMPLE=1 runs the pipeline on the synthetic sample in separate folders, so the
# committed real model in models/model.joblib is never overwritten.
ifeq ($(SAMPLE),1)
export PROCESSED_DIR := data/processed-sample
export MODELS_DIR := models/sample
export MODEL_PATH := models/sample/model.joblib
export MLFLOW_EXPERIMENT := vpn-proxy-classification-sample
endif

.PHONY: setup setup-automl setup-ui data train explain export shap comparison log-models publish-plan publish \
	automl serve ui mlflow test lint all clean clean-generated

setup:            ## create .venv with Python 3.12 and install pinned dependencies
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt

setup-automl:     ## separate .venv-ludwig for the optional Ludwig AutoML run (PyTorch + Ray)
	uv venv --python 3.11 .venv-ludwig
	uv pip install --python $(LUDWIG_PY) -r requirements-ludwig.txt

setup-ui:         ## add the Streamlit dashboard's dependencies to .venv
	uv pip install --python $(PY) -r requirements-ui.txt

data:             ## validate raw data and build per-IP train/test features (SAMPLE=1 for the small sample)
	$(PY) -m src.prepare_data $(SAMPLE_FLAG)

train:            ## train all candidates, log them to MLflow, save the best model (SAMPLE=1: sample run, saved to models/sample/)
	$(PY) -m src.train

explain:          ## global explainability: permutation importance of the served model on the test set
	$(PY) -m src.explain

export:           ## write the IP-level Excel, Parquet and CSV files to data/exports/
	$(PY) -m src.export_features

shap:             ## SHAP plots and ranking for the served model, in reports/figures/shap/
	$(PY) -m src.shap_report

comparison:       ## rebuild the verified model comparison from saved models and predictions
	$(PY) -m experiments.verified_comparison

log-models:       ## log the three models to the local packages database, and check they reload
	$(PY) -m experiments.log_models

publish-plan:     ## show what would be copied to DagsHub, contacting nothing
	$(PY) -m experiments.publish_history --plan

publish:          ## copy local run history to DagsHub (run `dagshub login` first), then verify it
	$(PY) -m experiments.publish_history

automl:           ## Ludwig hyperparameter search (configs/ludwig_automl_v2.yaml), about 1-2 hours. TRIALS=n for fewer
	$(LUDWIG_PY) -m src.automl_ludwig $(if $(TRIALS),--num-samples $(TRIALS),)

serve:            ## run the API on http://127.0.0.1:8000 (docs at /docs)
	$(PY) -m uvicorn src.app:app --host 0.0.0.0 --port 8000

ui:               ## run the dashboard on http://localhost:8501 (start `make serve` first)
	$(PY) -m streamlit run ui/streamlit_app.py

mlflow:           ## MLflow UI on http://127.0.0.1:5000 (TRACKING_DB=path for another database)
	$(PY) -m mlflow ui --backend-store-uri sqlite:///$(TRACKING_DB) --port 5000

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests ui experiments

all: data train explain   ## full pipeline: prepare data, train, explain

clean:            ## remove only the synthetic-sample outputs, which are quick to rebuild
	rm -rf data/processed-sample models/sample

clean-generated:  ## delete real results and models too. Refuses unless CONFIRM=delete-results
ifeq ($(CONFIRM),delete-results)
	rm -rf data/processed data/processed-sample models/sample ludwig_results mlruns mlflow.db models/*.joblib models/model_meta.json
else
	@echo "Refusing: this deletes trained models and results. Add CONFIRM=delete-results to really do it."
endif
