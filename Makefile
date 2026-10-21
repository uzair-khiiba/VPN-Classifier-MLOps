PY ?= .venv/bin/python
SAMPLE_FLAG := $(if $(filter 1,$(SAMPLE)),--sample,)

# SAMPLE=1 runs the pipeline on the synthetic sample in separate folders, so the
# committed real model in models/model.joblib is never overwritten.
ifeq ($(SAMPLE),1)
export PROCESSED_DIR := data/processed-sample
export MODELS_DIR := models/sample
export MODEL_PATH := models/sample/model.joblib
export MLFLOW_EXPERIMENT := vpn-proxy-classification-sample
endif

.PHONY: setup setup-automl data train explain automl serve mlflow test lint all clean

setup:            ## create .venv with Python 3.11 and install pinned dependencies
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt

setup-automl:     ## separate .venv-ludwig for the optional Ludwig AutoML run (PyTorch + Ray)
	uv venv --python 3.11 .venv-ludwig
	uv pip install --python .venv-ludwig/bin/python -r requirements-ludwig.txt

data:             ## validate raw data and build per-IP train/test features (SAMPLE=1 for the small sample)
	$(PY) -m src.prepare_data $(SAMPLE_FLAG)

train:            ## train all candidates, log them to MLflow, save the best model (SAMPLE=1: sample run, saved to models/sample/)
	$(PY) -m src.train

explain:          ## global explainability: permutation importance of the served model on the test set
	$(PY) -m src.explain

automl:           ## Ludwig hyperparameter search (configs/ludwig_automl.yaml), best trial logged to MLflow
	.venv-ludwig/bin/python -m src.automl_ludwig $(if $(TRIALS),--num-samples $(TRIALS),)

serve:            ## run the API on http://127.0.0.1:8000 (docs at /docs)
	$(PY) -m uvicorn src.app:app --host 0.0.0.0 --port 8000

mlflow:           ## MLflow UI on http://127.0.0.1:5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

all: data train explain   ## full pipeline: prepare data, train, explain

clean:
	rm -rf data/processed data/processed-sample models/sample ludwig_results mlruns mlflow.db models/*.joblib models/model_meta.json
