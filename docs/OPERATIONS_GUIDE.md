# Operations guide (Windows Command Prompt)

Run every command from the project folder, in Command Prompt (`cmd`). `make` isn't needed.

## The two Python environments

| Environment | Python | Used for | Requirements |
|---|---|---|---|
| `.venv` | 3.12 | API, dashboard, training, experiments, SHAP, export, MLflow, DagsHub | `requirements.txt`, `requirements-ui.txt` |
| `.venv-ludwig` | 3.11 | Only the Ludwig search and loading its model | `requirements-ludwig.txt` |

They're separate because Ludwig brings in PyTorch and Ray, and Ludwig 0.11 needs Python 3.11.

```bat
:: Main environment
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-ui.txt

:: Ludwig environment: optional, about 2 GB
py -3.11 -m venv .venv-ludwig
.venv-ludwig\Scripts\python -m pip install -r requirements-ludwig.txt
.venv-ludwig\Scripts\python experiments\patch_ludwig_windows.py
```

The last line fixes a Windows-only crash inside the installed Ludwig package. See that file for what it changes.

To "activate" an environment, so you can type `python` instead of the full path, run `.venv\Scripts\activate`. Leave it with `deactivate`. This guide uses full paths, so activation isn't needed.

## Run the API and dashboard

```bat
:: Terminal 1: the API, with docs at http://127.0.0.1:8000/docs
.venv\Scripts\python -m uvicorn src.app:app --host 127.0.0.1 --port 8000

:: Terminal 2: the dashboard, at http://localhost:8501
.venv\Scripts\python -m streamlit run ui/streamlit_app.py
```

Check which model is being served:

```bat
curl http://127.0.0.1:8000/model-info
```

## Generate the Excel dataset

```bat
.venv\Scripts\python -m src.export_features
```

This writes to `data\exports\`. It reads `data\processed\*.parquet` and recalculates nothing, so run `.venv\Scripts\python -m src.prepare_data` first if that folder is missing.

## Test an IP with the Excel workbook

The model scores 28 engineered features, not an IP address, and this project has no automatic Shodan lookup. For a held-out test IP, the **Look up an IP ID** tab fetches its features from its `attacker_ip_enum`. For an IP that isn't in the dataset, enter its attack events and Shodan ports on the **From attack events** tab, and the API computes the 28 features. The steps below use the workbook on the **Enter all 28 features** tab.

1. Open `data\exports\ip_level_aggregated_features.xlsx`, or the smaller `ip_level_sample_for_testing.xlsx`.
2. On the `IP_features` sheet, filter to the rows you want. Prefer `split` = `test`, because the model never trained on those rows. `label` = 1 means VPN/proxy.
3. Open **IP Detection** in the dashboard, go to the **Enter all 28 features** tab, then use one of these:
   - **Most exact:** under "Load a real record", enter the `attacker_ip_enum`, then click **Load this IP**.
   - **From Excel:** copy the row's 28 cells from `n_signals` to `no_print_with_ports`, paste them under "Paste a row from Excel", then click **Fill the form**. Excel copies the rounded values it displays, so a score can differ slightly.
   - **By hand:** type each value. Hover over a field's help icon for its meaning and unit.
4. Click **Predict**. Compare the answer with the row's `label`.

In the sample workbook, `api_json` holds the full-precision values, and you can paste it under "Paste a row from Excel" or into `POST /predict` at `/docs`.

## SHAP explanations

```bat
:: The deployed model
.venv\Scripts\python -m src.shap_report

:: The candidate model. First create its bundle with experiments.package_candidate
.venv\Scripts\python -m experiments.package_candidate
.venv\Scripts\python -m src.shap_report --model models\candidates\exp_c_trees_400_no_early_stop.joblib
```

Plots go to `reports\figures\shap\<model name>\`. The deployed model takes about 3 minutes, and the 400-tree model about 10. Local explanations need no setup: they appear in the dashboard after each prediction, and at `POST /explain/shap`.

SHAP values describe how the model reached a score. They don't show that a feature causes an IP to be a VPN or proxy.

## MLflow

### View the local history

```bat
.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///reports/experiments/artifacts/zainab_mlflow.db --port 5000
```

Open http://127.0.0.1:5000, and press `Ctrl+C` to stop it. Choose the store you want to see:

| Database | Experiments |
|---|---|
| `reports/experiments/artifacts/zainab_mlflow.db` | `zainab_experiments` (89 runs), `zainab_ludwig_check` (1) |
| `reports/experiments/artifacts/baseline/baseline_repro_mlflow.db` | `zainab_baseline_repro_local` (5) |
| `reports/experiments/artifacts/model_packages_mlflow.db` | `zainab_ludwig_trials` (21: 1 parent + 20 trials), `zainab_model_packages` (3 logged models) |
| `mlflow.db` | Doesn't exist until you run `src.train` |

Useful filters: `tags.experiment_track = 'exp_c'`, `metrics.cv_f1 > 0.78`, `attributes.run_name = 'ludwig_automl'`. Choose models by `cv_f1`, not `test_f1`.

### Connect to DagsHub

Log in once, in your own terminal. It opens a browser and stores the token outside this project. Don't paste the token into a file in this repository.

```bat
.venv\Scripts\dagshub login
```

The remote history is at https://dagshub.com/uzair-khiiba/VPN-Classifier-MLOps.mlflow. Your account needs write access to that repository to upload. These login tokens are short-lived, so if you get an authentication error later, log in again.

### Choose where new runs go

```bat
:: experiments\ scripts: local is the default
set EXPERIMENT_TRACKING=dagshub
.venv\Scripts\python -m experiments.sync_mlflow

:: src.train: local mlflow.db is the default
set MLFLOW_TRACKING_URI=https://dagshub.com/uzair-khiiba/VPN-Classifier-MLOps.mlflow
```

`set` lasts only for that terminal window.

### Publish the existing history

This copies finished runs without rerunning anything. The local databases stay the source of truth, and are only read.

```bat
:: See what would be sent, without contacting anything
.venv\Scripts\python -m experiments.publish_history --plan

:: Copy, then check every value on the server
.venv\Scripts\python -m experiments.publish_history

:: Only compare local and remote
.venv\Scripts\python -m experiments.publish_history --verify-only
```

It's safe to repeat. Every copy carries a `mirror.source_run_id` tag, so finished runs are skipped and interrupted ones resume. The result is written to `reports\experiments\results\dagshub_sync_report.json`, which is the record of what's actually on the server.

The server can't use the name `zainab_experiments`, because looking it up returns errors. So those runs are under `zainab_experiments_history`, with their original run names. If support clears the old name, set `DAGSHUB_MAIN_EXPERIMENT` to choose another. The repository is public, so anything you log there is public too.

### Log the models

```bat
:: Build the candidate's API bundle. Doesn't touch the deployed model
.venv\Scripts\python -m experiments.package_candidate

:: A separate local database. Your experiment history stays unchanged
.venv\Scripts\python -m experiments.log_models

:: DagsHub, after publishing the history. --register adds the Model Registry entries
.venv\Scripts\python -m experiments.log_models --target dagshub --register
```

Each model is downloaded again and its predictions compared with the original. The results are in `reports\experiments\results\model_logging_<target>.json`.

### Load and test each model

Replace `<model_uri>` with the `model_uri` value from that results file.

```bat
:: The API bundles, from disk
.venv\Scripts\python -c "from src.model_io import load_bundle; b = load_bundle('models/model.joblib'); print(b['metadata']['model_name'], b['threshold'])"
.venv\Scripts\python -c "from src.model_io import load_bundle; b = load_bundle('models/candidates/exp_c_trees_400_no_early_stop.joblib'); print(b['metadata']['model_name'], b['threshold'])"

:: An sklearn model logged locally
.venv\Scripts\python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///reports/experiments/artifacts/model_packages_mlflow.db'); print(mlflow.sklearn.load_model('<model_uri>'))"

:: The Ludwig model, using its own environment. Compares its predictions with the saved ones
.venv-ludwig\Scripts\python -m experiments.check_ludwig_artifact ludwig_results\best_model
```

MLflow's `predict_proba` gives the VPN/proxy probability in column 1. Apply the threshold from `model_info/model_card.json`: MLflow doesn't store it as part of the prediction.

## Change the deployed model

Logging or registering a model never deploys it.

```bat
:: Shows the checks, and changes nothing
.venv\Scripts\python -m experiments.promote_model models\candidates\exp_c_trees_400_no_early_stop.joblib

:: Deploys it. Only do this after a real decision
.venv\Scripts\python -m experiments.promote_model models\candidates\exp_c_trees_400_no_early_stop.joblib --approve --reason "why"
```

This refuses a lower CV F1 whatever the test score, archives the old model in `models\archive\`, and records the decision in `models\promotion_log.jsonl`. Restart the API afterwards. `test_deployed_model_has_not_changed` deliberately skips once a promotion is recorded.

To undo it, run `promote_model` on the file in `models\archive\`. If the CV check refuses, copy that file back by hand. An archived model holds the same metrics it had when it was deployed.

## Ludwig search

```bat
:: A quick check: about 10 to 20 minutes
set PYTHONPATH=.
.venv-ludwig\Scripts\python -m src.automl_ludwig --num-samples 3 --n-initial-points 2 --output-dir ludwig_results_smoke

:: The full v2 search: 30 trials, up to a 2-hour budget plus the final retrain
.venv-ludwig\Scripts\python -m src.automl_ludwig
```

Logging goes to `MLFLOW_TRACKING_URI`, so set it first. To keep new runs with the experiment history:

```bat
set MLFLOW_TRACKING_URI=sqlite:///reports/experiments/artifacts/model_packages_mlflow.db
set MLFLOW_EXPERIMENT=zainab_ludwig_trials
```

Results go to `ludwig_results_v2\`. A folder that already has results is refused, so `ludwig_results\` stays as it is. To repeat a search, pass a new `--output-dir`.

`experiments\run_ludwig.sh` is the record of how the first search ran. It deletes `ludwig_results`, so it refuses to run without `--delete-existing-results`. Don't use it for new searches.

## Evaluation

```bat
:: Rebuild the model comparison. Stops if a score doesn't match its saved record
.venv\Scripts\python -m experiments.verified_comparison

:: Population-feature audit: takes a few minutes, because it reads the raw signals
.venv\Scripts\python -m experiments.audit_population_features
```

The comparison is written to `reports\experiments\verified_model_comparison.md`.

## Tests and lint

```bat
.venv\Scripts\python -m pytest
.venv\Scripts\python -m pytest experiments\test_experiment_helpers.py
.venv\Scripts\python -m ruff check src tests ui experiments
```
