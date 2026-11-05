# UI and API guide

This guide explains how the dashboard and the API work together, and how to run and test them.

## The big picture

The project has two separate programs:

```
Browser  ->  Streamlit dashboard  --HTTP-->  FastAPI backend  ->  Trained model
                                 <--JSON--
```

| Part | File | Job |
|---|---|---|
| **Frontend** (Streamlit) | `ui/streamlit_app.py` | Shows pages and forms, and sends requests. It never loads the model |
| **Backend** (FastAPI) | `src/app.py` | Validates requests, runs the model, and returns JSON |
| **Model** | `models/model.joblib` | The trained classifier, loaded once when the backend starts |

Keeping them separate means you can test or replace either one on its own.

## What is an API endpoint?

An endpoint is an address that a program can send a request to, like a counter that handles one kind of job. Each one has a method (`GET` to read, `POST` to send data) and a path.

| Endpoint | What it does |
|---|---|
| `GET /health` | Reports whether the API is running and a model is loaded |
| `GET /model-info` | Gives the served model's identity and saved evaluation results |
| `POST /predict` | Takes one IP's features, and returns a prediction |
| `POST /explain/shap` | Returns SHAP values for the prediction, and names the explainer that ran |
| `GET /explain/shap/global` | Returns the saved SHAP ranking, if it matches the served model |
| `POST /explain` | Returns the prediction, plus how much each feature moved it |
| `GET /docs` | Interactive documentation (Swagger UI) |

## Which model is served?

The API serves **`hgb_slow`**, version `20261019-173911`, from `models/model.joblib`.

The experimental 400-tree model scored higher in `reports/experiments/`, but **it isn't served**. Its saved file is a bare estimator, without the threshold and medians the API needs, and adopting it hasn't been decided. The dashboard shows it only as a labelled comparison.

## What the model needs as input

**An IP address alone isn't enough, and this project has no IP lookup.** The model takes 28 numbers that summarise an IP's behaviour:

| Group | Features |
|---|---|
| Volume and timing | `n_signals`, `active_hours`, `n_active_days`, `signals_per_day`, `max_signals_per_hour`, `mean_gap_s`, `median_gap_s` |
| Breadth | `n_attack_types`, `n_watchers`, `n_watcher_as`, `n_watcher_countries`, `n_attacker_as` |
| Attack mix (0 to 1) | `share_http_spam`, `share_http_scan`, `share_http_exploit`, `share_http_bruteforce`, `share_http_crawl`, `share_ssh_bruteforce`, `share_pop3_imap_bruteforce`, `share_tcp_scan` |
| Network crowding | `asn_unique_ips` |
| Shodan | `has_shodan`, `n_tcp_ports`, `n_udp_ports`, `jarm_shared`, `ja3s_shared`, `headers_hash_shared`, `no_print_with_ports` |

These come from aggregating CrowdSec signals and Shodan data (`src/features.py`). The form builds itself from the API's `/openapi.json`, so it always matches what the API expects. Any preprocessing is inside the saved model, so the API needs no extra step.

**Serving caveat:** `headers_hash_shared`, `jarm_shared`, `ja3s_shared` and `asn_unique_ips` are counts across a population of IPs. They must be computed exactly as in training, or scores become unreliable. The experiments found that being off by one cost about 0.06 F1.

## Starting both services

You need two Command Prompt windows, both in the project folder.

**First time only,** install the dashboard's packages:

```bat
.venv\Scripts\python -m pip install -r requirements-ui.txt
```

**Terminal 1: the backend**

```bat
.venv\Scripts\python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Wait until you see `Application startup complete`.

**Terminal 2: the dashboard**

```bat
.venv\Scripts\python -m streamlit run ui/streamlit_app.py
```

It opens at <http://localhost:8501>. On its very first run, Streamlit may ask for an email address in the terminal. You can leave it blank and press Enter.

To stop either one, press `Ctrl+C` in its window.

## Configuring the API URL

The dashboard looks for the API at `http://127.0.0.1:8000` by default. To change that, in order of priority:

1. **Environment variable,** in the terminal before starting the dashboard:

   ```bat
   set API_BASE_URL=http://127.0.0.1:8010
   .venv\Scripts\python -m streamlit run ui/streamlit_app.py
   ```

2. **A `.env` file:** copy `.env.example` to `.env` and edit it.

Avoid `localhost` here on Windows, because it can try another address first and add a delay to every request.

The dashboard calls the API from its own server, not from the browser, so the API needs no CORS settings. Both services listen only on this computer (`127.0.0.1`).

## How `/health` works

The API reports its state without running the model. The API Status page also measures how long the request took.

**Request:** `GET /health`

**Response with a model loaded:**

```json
{"status": "ok", "model_loaded": true, "model_status": "loaded", "model": "hgb_slow", "version": "20261019-173911"}
```

**Response with no valid model.** The API keeps running, and the real reason goes only to the server log, so file paths stay private:

```json
{"status": "degraded", "model_loaded": false, "model_status": "failed", "detail": "The model file is missing or invalid. See the server log for details."}
```

## How `/predict` works

1. FastAPI checks the request against the `IPFeatures` schema.
2. The features go to the model in the exact order it was trained on.
3. The model gives a score from 0 to 1.
4. The score is compared with the saved threshold (0.9015).

**Request:** `POST /predict`, with all 28 features. A full example is in `examples/vpn_ip.json`.

```bat
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @examples/vpn_ip.json
```

**Response:**

```json
{"prediction": 1, "label": "VPN/Proxy", "is_vpn_proxy": true, "probability": 0.9923, "threshold": 0.9015, "model_version": "hgb_slow-20261019-173911"}
```

The `probability` field is the model's score. Class weights used in training inflate it, so read it against the threshold and not as a literal chance.

**Invalid input** returns HTTP 422 with the field and the reason:

```json
{"detail": [{"type": "greater_than_equal", "loc": ["body", "n_signals"], "msg": "Input should be greater than or equal to 1"}]}
```

| Status | Meaning |
|---|---|
| 200 | Success |
| 422 | A field is missing, unknown or out of range |
| 503 | The API is running, but no model is loaded |

## How `/explain` works

For each feature, the API replaces **only that feature** with its median from the training data, scores the IP again, and reports the change.

**Request:** `POST /explain?top_k=2`, with the same body as `/predict`.

**Response** (shortened to the first feature):

```json
{
  "prediction": 1,
  "label": "VPN/Proxy",
  "probability": 0.9923,
  "threshold": 0.9015,
  "method": "single-feature replacement with training median",
  "contributions": [
    {"feature": "has_shodan", "value": 1.0, "reference": 0.0, "effect": 0.3941,
     "probability_at_reference": 0.5982, "changes_prediction": true}
  ]
}
```

- `effect`: positive means this IP's value pushes the score toward VPN/proxy.
- `changes_prediction`: true if resetting only this feature would flip the class.

**Limits:**

- It describes how the model behaves, and doesn't show causes.
- It is **not SHAP**. Effects are measured one at a time, and needn't add up to the score.
- Features working together can be missed, and related features can cover for each other.
- The median can create impossible combinations, such as no Shodan record with open ports.

## Testing the API through Swagger

1. Start the backend, then open <http://127.0.0.1:8000/docs>.
2. Open `POST /predict` and click **Try it out**. A sample record is already filled in.
3. Click **Execute** and read the response below.
4. Change `n_signals` to `-1` and execute again, to see the 422 error.

## Testing through the dashboard

1. **API Status:** check that it says Online and Loaded.
2. **IP Detection:** click **Load: Sample labelled VPN/proxy**, then **Predict**. The result is marked as a demo record, because it's a stored test record and not a live IP.
3. **Explainability:** click **Explain this prediction** to see the chart.
4. **Model Performance:** see the saved results. Some sections need `reports/experiments/`, and say so if it's absent.
5. **Try a failure:** stop the backend and click **Check now** on API Status. It should say Offline and show how to restart it.

## Troubleshooting

| What you see | What to do |
|---|---|
| "The API is offline" | Start the backend in Terminal 1 |
| "API online, no model" | `models/model.joblib` is missing or invalid. The backend's terminal shows why |
| `No module named streamlit` | Run the first-time install command above |
| A "Page not found" box | The link is out of date. The Overview page is at `/`, with no path |
| Port already in use | Another copy is running. Stop it, or use another port and set `API_BASE_URL` |

## Automated tests

```bat
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check src tests ui
```

| File | Covers |
|---|---|
| `tests/test_app.py`, `tests/test_api_extended.py` | The API: health, prediction, invalid input, missing features, no model, explanations, documentation |
| `tests/test_ui_client.py` | The HTTP client: real calls to the API, backend down, invalid responses, and that `ui/` never imports the model |
| `tests/test_ui_app.py` | The pages, run headlessly against the API, including offline behaviour |

The tests use a tiny model made during the test run, so they need neither the dataset nor retraining.
