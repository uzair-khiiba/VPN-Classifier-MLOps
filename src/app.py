"""FastAPI service: VPN/proxy prediction for one attacker IP.

Run:  uvicorn src.app:app --reload     then open http://127.0.0.1:8000/docs
"""

import json
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.config import MODEL_PATH, ROOT
from src.model_io import ModelLoadError, load_bundle

logger = logging.getLogger("uvicorn.error")

STATE: dict = {"bundle": None, "error": None}

LABEL_POSITIVE = "VPN/Proxy"
LABEL_NEGATIVE = "Not VPN/Proxy"
# Shown to API clients. The specific reason, which can include file paths, goes to the server log only.
MODEL_ERROR = "The model file is missing or invalid. See the server log for details."
MODEL_UNAVAILABLE = {503: {"description": "No valid model is loaded, so predictions are unavailable"}}

# Two typical IPs from the held-out test set, shown as request examples in /docs.
EXAMPLES = [json.loads((ROOT / "examples" / f).read_text()) for f in ("vpn_ip.json", "regular_ip.json")]


def load_model(path=MODEL_PATH) -> None:
    try:
        STATE["bundle"], STATE["error"] = load_bundle(path), None
    except ModelLoadError as exc:
        logger.error("Model not loaded: %s", exc)
        STATE["bundle"], STATE["error"] = None, MODEL_ERROR


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="VPN/Proxy IP Classifier",
    description="Predicts whether an attacker IP is a VPN/proxy from its aggregated CrowdSec signals. "
                "It takes 28 pre-computed features, not an IP address: the API does not look up IPs.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Report the field and reason without echoing the input: a value like 1e999 parses to
    infinity, which can't be written back as JSON and would otherwise turn this 422 into a 500."""
    errors = [{key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


def _share(attack_type: str):
    return Field(ge=0, le=1, description=f"Fraction of this IP's signals that are {attack_type}")


class IPFeatures(BaseModel):
    """Aggregated behaviour of one attacker IP (see src/features.py for definitions)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, json_schema_extra={"examples": EXAMPLES})

    n_signals: float = Field(ge=1, description="Number of attack signals from this IP")
    n_attack_types: float = Field(ge=1, description="Distinct attack types used")
    n_watchers: float = Field(ge=1, description="Distinct CrowdSec watchers that reported the IP")
    n_watcher_as: float = Field(ge=0, description="Distinct watcher autonomous systems")
    n_watcher_countries: float = Field(ge=0, description="Distinct watcher countries (country can be missing)")
    n_attacker_as: float = Field(ge=0, description="Distinct attacker autonomous systems")
    active_hours: float = Field(ge=0, description="Hours between first and last signal")
    n_active_days: float = Field(ge=1, description="Distinct days with at least one signal")
    signals_per_day: float = Field(ge=0, description="n_signals / n_active_days")
    max_signals_per_hour: float = Field(ge=1, description="Busiest hour's signal count")
    mean_gap_s: float = Field(ge=0, description="Mean seconds between consecutive signals")
    median_gap_s: float = Field(ge=0, description="Median seconds between consecutive signals")
    asn_unique_ips: float = Field(ge=0, description="Distinct attacker IPs seen in this IP's ASN")
    share_http_spam: float = _share("http:spam")
    share_http_scan: float = _share("http:scan")
    share_http_exploit: float = _share("http:exploit")
    share_http_bruteforce: float = _share("http:bruteforce")
    share_http_crawl: float = _share("http:crawl")
    share_ssh_bruteforce: float = _share("ssh:bruteforce")
    share_pop3_imap_bruteforce: float = _share("pop3/imap:bruteforce")
    share_tcp_scan: float = _share("tcp:scan")
    has_shodan: float = Field(ge=0, le=1, description="1 if Shodan recorded at least one open port, otherwise 0")
    n_tcp_ports: float = Field(ge=0, description="Open TCP ports recorded by Shodan")
    n_udp_ports: float = Field(ge=0, description="Open UDP ports recorded by Shodan")
    jarm_shared: float = Field(ge=0, description="Other IPs sharing this IP's JARM fingerprints")
    ja3s_shared: float = Field(ge=0, description="Other IPs sharing this IP's JA3S fingerprints")
    headers_hash_shared: float = Field(ge=0, description="Other IPs sharing this IP's HTTP header hashes")
    no_print_with_ports: float = Field(ge=0, le=1, description="1 if Shodan saw ports but no fingerprints, otherwise 0")


class Prediction(BaseModel):
    prediction: int = Field(description="Predicted class: 1 = VPN/proxy, 0 = not VPN/proxy")
    label: str = Field(description="Human-readable predicted class")
    is_vpn_proxy: bool
    probability: float = Field(description="Model score for the VPN/proxy class, from 0 to 1")
    threshold: float = Field(description="The prediction is VPN/proxy when probability >= threshold")
    model_version: str


class Contribution(BaseModel):
    feature: str
    value: float
    reference: float = Field(description="This feature's median in the training data")
    effect: float = Field(description="Probability change caused by this feature's value vs. the training median")
    probability_at_reference: float = Field(description="Probability when only this feature is set to its median")
    changes_prediction: bool = Field(description="True if setting only this feature to its median flips the class")


class Explanation(Prediction):
    method: str
    contributions: list[Contribution]


class Health(BaseModel):
    status: str = Field(description="'ok' when a model is loaded, 'degraded' when the API runs without one")
    model_loaded: bool
    model_status: str = Field(description="'loaded' or 'failed'")
    model: str | None = None
    version: str | None = None
    detail: str | None = None


class ModelInfo(BaseModel):
    model: str
    version: str
    threshold: float
    n_features: int
    n_train: int | None = None
    trained_at: str | None = None
    metrics: dict[str, float] = Field(description="Evaluation results saved with the model, not live measurements")


def _bundle() -> dict:
    if STATE["bundle"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "model not loaded")
    return STATE["bundle"]


def _frame(bundle: dict, rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[bundle["features"]]


def _predict(bundle: dict, features: IPFeatures) -> tuple[dict, float]:
    proba = float(bundle["model"].predict_proba(_frame(bundle, [features.model_dump()]))[0, 1])
    meta = bundle["metadata"]
    is_vpn_proxy = proba >= bundle["threshold"]
    return {
        "prediction": int(is_vpn_proxy),
        "label": LABEL_POSITIVE if is_vpn_proxy else LABEL_NEGATIVE,
        "is_vpn_proxy": is_vpn_proxy,
        "probability": round(proba, 4),
        "threshold": round(float(bundle["threshold"]), 4),
        "model_version": f"{meta['model_name']}-{meta['version']}",
    }, proba


@app.get("/health", response_model=Health, response_model_exclude_none=True)
def health() -> dict:
    bundle = STATE["bundle"]
    if bundle is None:
        return {"status": "degraded", "model_loaded": False, "model_status": "failed",
                "detail": STATE["error"] or "model not loaded"}
    meta = bundle["metadata"]
    return {"status": "ok", "model_loaded": True, "model_status": "loaded",
            "model": meta["model_name"], "version": meta["version"]}


@app.get("/model-info", response_model=ModelInfo, responses=MODEL_UNAVAILABLE)
def model_info() -> dict:
    """Identity and saved evaluation results of the model this API is serving."""
    bundle = _bundle()
    meta = bundle["metadata"]
    return {"model": meta["model_name"], "version": meta["version"], "threshold": round(float(bundle["threshold"]), 4),
            "n_features": len(bundle["features"]), "n_train": meta.get("n_train"), "trained_at": meta.get("trained_at"),
            "metrics": meta.get("metrics", {})}


@app.post("/predict", response_model=Prediction, responses=MODEL_UNAVAILABLE)
def predict(features: IPFeatures) -> dict:
    result, _ = _predict(_bundle(), features)
    return result


@app.post("/explain", response_model=Explanation, responses=MODEL_UNAVAILABLE)
def explain(features: IPFeatures, top_k: int = Query(8, ge=1, description="How many features to return")) -> dict:
    """Per-feature effect: replace one feature at a time with its training median and
    measure how much the VPN/proxy probability changes (positive = pushes towards VPN)."""
    bundle = _bundle()
    result, proba = _predict(bundle, features)
    row = features.model_dump()
    reference = bundle["reference"]
    threshold = bundle["threshold"]
    variants = [{**row, f: reference[f]} for f in bundle["features"]]
    baseline_probas = bundle["model"].predict_proba(_frame(bundle, variants))[:, 1]
    contributions = [
        {"feature": f, "value": row[f], "reference": float(reference[f]), "effect": round(proba - float(p), 4),
         "probability_at_reference": round(float(p), 4),
         "changes_prediction": bool((p >= threshold) != (proba >= threshold))}
        for f, p in zip(bundle["features"], baseline_probas, strict=True)
    ]
    contributions.sort(key=lambda c: abs(c["effect"]), reverse=True)
    return {**result, "method": "single-feature replacement with training median",
            "contributions": contributions[:top_k]}
