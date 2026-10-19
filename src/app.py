"""FastAPI service: VPN/proxy prediction for one attacker IP.

Run:  uvicorn src.app:app --reload     then open http://127.0.0.1:8000/docs
"""

import json
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import MODEL_PATH, ROOT
from src.model_io import ModelLoadError, load_bundle

STATE: dict = {"bundle": None, "error": None}

# Two typical IPs from the held-out test set, shown as request examples in /docs.
EXAMPLES = [json.loads((ROOT / "examples" / f).read_text()) for f in ("vpn_ip.json", "regular_ip.json")]


def load_model(path=MODEL_PATH) -> None:
    try:
        STATE["bundle"], STATE["error"] = load_bundle(path), None
    except ModelLoadError as exc:
        STATE["bundle"], STATE["error"] = None, str(exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="VPN/Proxy IP Classifier",
    description="Predicts whether an attacker IP is a VPN/proxy from its aggregated CrowdSec signals.",
    version="1.0.0",
    lifespan=lifespan,
)

Count = Field(ge=0)
Share = Field(ge=0, le=1)
Flag = Field(ge=0, le=1)


class IPFeatures(BaseModel):
    """Aggregated behaviour of one attacker IP (see src/features.py for definitions)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": EXAMPLES})

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
    share_http_spam: float = Share
    share_http_scan: float = Share
    share_http_exploit: float = Share
    share_http_bruteforce: float = Share
    share_http_crawl: float = Share
    share_ssh_bruteforce: float = Share
    share_pop3_imap_bruteforce: float = Share
    share_tcp_scan: float = Share
    has_shodan: float = Flag
    n_tcp_ports: float = Count
    n_udp_ports: float = Count
    jarm_shared: float = Field(ge=0, description="Other IPs sharing this IP's JARM fingerprints")
    ja3s_shared: float = Field(ge=0, description="Other IPs sharing this IP's JA3S fingerprints")
    headers_hash_shared: float = Field(ge=0, description="Other IPs sharing this IP's HTTP header hashes")
    no_print_with_ports: float = Flag


class Prediction(BaseModel):
    is_vpn_proxy: bool
    probability: float
    threshold: float
    model_version: str


class Contribution(BaseModel):
    feature: str
    value: float
    reference: float
    effect: float = Field(description="Probability change caused by this feature's value vs. the training median")


class Explanation(Prediction):
    method: str
    contributions: list[Contribution]


def _bundle() -> dict:
    if STATE["bundle"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "model not loaded")
    return STATE["bundle"]


def _frame(bundle: dict, rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[bundle["features"]]


def _predict(bundle: dict, features: IPFeatures) -> tuple[dict, float]:
    proba = float(bundle["model"].predict_proba(_frame(bundle, [features.model_dump()]))[0, 1])
    meta = bundle["metadata"]
    return {
        "is_vpn_proxy": proba >= bundle["threshold"],
        "probability": round(proba, 4),
        "threshold": round(float(bundle["threshold"]), 4),
        "model_version": f"{meta['model_name']}-{meta['version']}",
    }, proba


@app.get("/health")
def health() -> dict:
    bundle = STATE["bundle"]
    if bundle is None:
        return {"status": "degraded", "model_loaded": False, "detail": STATE["error"]}
    meta = bundle["metadata"]
    return {"status": "ok", "model_loaded": True, "model": meta["model_name"], "version": meta["version"]}


@app.post("/predict", response_model=Prediction)
def predict(features: IPFeatures) -> dict:
    result, _ = _predict(_bundle(), features)
    return result


@app.post("/explain", response_model=Explanation)
def explain(features: IPFeatures, top_k: int = 8) -> dict:
    """Per-feature effect: replace one feature at a time with its training median and
    measure how much the VPN/proxy probability changes (positive = pushes towards VPN)."""
    bundle = _bundle()
    result, proba = _predict(bundle, features)
    row = features.model_dump()
    reference = bundle["reference"]
    variants = [{**row, f: reference[f]} for f in bundle["features"]]
    baseline_probas = bundle["model"].predict_proba(_frame(bundle, variants))[:, 1]
    contributions = [
        {"feature": f, "value": row[f], "reference": float(reference[f]), "effect": round(proba - float(p), 4)}
        for f, p in zip(bundle["features"], baseline_probas, strict=True)
    ]
    contributions.sort(key=lambda c: abs(c["effect"]), reverse=True)
    return {**result, "method": "single-feature replacement with training median",
            "contributions": contributions[: max(1, top_k)]}
