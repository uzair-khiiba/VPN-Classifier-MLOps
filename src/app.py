"""FastAPI service: VPN/proxy prediction for one attacker IP.

An IP can be scored three ways:
  GET  /predict/ip/{ip_id}   an IP from the held-out test set, looked up by its ID
  POST /predict/signals      a new IP, from its raw attack signals and Shodan ports
  POST /predict              any IP, from all 28 features entered directly

Run:  uvicorn src.app:app --reload     then open http://127.0.0.1:8000/docs
"""

import json
import logging
import math
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator, model_validator

from src.config import FEATURE_REFERENCE_PATH, ID_COLUMN, IP_LOOKUP_PATH, MODEL_PATH, ROOT, TARGET
from src.features import FEATURES
from src.model_io import ModelLoadError, load_bundle
from src.online_features import SIGNAL_COLUMNS, features_from_signals

logger = logging.getLogger("uvicorn.error")

STATE: dict = {"bundle": None, "error": None, "lookup": None, "reference": None}

LABEL_POSITIVE = "VPN/Proxy"
LABEL_NEGATIVE = "Not VPN/Proxy"
# Shown to API clients. The specific reason, which can include file paths, goes to the server log only.
MODEL_ERROR = "The model file is missing or invalid. See the server log for details."
MODEL_UNAVAILABLE = {503: {"description": "No valid model is loaded, so predictions are unavailable"}}
LOOKUP_ERROR = "The test-IP lookup table is missing or invalid. Rebuild it with `python -m src.online_features`."
REFERENCE_ERROR = "The feature reference is missing or invalid. Rebuild it with `python -m src.online_features`."
NOT_IN_TEST_SET = 404, {"description": "The IP ID is not in the held-out test set"}
DATA_UNAVAILABLE = 503, {"description": "No valid model, or the lookup data this endpoint needs is missing"}
# An IP with more signals than this has to be scored from its 28 features (the 99.9th percentile is ~52,000).
MAX_SIGNALS = 100_000

# Two typical IPs from the held-out test set, shown as request examples in /docs.
EXAMPLES = [json.loads((ROOT / "examples" / f).read_text()) for f in ("vpn_ip.json", "regular_ip.json")]
SIGNALS_EXAMPLE = json.loads((ROOT / "examples" / "signals_request.json").read_text())


def load_model(path=MODEL_PATH) -> None:
    try:
        bundle = load_bundle(path)
        if set(bundle["features"]) != set(FEATURES):
            raise ModelLoadError(f"{path} expects features that differ from the API's 28 inputs: "
                                 f"missing {sorted(set(FEATURES) - set(bundle['features']))}, "
                                 f"unexpected {sorted(set(bundle['features']) - set(FEATURES))}")
        STATE["bundle"], STATE["error"] = bundle, None
    except ModelLoadError as exc:
        logger.error("Model not loaded: %s", exc)
        STATE["bundle"], STATE["error"] = None, MODEL_ERROR


def load_ip_data(lookup_path=IP_LOOKUP_PATH, reference_path=FEATURE_REFERENCE_PATH) -> None:
    """Load the test-IP lookup table and the feature reference. Either can be missing: only the
    endpoints that need it stop working, and /predict with all 28 features keeps working."""
    try:
        lookup = pd.read_parquet(lookup_path)
        missing = {ID_COLUMN, TARGET, *FEATURES} - set(lookup.columns)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
        if lookup[ID_COLUMN].duplicated().any():
            raise ValueError("duplicate IP IDs")
        STATE["lookup"] = lookup.set_index(ID_COLUMN)
    except Exception as exc:  # noqa: BLE001 - any unreadable file means "not available"
        logger.error("IP lookup not loaded from %s: %s", lookup_path, exc)
        STATE["lookup"] = None
    try:
        reference = json.loads(reference_path.read_text())
        if not isinstance(reference.get("asn_ips"), dict) or not isinstance(reference.get("fingerprints"), dict):
            raise ValueError("unexpected structure")
        STATE["reference"] = reference
    except Exception as exc:  # noqa: BLE001
        logger.error("Feature reference not loaded from %s: %s", reference_path, exc)
        STATE["reference"] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    load_ip_data()
    yield


app = FastAPI(
    title="VPN/Proxy IP Classifier",
    description="Predicts whether an attacker IP is a VPN/proxy from its aggregated CrowdSec signals. "
                "Score an IP from the held-out test set by its ID (`GET /predict/ip/{ip_id}`), a new IP from "
                "its raw attack signals (`POST /predict/signals`), or any IP from its 28 features "
                "(`POST /predict`). IPs are anonymised IDs (`attacker_ip_enum`), not addresses like 1.2.3.4.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Report the field and reason without echoing the input: a value like 1e999 parses to
    infinity, which can't be written back as JSON and would otherwise turn this 422 into a 500."""
    errors = [{key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


# Features that count something, and the two yes/no flags.
COUNT_FEATURES = ("n_signals", "n_attack_types", "n_watchers", "n_watcher_as", "n_watcher_countries", "n_attacker_as",
                  "n_active_days", "max_signals_per_hour", "asn_unique_ips", "n_tcp_ports", "n_udp_ports",
                  "jarm_shared", "ja3s_shared", "headers_hash_shared")
FLAG_FEATURES = ("has_shodan", "no_print_with_ports")


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

    @field_validator(*COUNT_FEATURES)
    @classmethod
    def _whole_number(cls, value: float) -> float:
        if not float(value).is_integer():
            raise ValueError("must be a whole number (it is a count)")
        return value

    @field_validator(*FLAG_FEATURES)
    @classmethod
    def _flag(cls, value: float) -> float:
        if value not in (0, 1):
            raise ValueError("must be 0 or 1")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "IPFeatures":
        """Combinations the feature pipeline can never produce. Every IP in the data passes these checks."""
        n, ports = self.n_signals, self.n_tcp_ports + self.n_udp_ports
        shares = sum(getattr(self, name) for name in FEATURES if name.startswith("share_"))
        problems = [
            message for broken, message in (
                (self.has_shodan != (ports > 0), "has_shodan must be 1 exactly when n_tcp_ports + n_udp_ports > 0"),
                (self.no_print_with_ports == 1 and ports == 0, "no_print_with_ports can be 1 only with ports"),
                (self.max_signals_per_hour > n, "max_signals_per_hour cannot exceed n_signals"),
                (self.n_attack_types > n, "n_attack_types cannot exceed n_signals"),
                (self.n_watchers > n, "n_watchers cannot exceed n_signals"),
                (self.n_active_days > n, "n_active_days cannot exceed n_signals"),
                (shares > 1 + 1e-6, f"the share_* fields add up to {shares:.4f}, but cannot exceed 1"),
                (not math.isclose(self.signals_per_day, n / self.n_active_days, rel_tol=1e-3, abs_tol=1e-3),
                 f"signals_per_day must equal n_signals / n_active_days ({n / self.n_active_days:.4f})"),
                (max(self.mean_gap_s, self.median_gap_s) > self.active_hours * 3600 + 1,
                 "a gap between signals cannot be longer than active_hours"),
            ) if broken
        ]
        if problems:
            raise ValueError("inconsistent features: " + "; ".join(problems))
        return self


PortKey = Annotated[str, StringConstraints(pattern=r"^[0-9]{1,5}/(tcp|udp)$")]
# CrowdSec attack types are lowercase in the data, so "HTTP:Scan " must count as http:scan.
AttackType = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=1, max_length=64)]
# JARM and JA3S are lowercase hex in the data; an empty string means no fingerprint, as in training.
Fingerprint = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True, max_length=128)]


class Signal(BaseModel):
    """One CrowdSec attack signal, with the column names of the Kaggle train.parq."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    attack_time: datetime = Field(description="When the attack was seen. A time without a time zone is read as UTC")
    attack_type: AttackType = Field(description="CrowdSec attack type, e.g. http:scan (case and spaces ignored)")
    attacker_as_num: int | None = Field(None, ge=0, description="Attacker's autonomous system number (real ASN)")
    watcher_uuid_enum: int | str = Field(description="Any ID of the reporting watcher; only distinct IDs are counted")
    watcher_as_num: int | None = Field(None, ge=0, description="The watcher's autonomous system number")
    watcher_country: str | None = Field(None, max_length=8, description="The watcher's country code")


class ShodanPort(BaseModel):
    """What Shodan recorded on one open port. Other keys from a shodan_info cell are ignored."""

    model_config = ConfigDict(extra="ignore")

    jarm: Fingerprint | None = Field(None, description="JARM TLS fingerprint")
    ja3s: Fingerprint | None = Field(None, description="JA3S TLS fingerprint")
    headers_hash: int | str | None = Field(None, description="Shodan's hash of the HTTP response headers (an integer)")

    @field_validator("headers_hash", mode="before")
    @classmethod
    def _numeric_hash(cls, value):
        """Shodan's header hashes are integers. Text like "-12" (as a form or CSV sends it) is read as the
        integer, so "0" means no hash, exactly as the integer 0 does in training."""
        if isinstance(value, str):
            value = value.strip()
            if re.fullmatch(r"-?[0-9]+", value):
                return int(value)
            return value or None
        return value


class SignalsRequest(BaseModel):
    """A new IP's raw data. The API turns it into the 28 features exactly as the training pipeline does."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [SIGNALS_EXAMPLE]})

    signals: list[Signal] = Field(min_length=1, max_length=MAX_SIGNALS, description="The IP's attack signals")
    shodan_ports: dict[PortKey, ShodanPort] = Field(
        default_factory=dict, description="Open ports Shodan recorded, keyed like '443/tcp'. Empty if Shodan has none")


class Prediction(BaseModel):
    prediction: int = Field(description="Predicted class: 1 = VPN/proxy, 0 = not VPN/proxy")
    label: str = Field(description="Human-readable predicted class")
    is_vpn_proxy: bool
    probability: float = Field(description="Model score for the VPN/proxy class, from 0 to 1")
    threshold: float = Field(description="The prediction is VPN/proxy when probability >= threshold")
    model_version: str


class IPRecord(BaseModel):
    ip_id: int
    features: IPFeatures
    actual_label: int = Field(description="True class in the held-out test set: 1 = VPN/proxy, 0 = not")
    actual_label_name: str


class IPPrediction(Prediction):
    ip_id: int
    source: str = Field(description="Where the features came from")
    features: IPFeatures
    actual_label: int = Field(description="True class in the held-out test set: 1 = VPN/proxy, 0 = not")
    actual_label_name: str
    correct: bool = Field(description="True if the prediction matches the true class")


class SignalsPrediction(Prediction):
    source: str = Field(description="Where the features came from")
    features: IPFeatures = Field(description="The 28 features computed from the signals")


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
    ip_lookup_loaded: bool = Field(False, description="True if IPs can be looked up by ID (/predict/ip)")
    n_lookup_ips: int | None = None
    feature_reference_loaded: bool = Field(False, description="True if features can be built from signals")


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


def _lookup() -> pd.DataFrame:
    if STATE["lookup"] is None:
        raise HTTPException(status_code=503, detail=LOOKUP_ERROR)
    return STATE["lookup"]


def _reference() -> dict:
    if STATE["reference"] is None:
        raise HTTPException(status_code=503, detail=REFERENCE_ERROR)
    return STATE["reference"]


def _label_name(label: int) -> str:
    return LABEL_POSITIVE if label == 1 else LABEL_NEGATIVE


def _record(ip_id: int) -> dict:
    lookup = _lookup()
    if ip_id not in lookup.index:
        raise HTTPException(status_code=404, detail=f"IP {ip_id} is not in the held-out test set. Send its signals "
                                                    "to POST /predict/signals, or its 28 features to POST /predict.")
    row = lookup.loc[ip_id]
    label = int(row[TARGET])
    return {"ip_id": ip_id, "features": {f: float(row[f]) for f in FEATURES}, "actual_label": label,
            "actual_label_name": _label_name(label)}


def _signals_frame(request: SignalsRequest) -> pd.DataFrame:
    rows = [signal.model_dump() for signal in request.signals]
    frame = pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
    times = [s.attack_time if s.attack_time.tzinfo else s.attack_time.replace(tzinfo=UTC) for s in request.signals]
    frame["attack_time"] = pd.to_datetime(times, utc=True)
    frame["attacker_as_num"] = frame["attacker_as_num"].astype("float64")
    frame["watcher_as_num"] = frame["watcher_as_num"].astype("float64")
    frame["watcher_uuid_enum"] = frame["watcher_uuid_enum"].astype(str)
    return frame


def _features_from_request(request: SignalsRequest) -> IPFeatures:
    ports = {port: info.model_dump() for port, info in request.shodan_ports.items()}
    values = features_from_signals(_signals_frame(request), ports, _reference())
    try:
        return IPFeatures(**values)
    except ValidationError as exc:  # a safeguard: features computed from valid signals always pass
        logger.error("Features computed from signals failed validation: %s", exc)
        raise HTTPException(status_code=422, detail="The signals produce features the model cannot accept.") from exc


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
    bundle, lookup = STATE["bundle"], STATE["lookup"]
    data = {"ip_lookup_loaded": lookup is not None, "n_lookup_ips": None if lookup is None else len(lookup),
            "feature_reference_loaded": STATE["reference"] is not None}
    if bundle is None:
        return {"status": "degraded", "model_loaded": False, "model_status": "failed",
                "detail": STATE["error"] or "model not loaded", **data}
    meta = bundle["metadata"]
    return {"status": "ok", "model_loaded": True, "model_status": "loaded",
            "model": meta["model_name"], "version": meta["version"], **data}


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
    """Score an IP from all 28 features, entered directly."""
    result, _ = _predict(_bundle(), features)
    return result


@app.get("/ips/{ip_id}", response_model=IPRecord, responses=dict([NOT_IN_TEST_SET, DATA_UNAVAILABLE]))
def get_ip(ip_id: int = Path(ge=0, description="The IP's ID (attacker_ip_enum) in the dataset")) -> dict:
    """The stored features and true class of an IP from the held-out test set."""
    return _record(ip_id)


@app.get("/predict/ip/{ip_id}", response_model=IPPrediction, responses=dict([NOT_IN_TEST_SET, DATA_UNAVAILABLE]))
def predict_ip(ip_id: int = Path(ge=0, description="The IP's ID (attacker_ip_enum) in the dataset")) -> dict:
    """Score an IP from the held-out test set by its ID. Its features are fetched, not typed in."""
    bundle = _bundle()
    record = _record(ip_id)
    result, _ = _predict(bundle, IPFeatures(**record["features"]))
    return {**result, **record, "source": "held-out test set",
            "correct": result["prediction"] == record["actual_label"]}


@app.post("/features/from-signals", response_model=IPFeatures, responses=dict([DATA_UNAVAILABLE]))
def features_from_signals_endpoint(request: SignalsRequest) -> IPFeatures:
    """The 28 features the API computes from an IP's signals, without scoring them.
    Useful to inspect them, or to adjust some and send them to POST /predict."""
    return _features_from_request(request)


@app.post("/predict/signals", response_model=SignalsPrediction, responses=dict([DATA_UNAVAILABLE]))
def predict_signals(request: SignalsRequest) -> dict:
    """Score a new IP from its raw attack signals and Shodan ports. The behaviour features (volume,
    timing, attack mix, network crowding, fingerprint sharing) are computed as in training."""
    bundle = _bundle()
    features = _features_from_request(request)
    result, _ = _predict(bundle, features)
    return {**result, "source": "computed from signals", "features": features}


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
