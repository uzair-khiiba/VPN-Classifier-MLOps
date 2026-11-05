"""HTTP client for the FastAPI backend. This is the only way the dashboard reaches the model.

It has no Streamlit code, so it can be tested and reused on its own. Failures become specific
exceptions with messages that are safe to show to the user.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_S = 15.0
CONNECT_TIMEOUT_S = 1.5  # an offline API should be reported quickly: pages check it on every rerun
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
START_COMMAND = ".venv\\Scripts\\python -m uvicorn src.app:app --host 127.0.0.1 --port 8000"


class ApiError(Exception):
    """Base class for every failure talking to the backend."""


class ApiUnavailableError(ApiError):
    """The backend could not be reached, or did not answer in time."""


class ModelUnavailableError(ApiError):
    """The backend is running, but has no model loaded (HTTP 503)."""


class ApiResponseError(ApiError):
    """The backend answered with something this client cannot use."""


class ApiInvalidInputError(ApiError):
    """The backend rejected the input (HTTP 422)."""

    def __init__(self, message: str, field_errors: list[tuple[str, str]]):
        super().__init__(message)
        self.field_errors = field_errors


@dataclass(frozen=True)
class ApiResult:
    data: dict
    elapsed_ms: float


@dataclass(frozen=True)
class FeatureSpec:
    """One model input, as declared by the API's own OpenAPI schema."""

    name: str
    description: str | None
    minimum: float | None
    maximum: float | None
    kind: str | None = None
    kind_help: str | None = None
    unit: str | None = None
    meaning: str | None = None
    interpret: str | None = None


def get_api_base_url() -> str:
    """API_BASE_URL from the environment, then from the project's .env file, then the default."""
    value = os.getenv("API_BASE_URL")
    if not value and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, _, found = line.partition("=")
            if key.strip() == "API_BASE_URL":
                value = found.strip().strip("\"'")
    return (value or DEFAULT_BASE_URL).rstrip("/")


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT_S,
                 client: httpx.Client | None = None):
        self.base_url = (base_url or get_api_base_url()).rstrip("/")
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=min(timeout, CONNECT_TIMEOUT_S)))

    def _request(self, method: str, path: str, required: tuple[str, ...] = (), **kwargs) -> ApiResult:
        url = f"{self.base_url}{path}"
        started = time.perf_counter()
        try:
            response = self._client.request(method, url, **kwargs)
        except (httpx.TransportError, httpx.InvalidURL) as exc:
            raise ApiUnavailableError(f"Could not reach the API at {self.base_url} ({type(exc).__name__}).") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000

        try:
            body = response.json()
        except ValueError as exc:
            raise ApiResponseError(f"{path} returned a response that is not valid JSON "
                                   f"(HTTP {response.status_code}).") from exc
        if response.status_code == 422:
            raise self._invalid_input(body)
        if response.status_code == 503:
            detail = body.get("detail") if isinstance(body, dict) else None
            raise ModelUnavailableError(str(detail or "The API is running, but no model is loaded."))
        if response.status_code != 200:
            raise ApiResponseError(f"{path} returned HTTP {response.status_code}.")
        if not isinstance(body, dict):
            raise ApiResponseError(f"{path} returned an unexpected response format.")
        missing = [key for key in required if key not in body]
        if missing:
            raise ApiResponseError(f"{path} response is missing: {', '.join(missing)}.")
        return ApiResult(body, elapsed_ms)

    @staticmethod
    def _invalid_input(body) -> ApiInvalidInputError:
        errors = []
        details = body.get("detail") if isinstance(body, dict) else None
        for item in details if isinstance(details, list) else []:
            if isinstance(item, dict):
                location = [str(part) for part in item.get("loc", []) if part != "body"]
                errors.append((".".join(location) or "request", str(item.get("msg", "invalid value"))))
        return ApiInvalidInputError("The API rejected the input.", errors)

    def health(self) -> ApiResult:
        result = self._request("GET", "/health", required=("status", "model_loaded"))
        if not isinstance(result.data["model_loaded"], bool):
            raise ApiResponseError("/health returned an unexpected response format.")
        return result

    def model_info(self) -> ApiResult:
        return self._request("GET", "/model-info", required=("model", "version", "threshold", "metrics"))

    def predict(self, features: dict) -> ApiResult:
        required = ("prediction", "label", "probability", "threshold", "model_version")
        return self._request("POST", "/predict", required=required, json=features)

    def explain(self, features: dict, top_k: int = 8) -> ApiResult:
        required = ("label", "probability", "threshold", "method", "contributions")
        result = self._request("POST", "/explain", required=required, json=features, params={"top_k": top_k})
        if not isinstance(result.data["contributions"], list):
            raise ApiResponseError("/explain returned an unexpected response format.")
        return result

    def explain_shap(self, features: dict, top_k: int = 28) -> ApiResult:
        required = ("explainer", "method", "base_value", "base_probability", "output_value", "probability", "threshold",
                    "limitation", "contributions")
        result = self._request("POST", "/explain/shap", required=required, json=features, params={"top_k": top_k})
        if not isinstance(result.data["contributions"], list):
            raise ApiResponseError("/explain/shap returned an unexpected response format.")
        return result

    def feature_schema(self) -> tuple[list[FeatureSpec], list[dict]]:
        """The model's inputs and example records, read from the API rather than copied here."""
        spec = self._request("GET", "/openapi.json", required=("components",)).data
        try:
            schema = spec["components"]["schemas"]["IPFeatures"]
            specs = [FeatureSpec(name, field.get("description"), field.get("minimum"), field.get("maximum"),
                                 field.get("kind"), field.get("kind_help"), field.get("unit"), field.get("meaning"),
                                 field.get("interpret"))
                     for name, field in schema["properties"].items()]
        except (KeyError, AttributeError, TypeError) as exc:
            raise ApiResponseError("/openapi.json does not describe the model's input features.") from exc
        if not specs:
            raise ApiResponseError("/openapi.json does not describe the model's input features.")
        return specs, schema.get("examples", [])
