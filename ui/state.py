"""Session state and display helpers shared by the pages."""

import streamlit as st

from ui.api_client import (
    START_COMMAND,
    ApiClient,
    ApiInvalidInputError,
    ApiUnavailableError,
    ModelUnavailableError,
    get_api_base_url,
)
from ui.features import label_for

# The last request sent to /predict, kept so the Explainability page explains that exact record.
LAST_FEATURES = "last_features"
LAST_SOURCE = "last_source"
LAST_PREDICTION = "last_prediction"
LAST_EXPLANATION = "last_explanation"
LAST_SHAP = "last_shap"


@st.cache_resource(show_spinner=False)
def _client_for(base_url: str) -> ApiClient:
    return ApiClient(base_url)  # creating an HTTP client is slow, so every rerun shares one


def get_client() -> ApiClient:
    return _client_for(get_api_base_url())


def clear_explanation() -> None:
    st.session_state.pop(LAST_EXPLANATION, None)
    st.session_state.pop(LAST_SHAP, None)


def show_api_error(error: Exception) -> None:
    """Explain a failed request. Nothing is shown as a result, so a failure can't look like a prediction."""
    if isinstance(error, ApiUnavailableError):
        st.error(f"The API is offline. {error}", icon=":material/cloud_off:")
        st.markdown("Start the backend in a separate terminal, from the project folder:")
        st.code(START_COMMAND, language=None)
    elif isinstance(error, ModelUnavailableError):
        st.warning(f"The API is running, but cannot predict. {error}", icon=":material/warning:")
    elif isinstance(error, ApiInvalidInputError):
        st.error("The API rejected the input.", icon=":material/error:")
        for field, message in error.field_errors:
            st.markdown(f"- **{label_for(field)}** (`{field}`): {message}")
    else:
        st.error(f"The API returned an unusable response. {error}", icon=":material/error:")


def format_metric(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def pct(value) -> str:
    return "n/a" if value is None else f"{value:.2%}"
