"""API Status: a real GET /health on every visit and refresh."""

from datetime import datetime

import streamlit as st

from ui.api_client import ApiError
from ui.state import get_client, show_api_error


def render() -> None:
    st.title("API Status")
    client = get_client()
    st.markdown(f"Checking `GET {client.base_url}/health`.")
    st.button("Check now", type="primary", key="check_now")  # any click reruns the page, which checks again

    checked_at = datetime.now().strftime("%H:%M:%S")
    try:
        result = client.health()
    except ApiError as error:
        columns = st.columns(2)
        with columns[0].container(border=True):
            st.metric("API", "Offline")
        with columns[1].container(border=True):
            st.metric("Model", "Unknown")
        show_api_error(error)
        st.caption(f"Checked at {checked_at}.")
        return

    data = result.data
    columns = st.columns(4)
    with columns[0].container(border=True):
        st.metric("API", "Online")
    with columns[1].container(border=True):
        st.metric("Model", "Loaded" if data["model_loaded"] else "Not loaded")
    with columns[2].container(border=True):
        st.metric("Model identifier", data.get("model") or "None")
    with columns[3].container(border=True):
        st.metric("Response time", f"{result.elapsed_ms:.0f} ms", help="Measured by this dashboard, not the API")

    if data["model_loaded"]:
        st.success(f"The API responded and the model is loaded. Version: {data.get('version')}.",
                   icon=":material/check_circle:")
    else:
        st.warning(f"The API responded, but can't predict. {data.get('detail') or 'No model is loaded.'}",
                   icon=":material/warning:")
    st.caption(f"Checked at {checked_at}.")
    with st.expander("Raw /health response"):
        st.json(data)
