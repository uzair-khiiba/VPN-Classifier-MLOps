"""Explainable VPN & Proxy Detector: the Streamlit dashboard.

It reaches the model only through HTTP calls to the FastAPI backend (see ui/api_client.py).

Run from the project root:  streamlit run ui/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Streamlit only adds ui/ itself to the path

from ui.api_client import ApiError  # noqa: E402
from ui.state import get_client  # noqa: E402
from ui.views import detect, explain, home, performance, status  # noqa: E402

st.set_page_config(page_title="Explainable VPN & Proxy Detector", page_icon=":material/shield:", layout="wide")

pages = {
    "home": st.Page(home.render, title="Overview", icon=":material/home:", default=True),
    "detect": st.Page(detect.render, title="IP Detection", icon=":material/radar:", url_path="detection"),
    "explain": st.Page(explain.render, title="Explainability", icon=":material/psychology:", url_path="explainability"),
    "performance": st.Page(performance.render, title="Model Performance", icon=":material/monitoring:",
                           url_path="performance"),
    "status": st.Page(status.render, title="API Status", icon=":material/dns:", url_path="status"),
}
st.session_state["pages"] = pages
current = st.navigation(list(pages.values()))

client = get_client()
with st.sidebar:
    try:
        health = client.health().data
    except ApiError:
        st.markdown(":material/cloud_off: **API offline**")
    else:
        if health["model_loaded"]:
            st.markdown(f":material/check_circle: **API online**, model `{health.get('model')}`")
        else:
            st.markdown(":material/warning: **API online, no model**")
    st.caption(f"API: {client.base_url}")

current.run()
