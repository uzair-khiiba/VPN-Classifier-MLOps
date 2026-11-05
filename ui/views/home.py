"""Overview: what the project does, and what is known about the model."""

import streamlit as st

from ui import reports
from ui.api_client import ApiError, ModelUnavailableError
from ui.state import format_metric, get_client, pct, show_api_error


def _metric_cards(metrics: dict) -> None:
    cards = (("Precision", "test_precision", "Of the IPs flagged, how many are really VPN/proxy"),
             ("Recall", "test_recall", "Of the real VPN/proxy IPs, how many are caught"),
             ("F1 score", "test_f1", "Balance of precision and recall, for the VPN/proxy class"),
             ("PR-AUC", "test_pr_auc", "Ranking quality on the rare class, at any threshold"))
    for column, (title, key, help_text) in zip(st.columns(4), cards, strict=True):
        with column.container(border=True):
            st.metric(title, format_metric(metrics.get(key)), help=help_text)


def render() -> None:
    st.title("Explainable VPN & Proxy Detector")
    st.markdown("Predicts whether an attacker IP seen by CrowdSec is a **VPN or proxy**, from how it behaves "
                "across its attack signals, and shows which features drove each prediction.")

    st.subheader("How it works")
    steps = (("1. Aggregate", "Millions of attack signals are summarised into 28 features per IP."),
             ("2. Predict", "This dashboard sends those features to the FastAPI backend over HTTP."),
             ("3. Decide", "The model gives a score, and a threshold tuned for F1 turns it into a class."),
             ("4. Explain", "Each feature is reset to its training median to measure its effect."))
    for column, (title, text) in zip(st.columns(4), steps, strict=True):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)
    st.info("Score an IP three ways on the IP Detection page: look up an IP from the held-out test set by its ID, "
            "enter a new IP's raw attack events and let the API compute the features, or enter all 28 features "
            "yourself.", icon=":material/info:")

    st.subheader("Served model")
    st.caption("Saved evaluation results on the held-out test set. These are not live statistics.")
    try:
        info = get_client().model_info().data
    except ModelUnavailableError as error:
        show_api_error(error)
    except ApiError as error:
        show_api_error(error)
        meta = reports.model_meta()
        if meta:
            st.caption(f"Offline fallback: results saved in models/model_meta.json for {meta.get('model_name')}. "
                       "The API has not confirmed that it serves this model.")
            _metric_cards(meta.get("metrics", {}))
    else:
        trained_on = f"{info['n_train']:,}" if info.get("n_train") else "n/a"
        st.markdown(f"**{info['model']}**, version `{info['version']}`, decision threshold "
                    f"{format_metric(info['threshold'], 4)}, trained on {trained_on} IPs.")
        _metric_cards(info["metrics"])

    st.subheader("Dataset")
    st.caption("Fixed statistics of the CrowdSec Kaggle data, saved during data preparation. These are not live.")
    stats = reports.dataset_stats()
    if stats is None:
        st.info("Dataset statistics are unavailable, because data/processed/prepare_report.json is not in this copy. "
                "Run the data preparation step to create it.", icon=":material/info:")
        return
    cards = (("Attack signals", f"{stats['n_rows']:,}"), ("Attacker IPs", f"{stats['n_ips']:,}"),
             ("VPN/proxy IPs", pct(stats["positive_rate"])), ("Train IPs", f"{stats['n_train']:,}"),
             ("Test IPs", f"{stats['n_test']:,}"))
    for column, (title, value) in zip(st.columns(5), cards, strict=True):
        with column.container(border=True):
            st.metric(title, value)
    if stats.get("time_range"):
        st.caption(f"Signals from {stats['time_range'][0][:10]} to {stats['time_range'][1][:10]}. "
                   "Because so few IPs are VPN/proxy, accuracy is misleading, so F1 is the main metric.")
