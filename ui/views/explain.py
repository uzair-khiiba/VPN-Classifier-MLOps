"""Explainability: show what /explain returns for the last predicted record."""

import altair as alt
import pandas as pd
import streamlit as st

from ui import state
from ui.api_client import ApiError
from ui.features import label_for
from ui.state import format_metric, get_client, show_api_error

TOWARD, AWAY = "Pushes toward VPN/proxy", "Pushes away from VPN/proxy"
COLORS = {TOWARD: "#d95926", AWAY: "#3987e5"}


def _frame(contributions: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(contributions)
    frame["name"] = frame["feature"].map(label_for)
    frame["direction"] = [TOWARD if effect >= 0 else AWAY for effect in frame["effect"]]
    return frame


def effect_chart(frame: pd.DataFrame) -> alt.LayerChart:
    order = frame["name"].tolist()
    base = alt.Chart(frame).encode(y=alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)))
    bars = base.mark_bar(cornerRadius=4, size=16).encode(
        x=alt.X("effect:Q", title="Change in score compared with the training median", axis=alt.Axis(tickCount=5)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                        legend=alt.Legend(title=None, orient="top", labelLimit=0)),
        tooltip=[alt.Tooltip("name:N", title="Feature"), alt.Tooltip("value:Q", title="This IP"),
                 alt.Tooltip("reference:Q", title="Training median"), alt.Tooltip("effect:Q", title="Effect"),
                 alt.Tooltip("probability_at_reference:Q", title="Score at median"),
                 alt.Tooltip("changes_prediction:N", title="Flips the class")])
    labels = base.transform_filter("datum.effect >= 0").mark_text(align="left", dx=4, color="#e6edf3").encode(
        x="effect:Q", text=alt.Text("effect:Q", format="+.3f"))
    negative = base.transform_filter("datum.effect < 0").mark_text(align="right", dx=-4, color="#e6edf3").encode(
        x="effect:Q", text=alt.Text("effect:Q", format="+.3f"))
    return (bars + labels + negative).properties(height=alt.Step(30))


def _flip_summary(frame: pd.DataFrame, label: str) -> None:
    flips = frame[frame["changes_prediction"]]
    if flips.empty:
        st.info(f"No single feature shown here flips the class. Setting any one of them to its median alone would "
                f"leave the prediction at **{label}**.", icon=":material/info:")
    else:
        names = ", ".join(f"**{name}**" for name in flips["name"])
        st.warning(f"Setting one of these alone to its training median would flip the class from **{label}**: {names}.",
                   icon=":material/swap_horiz:")


def render() -> None:
    st.title("Explainability")
    st.markdown("Shows which features moved this prediction, using the model's own `POST /explain` response.")
    with st.expander("How this explanation works, and its limits", expanded=True):
        st.markdown(
            "**Method:** for each feature, the API replaces only that feature with its median from the training data, "
            "re-scores the IP, and reports the change. A positive effect means this IP's value pushes the score toward "
            "VPN/proxy.\n\n"
            "**Limits:**\n"
            "- It describes how this model behaves, not what makes an IP a VPN. **It doesn't show causes.**\n"
            "- It is **not SHAP**. Effects are measured one at a time, and needn't add up to the score.\n"
            "- Features working together can be missed, and related features can cover for each other.\n"
            "- The median can create impossible combinations, such as no Shodan record with open ports.")

    features = st.session_state.get(state.LAST_FEATURES)
    prediction = st.session_state.get(state.LAST_PREDICTION)
    if not features or not prediction:
        st.info("Nothing to explain yet. Run a prediction on the IP Detection page first.", icon=":material/info:")
        pages = st.session_state.get("pages", {})
        if "detect" in pages:
            st.page_link(pages["detect"], label="Go to IP Detection", icon=":material/radar:")
        return

    source = st.session_state.get(state.LAST_SOURCE)
    st.markdown(f"Record from your last prediction: **{prediction['label']}**, score "
                f"{format_metric(prediction['probability'], 4)}, threshold {format_metric(prediction['threshold'], 4)}."
                + (f" Demo record: {source}." if source else ""))
    top_k = st.slider("Features to show", 1, len(features), 8, key="top_k")
    if st.button("Explain this prediction", type="primary", key="explain"):
        try:
            with st.spinner("Asking the model..."):
                st.session_state[state.LAST_EXPLANATION] = get_client().explain(features, top_k).data
        except ApiError as error:
            state.clear_explanation()
            show_api_error(error)

    explanation = st.session_state.get(state.LAST_EXPLANATION)
    if not explanation:
        return
    if not explanation["contributions"]:
        st.warning("The API returned no feature effects.", icon=":material/warning:")
        return
    frame = _frame(explanation["contributions"])
    st.subheader("Most influential features")
    st.caption(f"Method reported by the API: {explanation['method']}. "
               f"Model: {explanation.get('model_version', 'n/a')}.")
    st.altair_chart(effect_chart(frame), width="stretch")
    _flip_summary(frame, explanation["label"])
    st.markdown("**The same results as a table**")
    table = frame[["name", "value", "reference", "effect", "probability_at_reference", "changes_prediction"]].rename(
        columns={"name": "Feature", "value": "This IP", "reference": "Training median", "effect": "Effect",
                 "probability_at_reference": "Score at median", "changes_prediction": "Flips the class"})
    st.dataframe(table, hide_index=True, width="stretch")
