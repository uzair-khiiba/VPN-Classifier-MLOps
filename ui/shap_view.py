"""Show SHAP results in plain language. It only displays what the API and saved reports provide."""

import altair as alt
import pandas as pd
import streamlit as st

from ui import theme
from ui.features import label_for
from ui.state import format_metric

TOWARD, AWAY = "Pushes toward VPN/proxy", "Pushes away from VPN/proxy"
COLORS = {TOWARD: theme.TOWARD_VPN, AWAY: theme.AWAY_FROM_VPN}
CAUSATION_NOTE = ("**This describes the model, not the real world.** A feature with a large push is one this model "
                  "relied on for this score. It doesn't show that the feature causes an IP to be a VPN or proxy.")


def _frame(contributions: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(contributions)
    frame["name"] = frame["feature"].map(label_for)
    frame["side"] = [TOWARD if value > 0 else AWAY for value in frame["shap_value"]]
    return frame


def _chart(frame: pd.DataFrame) -> alt.LayerChart:
    order = frame["name"].tolist()
    base = alt.Chart(frame).encode(y=alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
                                   x=alt.X("shap_value:Q", title="Push on the score (SHAP value, log-odds)",
                                           axis=alt.Axis(tickCount=5)))
    bars = base.mark_bar(cornerRadius=4, size=16).encode(
        color=alt.Color("side:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                        legend=alt.Legend(title=None, orient="top", labelLimit=0)),
        tooltip=[alt.Tooltip("name:N", title="Feature"), alt.Tooltip("value:Q", title="This IP"),
                 alt.Tooltip("shap_value:Q", title="SHAP value", format="+.3f")])
    right = base.transform_filter("datum.shap_value >= 0").mark_text(align="left", dx=5, color=theme.INK).encode(
        text=alt.Text("shap_value:Q", format="+.2f"))
    left = base.transform_filter("datum.shap_value < 0").mark_text(align="right", dx=-5, color=theme.INK).encode(
        text=alt.Text("shap_value:Q", format="+.2f"))
    return (bars + right + left).properties(height=alt.Step(28))


def _sentence(frame: pd.DataFrame, side: str, count: int = 3) -> str:
    rows = frame[frame["side"] == side].head(count)
    return ", ".join(f"**{row['name']}** ({row['value']:g})" for _, row in rows.iterrows()) or "none"


def render_local_shap(explanation: dict, top_k: int = 8) -> None:
    frame = _frame([c for c in explanation["contributions"] if c["shap_value"] != 0])
    if frame.empty:
        st.info("No feature moved this score away from the average.", icon=":material/info:")
        return
    start, end = explanation["base_probability"], explanation["probability"]
    st.markdown(f"The model starts every IP at an average score of **{format_metric(start, 4)}**. This IP's 28 values "
                f"move it to **{format_metric(end, 4)}**, and it's flagged at "
                f"{format_metric(explanation['threshold'], 4)} or above.")
    st.markdown(f"- Pushing **toward** VPN/proxy most: {_sentence(frame, TOWARD)}\n"
                f"- Pushing **away** from VPN/proxy most: {_sentence(frame, AWAY)}")
    st.altair_chart(theme.style(_chart(frame.head(top_k))), width="stretch", theme=None)
    st.info(CAUSATION_NOTE, icon=":material/info:")
    table = frame.head(top_k)[["name", "feature", "value", "shap_value", "side"]].rename(
        columns={"name": "Feature", "feature": "API field", "value": "This IP", "shap_value": "SHAP value",
                 "side": "Effect"})
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(f"Explainer that ran: {explanation['explainer']} ({explanation['method']}). SHAP values are in "
               "log-odds, the units the model adds up internally, so they don't convert directly into score points. "
               f"They add up exactly: {explanation['base_value']:.3f} + all values = "
               f"{explanation['output_value']:.3f}.")


def render_global_shap(report: dict | None) -> None:
    st.subheader("What the model relies on overall")
    if not report:
        st.info("No SHAP report exists for the served model. To create one, run this from the project folder: "
                "`.venv\\Scripts\\python -m src.shap_report`", icon=":material/info:")
        return
    st.caption(f"{report['explainer']} on {report['rows_explained']:,} held-out test IPs, for {report['model']}. "
               "Permutation importance on the Model Performance page is a different measure: it shows how much F1 "
               "drops when a feature is shuffled.")
    left, right = st.columns(2)
    left.image(str(report["bar_plot"]), caption="Average size of each feature's effect")
    right.image(str(report["summary_plot"]), caption="Each dot is one IP. Right of 0 pushes toward VPN/proxy; "
                                                     "red means a high feature value.")
    table = pd.DataFrame(report["ranking"]).rename(columns={"feature": "API field", "mean_abs_shap": "Mean |SHAP|",
                                                            "mean_shap": "Mean SHAP"})
    st.dataframe(table, hide_index=True, width="stretch")
    st.info(CAUSATION_NOTE, icon=":material/info:")
