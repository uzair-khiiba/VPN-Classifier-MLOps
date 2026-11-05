"""Model Performance: saved evaluation results for the served model, with experiments for comparison."""

import altair as alt
import pandas as pd
import streamlit as st

from ui import reports
from ui.api_client import ApiError
from ui.features import label_for
from ui.state import format_metric, get_client, show_api_error

ROLE_COLORS = {"Served by the API": "#3987e5", "Experimental, not served": "#d95926",
               "Baseline candidate": "#199e70", "Experiment, not served": "#c98500"}
METRICS = (("Precision", "test_precision"), ("Recall", "test_recall"), ("F1", "test_f1"), ("PR-AUC", "test_pr_auc"),
           ("ROC-AUC", "test_roc_auc"), ("Threshold", "threshold"))


def _confusion_table(counts: dict) -> None:
    st.dataframe(pd.DataFrame({"": ["Actually VPN/proxy", "Actually not VPN/proxy"],
                               "Flagged as VPN/proxy": [f"{counts['tp']:,}", f"{counts['fp']:,}"],
                               "Not flagged": [f"{counts['fn']:,}", f"{counts['tn']:,}"]}),
                 hide_index=True, width="stretch")
    st.caption(f"{counts['tp']} caught, {counts['fn']} missed, {counts['fp']} false alarms, "
               f"{counts['tn']:,} correctly left alone.")


def _comparison_chart(rows: list[dict]) -> alt.LayerChart:
    frame = pd.DataFrame(rows)
    order = frame["model"].tolist()
    roles = [role for role in ROLE_COLORS if role in set(frame["role"])]
    base = alt.Chart(frame).encode(y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
                                   x=alt.X("test_f1:Q", title="Test F1 (VPN/proxy class)",
                                           scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=5)))
    bars = base.mark_bar(cornerRadius=4, size=18).encode(
        color=alt.Color("role:N", scale=alt.Scale(domain=roles, range=[ROLE_COLORS[role] for role in roles]),
                        legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=0)),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("role:N", title="Role"),
                 alt.Tooltip("test_f1:Q", title="Test F1", format=".4f"),
                 alt.Tooltip("test_precision:Q", title="Precision", format=".3f"),
                 alt.Tooltip("test_recall:Q", title="Recall", format=".3f"),
                 alt.Tooltip("test_pr_auc:Q", title="PR-AUC", format=".3f")])
    labels = base.mark_text(align="left", dx=4, color="#e6edf3").encode(text=alt.Text("test_f1:Q", format=".3f"))
    return (bars + labels).properties(height=alt.Step(34))


def _importance_chart(rows: list[dict]) -> alt.LayerChart:
    frame = pd.DataFrame(rows)
    frame["name"] = frame["feature"].map(label_for)
    base = alt.Chart(frame).encode(y=alt.Y("name:N", sort=frame["name"].tolist(), title=None,
                                           axis=alt.Axis(labelLimit=260)),
                                   x=alt.X("f1_drop:Q", title="Drop in test F1 when the feature is shuffled",
                                           axis=alt.Axis(tickCount=5)))
    bars = base.mark_bar(cornerRadius=4, size=16, color="#3987e5").encode(
        tooltip=[alt.Tooltip("name:N", title="Feature"), alt.Tooltip("f1_drop:Q", title="F1 drop", format=".3f"),
                 alt.Tooltip("f1_drop_std:Q", title="Std across shuffles", format=".3f")])
    labels = base.mark_text(align="left", dx=4, color="#e6edf3").encode(text=alt.Text("f1_drop:Q", format=".3f"))
    return (bars + labels).properties(height=alt.Step(28))


def _served_section() -> tuple[str | None, dict]:
    """Prefer the API's own answer, so the numbers belong to the model it is really serving."""
    try:
        info = get_client().model_info().data
        st.markdown(f"**Served by the API now:** {info['model']}, version `{info['version']}`.")
        return info["model"], info["metrics"]
    except ApiError as error:
        show_api_error(error)
    meta = reports.model_meta()
    if not meta:
        return None, {}
    st.caption(f"Showing models/model_meta.json for {meta.get('model_name')}. "
               "The API has not confirmed it serves this.")
    return meta.get("model_name"), meta.get("metrics", {})


def render() -> None:
    st.title("Model Performance")
    st.caption("Everything here is a saved evaluation result on the held-out test set, not live API activity.")

    served, metrics = _served_section()
    if metrics:
        for column, (title, key) in zip(st.columns(len(METRICS)), METRICS, strict=True):
            with column.container(border=True):
                st.metric(title, format_metric(metrics.get(key)))

    st.subheader("Confusion matrix of the served model")
    counts = reports.served_confusion_matrix(served, metrics.get("test_f1")) if served else None
    if counts:
        _confusion_table(counts)
    else:
        st.info("Unavailable. It needs reports/experiments/results/baseline.json from the same model.",
                icon=":material/info:")

    selected = reports.experimental_model()
    st.subheader("Baseline vs experimental model")
    if selected and metrics:
        st.warning(f"The experimental model `{selected['name']}` is **not served by this API**. It is shown for "
                   "comparison only, because adopting it hasn't been decided.", icon=":material/science:")
        rows = [{"Metric": title, f"Served ({served})": format_metric(metrics.get(key)),
                 "Experimental": format_metric(selected["metrics"].get(key))} for title, key in METRICS]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        experimental_counts = reports.confusion_matrix(selected["metrics"])
        if experimental_counts:
            st.markdown("**Experimental model's confusion matrix**")
            _confusion_table(experimental_counts)
    else:
        st.info("Experiment results are unavailable in this copy (reports/experiments/results/).",
                icon=":material/info:")

    st.subheader("Model comparison")
    rows = reports.comparison_rows(served)
    if rows:
        st.altair_chart(_comparison_chart(rows), width="stretch")
        st.caption("Test scores in this chart are for learning. Models should be picked on cross-validation, "
                   "not by the highest test score.")
        with st.expander("Show as a table"):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("Unavailable. It needs models/model_meta.json.", icon=":material/info:")

    st.subheader("Feature importance")
    importance = reports.feature_importance()
    if importance and importance.get("importance"):
        st.caption(f"Permutation importance of {importance.get('model')} on {importance.get('n_test'):,} test IPs. "
                   "It measures how much this model relies on a feature, not whether the feature causes the label.")
        st.altair_chart(_importance_chart(importance["importance"][:12]), width="stretch")
        groups = reports.group_importance()
        if groups:
            st.markdown("**Shuffling whole feature groups** (related features can't cover for each other)")
            frame = pd.DataFrame(groups).rename(columns={"group": "Group", "n_features": "Features",
                                                         "f1_drop": "F1 drop", "f1_drop_std": "Std"})
            st.dataframe(frame, hide_index=True, width="stretch")
    else:
        st.info("Unavailable. It needs models/feature_importance.json.", icon=":material/info:")

    st.subheader("Precision-recall curves")
    curve = reports.figure("pr_curves.png")
    if curve:
        st.image(str(curve), caption="Out-of-fold on training IPs, from the experiment reports. Baseline vs 400 trees.")
    else:
        st.info("Unavailable. It needs reports/experiments/figures/pr_curves.png.", icon=":material/info:")
