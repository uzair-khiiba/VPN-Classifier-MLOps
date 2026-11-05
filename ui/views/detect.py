"""IP Detection: collect the model's features, and show what /predict returns."""

import streamlit as st

from ui import state
from ui.api_client import ApiError, FeatureSpec
from ui.features import BINARY_FEATURES, FRACTION_PREFIX, consistency_warnings, group_specs, label_for, validate_inputs
from ui.state import format_metric, get_client, show_api_error

# The API lists its example records in this order: examples/vpn_ip.json, then examples/regular_ip.json.
SAMPLE_NAMES = ("Sample labelled VPN/proxy", "Sample labelled not VPN/proxy")
SAMPLE_KEY = "loaded_sample"


def _widget_key(name: str) -> str:
    return f"feature_{name}"


def _load_sample(name: str, record: dict) -> None:
    for feature, value in record.items():
        st.session_state[_widget_key(feature)] = float(value)
    st.session_state[SAMPLE_KEY] = name


def _clear_form(specs: list[FeatureSpec]) -> None:
    for spec in specs:
        st.session_state[_widget_key(spec.name)] = None
    st.session_state.pop(SAMPLE_KEY, None)


def _input(spec: FeatureSpec, column):
    key = _widget_key(spec.name)
    label = label_for(spec.name)
    if spec.name in BINARY_FEATURES:
        return column.selectbox(label, (0.0, 1.0), index=None, key=key, help=spec.description, placeholder="Choose",
                                format_func=lambda value: "Yes (1)" if value else "No (0)")
    is_fraction = spec.name.startswith(FRACTION_PREFIX)
    return column.number_input(label, min_value=spec.minimum, max_value=spec.maximum, value=None, key=key,
                               help=spec.description, placeholder="Required", step=0.01 if is_fraction else 1.0,
                               format="%.4f" if is_fraction else "%.3f")


def _show_result() -> None:
    result = st.session_state.get(state.LAST_PREDICTION)
    if not result:
        return
    st.subheader("Prediction")
    source = st.session_state.get(state.LAST_SOURCE)
    if source:
        st.warning(f"Demo record: {source}. This is a stored example from the held-out test set, scored by the real "
                   "model. It is not a detection of a live IP.", icon=":material/science:")
    columns = st.columns(3)
    with columns[0].container(border=True):
        icon = ":material/gpp_maybe:" if result["prediction"] == 1 else ":material/verified_user:"
        st.markdown(f"{icon} **Predicted class**")
        st.markdown(f"### {result['label']}")
    with columns[1].container(border=True):
        st.metric("VPN/proxy score", format_metric(result["probability"], 4))
    with columns[2].container(border=True):
        st.metric("Decision threshold", format_metric(result["threshold"], 4))
    st.progress(min(max(float(result["probability"]), 0.0), 1.0))
    st.caption(f"Flagged as VPN/proxy when the score is at least {format_metric(result['threshold'], 4)}. The model "
               "was trained with class weights, which inflate scores, so read the score against the threshold, not "
               f"as a literal chance. Model: {result['model_version']}.")
    pages = st.session_state.get("pages", {})
    if "explain" in pages:
        st.page_link(pages["explain"], label="See why on the Explainability page", icon=":material/psychology:")


def render() -> None:
    st.title("IP Detection")
    st.markdown("Enter one IP's 28 aggregated features, and the real model scores them through `POST /predict`.")
    st.info("**This app can't look up an IP address.** The features come from aggregating CrowdSec signals and "
            "Shodan data (see `src/features.py`), so you need to supply them yourself or use a sample record.",
            icon=":material/info:")

    try:
        specs, examples = get_client().feature_schema()
    except ApiError as error:
        show_api_error(error)
        return

    sample_columns = st.columns(3)
    for column, name, record in zip(sample_columns, SAMPLE_NAMES, examples, strict=False):
        column.button(f"Load: {name}", on_click=_load_sample, args=(name, record), key=f"load_{name}", width="stretch")
    sample_columns[2].button("Clear form", on_click=_clear_form, args=(specs,), key="clear_form", width="stretch")
    if st.session_state.get(SAMPLE_KEY):
        st.caption(f"Form filled from: {st.session_state[SAMPLE_KEY]} (a held-out test record, not a live IP).")

    with st.form("features"):
        values = {}
        for group, items in group_specs(specs).items():
            st.markdown(f"**{group}**")
            columns = st.columns(4)
            for index, spec in enumerate(items):
                values[spec.name] = _input(spec, columns[index % 4])
        submitted = st.form_submit_button("Predict", type="primary", key="predict")

    if submitted:
        problems = validate_inputs(values, specs)
        if problems:
            state.clear_explanation()
            st.session_state.pop(state.LAST_PREDICTION, None)
            st.error(f"{len(problems)} field(s) need attention. Nothing was sent to the API.", icon=":material/error:")
            for problem in problems:
                st.markdown(f"- {problem}")
        else:
            for warning in consistency_warnings(values):
                st.warning(warning, icon=":material/warning:")
            sample = st.session_state.get(SAMPLE_KEY)
            record = dict(zip(SAMPLE_NAMES, examples, strict=False)).get(sample)
            unchanged = record is not None and all(float(record[name]) == values[name] for name in values)
            state.clear_explanation()
            st.session_state.pop(state.LAST_PREDICTION, None)
            try:
                with st.spinner("Asking the model..."):
                    result = get_client().predict(values).data
            except ApiError as error:
                show_api_error(error)
            else:
                st.session_state[state.LAST_FEATURES] = values
                st.session_state[state.LAST_SOURCE] = sample if unchanged else None
                st.session_state[state.LAST_PREDICTION] = result
    _show_result()
