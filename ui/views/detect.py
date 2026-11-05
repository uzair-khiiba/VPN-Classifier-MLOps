"""IP Detection: collect the model's features, and show what /predict returns."""

import pandas as pd
import streamlit as st

from ui import reports, state
from ui.api_client import ApiError, FeatureSpec
from ui.features import (
    BINARY_FEATURES,
    FRACTION_PREFIX,
    consistency_warnings,
    group_specs,
    help_for,
    label_for,
    parse_pasted_row,
    validate_inputs,
)
from ui.shap_view import render_local_shap
from ui.state import format_metric, get_client, show_api_error

# The API lists its example records in this order: examples/vpn_ip.json, then examples/regular_ip.json.
SAMPLE_NAMES = ("Sample labelled VPN/proxy", "Sample labelled not VPN/proxy")
SAMPLE_KEY = "loaded_sample"
LOADED_VALUES = "loaded_values"
LOAD_MESSAGE = "load_message"
ID_COLUMN = "attacker_ip_enum"


def _widget_key(name: str) -> str:
    return f"feature_{name}"


def _load_sample(name: str, record: dict) -> None:
    for feature, value in record.items():
        st.session_state[_widget_key(feature)] = float(value)
    st.session_state[SAMPLE_KEY] = name
    st.session_state[LOADED_VALUES] = {feature: float(value) for feature, value in record.items()}


def _clear_form(specs: list[FeatureSpec]) -> None:
    for spec in specs:
        st.session_state[_widget_key(spec.name)] = None
    for key in (SAMPLE_KEY, LOADED_VALUES, LOAD_MESSAGE):
        st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def _export() -> pd.DataFrame | None:
    return reports.load_export()


def _load_row(row: pd.Series, specs: list[FeatureSpec]) -> None:
    name = (f"IP {row[ID_COLUMN]} from the export (split: {row['split']}, label: {row['label_name']})")
    _load_sample(name, {spec.name: float(row[spec.name]) for spec in specs})
    st.session_state[LOAD_MESSAGE] = None


def _load_by_id(table: pd.DataFrame, specs: list[FeatureSpec]) -> None:
    text = st.session_state.get("export_ip_id", "").strip()
    matches = table[table[ID_COLUMN] == int(text)] if text.isdigit() else table.iloc[0:0]
    if matches.empty:
        st.session_state[LOAD_MESSAGE] = f"No record has attacker_ip_enum {text or '(empty)'}."
    else:
        _load_row(matches.iloc[0], specs)


def _load_random(table: pd.DataFrame, specs: list[FeatureSpec]) -> None:
    pool = table[table["split"] == st.session_state.get("export_split", "test")]
    label = st.session_state.get("export_label", "Any")
    if label != "Any":
        pool = pool[pool["label_name"] == label]
    if pool.empty:
        st.session_state[LOAD_MESSAGE] = "No records match that choice."
    else:
        _load_row(pool.sample(1).iloc[0], specs)


def _load_pasted(specs: list[FeatureSpec]) -> None:
    try:
        values = parse_pasted_row(st.session_state.get("pasted_row", ""), [spec.name for spec in specs])
    except ValueError as error:
        st.session_state[LOAD_MESSAGE] = str(error)
        return
    for feature, value in values.items():
        st.session_state[_widget_key(feature)] = value
    for key in (SAMPLE_KEY, LOADED_VALUES, LOAD_MESSAGE):
        st.session_state.pop(key, None)


def _loaders(specs: list[FeatureSpec]) -> None:
    with st.expander("Load a real record from the aggregated dataset"):
        table = _export()
        if table is None:
            st.info("The export hasn't been generated yet. From the project folder, run:", icon=":material/info:")
            st.code(reports.EXPORT_COMMAND, language=None)
        else:
            st.caption(f"{len(table):,} real IPs, loaded at full precision. Prefer test records: the model never "
                       "trained on them, so their scores are realistic.")
            left, right = st.columns(2)
            left.text_input("attacker_ip_enum (the ID column in the Excel workbook)", key="export_ip_id")
            left.button("Load this IP", on_click=_load_by_id, args=(table, specs), key="load_by_id")
            right.radio("Split", ("test", "train"), horizontal=True, key="export_split")
            right.radio("Label", ("Any", "VPN/proxy", "Not VPN/proxy"), horizontal=True, key="export_label")
            right.button("Load a random record", on_click=_load_random, args=(table, specs), key="load_random")
    with st.expander("Paste a row from Excel"):
        st.caption("Copy the 28 feature cells of one row, from n_signals to no_print_with_ports, or paste a JSON "
                   "object. Excel copies the rounded values it displays, so the score can differ slightly from the "
                   "exact record.")
        st.text_area("Pasted values", key="pasted_row", height=80)
        st.button("Fill the form", on_click=_load_pasted, args=(specs,), key="load_pasted")
    if st.session_state.get(LOAD_MESSAGE):
        st.error(st.session_state[LOAD_MESSAGE], icon=":material/error:")


def _input(spec: FeatureSpec, column):
    key = _widget_key(spec.name)
    label = label_for(spec.name)
    tooltip = help_for(spec)
    if spec.name in BINARY_FEATURES:
        return column.selectbox(label, (0.0, 1.0), index=None, key=key, help=tooltip, placeholder="Choose",
                                format_func=lambda value: "Yes (1)" if value else "No (0)")
    is_fraction = spec.name.startswith(FRACTION_PREFIX)
    number_format = "%.6f" if is_fraction else "%.0f" if spec.kind == "count" else "%.3f"
    return column.number_input(label, min_value=spec.minimum, max_value=spec.maximum, value=None, key=key,
                               help=tooltip, placeholder="Required", step=0.01 if is_fraction else 1.0,
                               format=number_format)


def _show_result() -> None:
    result = st.session_state.get(state.LAST_PREDICTION)
    if not result:
        return
    st.subheader("Prediction")
    source = st.session_state.get(state.LAST_SOURCE)
    if source:
        st.warning(f"Demo record: {source}. This is a stored dataset record, scored by the real model. It is not a "
                   "detection of a live IP.", icon=":material/science:")
        if "split: train" in source:
            st.warning("The model trained on this record, so this score looks better than the model really is.",
                       icon=":material/warning:")
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
    st.markdown(f"**Model that answered:** `{result['model_version']}`")
    st.subheader("Why the model gave this score")
    explanation = st.session_state.get(state.LAST_SHAP)
    if explanation:
        render_local_shap(explanation, top_k=8)
    else:
        st.info("A SHAP explanation isn't available for this prediction. The prediction above isn't affected.",
                icon=":material/info:")
    pages = st.session_state.get("pages", {})
    if "explain" in pages:
        st.page_link(pages["explain"], label="More on the Explainability page", icon=":material/psychology:")


def render() -> None:
    st.title("IP Detection")
    st.markdown("Enter one IP's 28 aggregated features, and the real model scores them through `POST /predict`.")
    st.info("**This form expects engineered features, not an IP address.** The model can't classify an IP from its "
            "address alone, and this app doesn't look anything up. The values come from aggregating CrowdSec signals "
            "and Shodan data (see `src/features.py`), so enter them yourself or load a record. Hover over a field's "
            ":material/help: icon for its meaning and unit.", icon=":material/info:")

    try:
        specs, examples = get_client().feature_schema()
    except ApiError as error:
        show_api_error(error)
        return

    sample_columns = st.columns(3)
    for column, name, record in zip(sample_columns, SAMPLE_NAMES, examples, strict=False):
        column.button(f"Load: {name}", on_click=_load_sample, args=(name, record), key=f"load_{name}", width="stretch")
    sample_columns[2].button("Reset form", on_click=_clear_form, args=(specs,), key="clear_form", width="stretch")
    _loaders(specs)
    if st.session_state.get(SAMPLE_KEY):
        st.caption(f"Form filled from: {st.session_state[SAMPLE_KEY]}. This is a stored record, not a live IP.")

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
            for warning in consistency_warnings(values, specs):
                st.warning(warning, icon=":material/warning:")
            sample = st.session_state.get(SAMPLE_KEY)
            record = st.session_state.get(LOADED_VALUES)
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
                try:
                    st.session_state[state.LAST_SHAP] = get_client().explain_shap(values).data
                except ApiError:
                    st.session_state.pop(state.LAST_SHAP, None)  # the prediction above still stands
    _show_result()
