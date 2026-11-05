"""IP Detection: score an IP by looking up its ID, from its raw attack signals, or from all 28 features."""

import math
import re

import pandas as pd
import streamlit as st

from ui import state
from ui.api_client import ApiError, FeatureSpec
from ui.features import BINARY_FEATURES, FRACTION_PREFIX, consistency_warnings, group_specs, label_for, validate_inputs
from ui.state import format_metric, get_client, show_api_error

# The API lists its example records in this order: examples/vpn_ip.json, then examples/regular_ip.json.
SAMPLE_NAMES = ("Sample labelled VPN/proxy", "Sample labelled not VPN/proxy")
SAMPLE_KEY = "loaded_sample"

# Two IPs from the held-out test set that the model classifies correctly, offered as quick examples.
EXAMPLE_IPS = ((47584, "a VPN/proxy"), (6, "not a VPN/proxy"))
IP_KEY = "ip_id"

# The raw signal columns the API accepts (names from the Kaggle train.parq). Kept here, not imported from
# src/, because the dashboard talks to the backend only over HTTP.
SIGNAL_COLUMNS = ["attack_time", "attack_type", "attacker_as_num", "watcher_uuid_enum", "watcher_as_num",
                  "watcher_country"]
REQUIRED_SIGNAL_COLUMNS = ("attack_time", "attack_type", "watcher_uuid_enum")
PORT_COLUMNS = ["port", "jarm", "ja3s", "headers_hash"]
SIGNALS_DATA, PORTS_DATA, EDITOR_VERSION, UPLOADED_FILE = "signals_data", "ports_data", "editor_version", "signals_csv"


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


def _store_result(result: dict, features: dict, source: str | None, actual: dict | None = None) -> None:
    st.session_state[state.LAST_FEATURES] = features
    st.session_state[state.LAST_SOURCE] = source
    st.session_state[state.LAST_PREDICTION] = result
    st.session_state[state.LAST_ACTUAL] = actual


def _show_result() -> None:
    result = st.session_state.get(state.LAST_PREDICTION)
    if not result:
        return
    st.subheader("Prediction")
    source = st.session_state.get(state.LAST_SOURCE)
    if source:
        st.warning(f"Demo record: {source}. This is a stored example from the held-out test set, scored by the real "
                   "model. It is not a detection of a live IP.", icon=":material/science:")
    actual = st.session_state.get(state.LAST_ACTUAL)
    columns = st.columns(4 if actual else 3)
    with columns[0].container(border=True):
        icon = ":material/gpp_maybe:" if result["prediction"] == 1 else ":material/verified_user:"
        st.markdown(f"{icon} **Predicted class**")
        st.markdown(f"### {result['label']}")
    with columns[1].container(border=True):
        st.metric("VPN/proxy score", format_metric(result["probability"], 4))
    with columns[2].container(border=True):
        st.metric("Decision threshold", format_metric(result["threshold"], 4))
    if actual:
        with columns[3].container(border=True):
            verdict = "the model is right" if actual["correct"] else "the model is wrong"
            st.metric("Actual class (test set)", actual["label"], help=f"The true label of this IP: {verdict}.")
            st.caption(":material/check_circle: Correct" if actual["correct"] else ":material/cancel: Wrong")
    st.progress(min(max(float(result["probability"]), 0.0), 1.0))
    st.caption(f"Flagged as VPN/proxy when the score is at least {format_metric(result['threshold'], 4)}. The model "
               "was trained with class weights, which inflate scores, so read the score against the threshold, not "
               f"as a literal chance. Model: {result['model_version']}.")
    features = st.session_state.get(state.LAST_FEATURES)
    if features and "features" in result:
        with st.expander("The 28 features the API used"):
            st.dataframe(pd.DataFrame({"Feature": [label_for(name) for name in features], "Name": list(features),
                                       "Value": list(features.values())}), hide_index=True, width="stretch")
    pages = st.session_state.get("pages", {})
    if "explain" in pages:
        st.page_link(pages["explain"], label="See why on the Explainability page", icon=":material/psychology:")


# ---- Tab 1: look up an IP by ID ----------------------------------------------------------------


def _set_ip(ip_id: int) -> None:
    st.session_state[IP_KEY] = ip_id


def _lookup_tab() -> None:
    st.markdown("Enter the ID of an IP from the **held-out test set** (29,652 IPs the model never trained on). The "
                "API fetches its stored features, scores them, and shows the IP's true class next to the prediction.")
    st.caption("The dataset anonymises IPs, so an IP is identified by its ID (`attacker_ip_enum`), not by an "
               "address like 1.2.3.4. For an IP that isn't in the test set, use **From attack events**.")
    columns = st.columns([2, 1, 1])
    ip_id = columns[0].number_input("IP ID", min_value=0, step=1, value=None, key=IP_KEY, placeholder="e.g. 47584",
                                    format="%d")
    for column, (example, kind) in zip(columns[1:], EXAMPLE_IPS, strict=True):
        column.button(f"Try {example} ({kind})", on_click=_set_ip, args=(example,), key=f"example_ip_{example}",
                      width="stretch")
    if not st.button("Look up and predict", type="primary", key="predict_ip"):
        return
    if ip_id is None:
        state.clear_result()
        st.error("Enter an IP ID first. Nothing was sent to the API.", icon=":material/error:")
        return
    state.clear_result()
    try:
        with st.spinner("Looking up the IP and asking the model..."):
            result = get_client().predict_ip(int(ip_id)).data
    except ApiError as error:
        show_api_error(error)
        return
    actual = {"label": result.get("actual_label_name", str(result["actual_label"])),
              "correct": result["prediction"] == result["actual_label"]}
    _store_result(result, result["features"], f"held-out test IP {int(ip_id)}", actual)


# ---- Tab 2: from raw attack events -------------------------------------------------------------


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame({"attack_time": pd.Series(dtype="str"), "attack_type": pd.Series(dtype="str"),
                         "attacker_as_num": pd.Series(dtype="float"), "watcher_uuid_enum": pd.Series(dtype="str"),
                         "watcher_as_num": pd.Series(dtype="float"), "watcher_country": pd.Series(dtype="str")})


def _empty_ports() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="str") for column in PORT_COLUMNS})


def parse_asns(values: pd.Series) -> tuple[pd.Series, int]:
    """ASNs as numbers. "AS16509" and "as16509" become 16509. Returns how many non-empty values were unreadable."""
    text = values.map(lambda value: None if _blank(value) else re.sub(r"^\s*as\s*", "", str(value), flags=re.I))
    numbers = pd.to_numeric(text, errors="coerce")
    return numbers, int((text.notna() & numbers.isna()).sum())


def normalise_signals(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Coerce a signals table (from the example or a CSV) to the editor's column types.
    Also returns how many ASN values could not be read as numbers."""
    frame = frame.reindex(columns=SIGNAL_COLUMNS)
    for column in ("attack_time", "attack_type", "watcher_uuid_enum", "watcher_country"):
        frame[column] = frame[column].map(lambda value: None if _blank(value) else str(value))
    unreadable = 0
    for column in ("attacker_as_num", "watcher_as_num"):
        frame[column], bad = parse_asns(frame[column])
        unreadable += bad
    return frame.reset_index(drop=True), unreadable


def _replace_tables(signals: pd.DataFrame, ports: pd.DataFrame | None = None) -> None:
    st.session_state[SIGNALS_DATA], unreadable = normalise_signals(signals)
    if unreadable:
        st.session_state["unreadable_asns"] = unreadable
    if ports is not None:
        # Every port column is edited as text: headers hashes are integers, but are treated as identifiers.
        st.session_state[PORTS_DATA] = ports.reindex(columns=PORT_COLUMNS).map(
            lambda value: None if _blank(value) else str(value))
    st.session_state[EDITOR_VERSION] = st.session_state.get(EDITOR_VERSION, 0) + 1  # new editors show the new data


def _load_signals_example() -> None:
    try:
        example = get_client().signals_example()
    except ApiError:
        example = None
    if not example:
        st.session_state["signals_example_missing"] = True
        return
    ports = pd.DataFrame([{"port": port, **{key: info.get(key) for key in PORT_COLUMNS[1:]}}
                          for port, info in example.get("shodan_ports", {}).items()], columns=PORT_COLUMNS)
    _replace_tables(pd.DataFrame(example["signals"]), ports)


def _clear_signals() -> None:
    _replace_tables(_empty_signals(), _empty_ports())


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or (isinstance(value, str)
                                                                                and not value.strip())


def signals_payload(signals: pd.DataFrame, ports: pd.DataFrame) -> tuple[list[dict], dict, list[str]]:
    """The request body for POST /predict/signals, and the problems that stop it being sent."""
    rows, problems = [], []
    for number, record in enumerate(signals.to_dict("records"), start=1):
        if all(_blank(value) for value in record.values()):
            continue  # the editor's empty trailing rows
        missing = [column for column in REQUIRED_SIGNAL_COLUMNS if _blank(record.get(column))]
        if missing:
            problems.append(f"Signal row {number} is missing: {', '.join(missing)}.")
            continue
        row = {"attack_time": str(record["attack_time"]).strip(), "attack_type": str(record["attack_type"]).strip(),
               "watcher_uuid_enum": str(record["watcher_uuid_enum"]).strip()}
        for column in ("attacker_as_num", "watcher_as_num"):
            value = record.get(column)
            if not _blank(value):
                row[column] = int(float(value))
        if not _blank(record.get("watcher_country")):
            row["watcher_country"] = str(record["watcher_country"]).strip()
        rows.append(row)
    if not rows and not problems:
        problems.append("Add at least one attack signal.")

    shodan_ports = {}
    for number, record in enumerate(ports.to_dict("records"), start=1):
        if all(_blank(value) for value in record.values()):
            continue
        port = "" if _blank(record.get("port")) else str(record["port"]).strip().lower()
        if not port:
            problems.append(f"Shodan port row {number} needs a port, like 443/tcp.")
            continue
        shodan_ports[port] = {key: None if _blank(record.get(key)) else str(record[key]).strip()
                              for key in PORT_COLUMNS[1:]}
    return rows, shodan_ports, problems


def _read_uploaded_csv(upload) -> None:
    if upload is None or st.session_state.get(UPLOADED_FILE) == upload.file_id:
        return
    st.session_state[UPLOADED_FILE] = upload.file_id
    try:
        frame = pd.read_csv(upload)
    except (ValueError, UnicodeDecodeError) as error:
        st.error(f"Couldn't read that CSV ({type(error).__name__}).", icon=":material/error:")
        return
    missing = [column for column in REQUIRED_SIGNAL_COLUMNS if column not in frame.columns]
    if missing:
        st.error(f"The CSV needs the columns {', '.join(REQUIRED_SIGNAL_COLUMNS)}; missing: {', '.join(missing)}.",
                 icon=":material/error:")
        return
    _replace_tables(frame)


def _signals_tab() -> None:
    st.markdown("For an IP that isn't in the test set, enter **what CrowdSec saw**: one row per attack signal, plus "
                "the open ports Shodan recorded. The API computes the 28 behaviour features exactly as the training "
                "pipeline does, including how crowded the IP's network is and how many IPs share its fingerprints.")
    columns = st.columns(3)
    columns[0].button("Load example events", on_click=_load_signals_example, key="load_signals_example",
                      width="stretch")
    columns[1].button("Clear tables", on_click=_clear_signals, key="clear_signals", width="stretch")
    with columns[2].popover("Upload a CSV", width="stretch"):
        st.caption(f"Columns: {', '.join(SIGNAL_COLUMNS)}. Extra columns are ignored, so a filtered train.parq "
                   "export works.")
        _read_uploaded_csv(st.file_uploader("Signals CSV", type="csv", key="signals_upload"))
    if st.session_state.pop("signals_example_missing", False):
        st.warning("The API offers no example request.", icon=":material/warning:")
    unreadable = st.session_state.pop("unreadable_asns", 0)
    if unreadable:
        st.warning(f"{unreadable} ASN value(s) in the CSV are not numbers (like 16509 or AS16509) and were left "
                   "empty. Fix them in the table: a missing attacker ASN makes the network crowding feature 0.",
                   icon=":material/warning:")

    version = st.session_state.get(EDITOR_VERSION, 0)
    st.markdown("**Attack signals**")
    signals = st.data_editor(
        st.session_state.get(SIGNALS_DATA, _empty_signals()), num_rows="dynamic", width="stretch",
        key=f"signals_editor_{version}",
        column_config={
            "attack_time": st.column_config.TextColumn("Time (UTC)", help="e.g. 2023-09-01 10:00:00", required=True),
            "attack_type": st.column_config.TextColumn(
                "Attack type", required=True,
                help="CrowdSec type, e.g. http:scan, http:exploit, ssh:bruteforce, tcp:scan"),
            "attacker_as_num": st.column_config.NumberColumn("Attacker ASN", min_value=0, step=1, format="%d",
                                                             help="The attacking IP's network, e.g. 16509"),
            "watcher_uuid_enum": st.column_config.TextColumn("Watcher ID", required=True,
                                                             help="Any ID per watcher: only distinct IDs are counted"),
            "watcher_as_num": st.column_config.NumberColumn("Watcher ASN", min_value=0, step=1, format="%d"),
            "watcher_country": st.column_config.TextColumn("Watcher country", help="e.g. FR"),
        })
    st.markdown("**Shodan open ports** (leave empty if Shodan has no record of the IP)")
    ports = st.data_editor(
        st.session_state.get(PORTS_DATA, _empty_ports()), num_rows="dynamic", width="stretch",
        key=f"ports_editor_{version}",
        column_config={
            "port": st.column_config.TextColumn("Port", help="Port and protocol, e.g. 443/tcp", required=True),
            "jarm": st.column_config.TextColumn("JARM", help="JARM TLS fingerprint, if Shodan has one"),
            "ja3s": st.column_config.TextColumn("JA3S", help="JA3S TLS fingerprint, if Shodan has one"),
            "headers_hash": st.column_config.TextColumn("Headers hash", help="Shodan's HTTP headers hash"),
        })

    if not st.button("Compute features and predict", type="primary", key="predict_signals"):
        return
    rows, shodan_ports, problems = signals_payload(signals, ports)
    state.clear_result()
    if problems:
        st.error(f"{len(problems)} problem(s) to fix. Nothing was sent to the API.", icon=":material/error:")
        for problem in problems:
            st.markdown(f"- {problem}")
        return
    try:
        with st.spinner("Computing the features and asking the model..."):
            result = get_client().predict_signals(rows, shodan_ports).data
    except ApiError as error:
        show_api_error(error)
        return
    _store_result(result, result["features"], None)


# ---- Tab 3: all 28 features --------------------------------------------------------------------


def render() -> None:
    st.title("IP Detection")
    st.markdown("Score one IP with the real model, in whichever way fits what you know about it.")
    lookup, signals, manual = st.tabs(["Look up an IP ID", "From attack events", "Enter all 28 features"])
    with lookup:
        _lookup_tab()
    with signals:
        _signals_tab()
    with manual:
        _manual_tab()
    _show_result()


def _manual_tab() -> None:
    st.markdown("Enter one IP's 28 aggregated features yourself, and the model scores them through `POST /predict`.")
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
        problems += consistency_warnings(values) if not problems else []
        if problems:
            state.clear_result()
            st.error(f"{len(problems)} field(s) need attention. Nothing was sent to the API.", icon=":material/error:")
            for problem in problems:
                st.markdown(f"- {problem}")
        else:
            sample = st.session_state.get(SAMPLE_KEY)
            record = dict(zip(SAMPLE_NAMES, examples, strict=False)).get(sample)
            unchanged = record is not None and all(float(record[name]) == values[name] for name in values)
            state.clear_result()
            try:
                with st.spinner("Asking the model..."):
                    result = get_client().predict(values).data
            except ApiError as error:
                show_api_error(error)
            else:
                _store_result(result, values, sample if unchanged else None)
