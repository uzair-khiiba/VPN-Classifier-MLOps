"""Presentation details for the model's inputs. The inputs themselves come from the API's schema.

This file only adds friendly labels and groups. A feature the API declares but this file does not
know still appears in the form, under "Other", so the form can never hide a required input.
"""

import json
import re

from ui.api_client import FeatureSpec

GROUPS = {
    "Volume and timing": {
        "n_signals": "Attack signals", "active_hours": "Active hours (first to last signal)",
        "n_active_days": "Active days", "signals_per_day": "Signals per active day",
        "max_signals_per_hour": "Busiest hour (signals)", "mean_gap_s": "Mean gap between signals (s)",
        "median_gap_s": "Median gap between signals (s)",
    },
    "Breadth": {
        "n_attack_types": "Attack types", "n_watchers": "Reporting watchers", "n_watcher_as": "Watcher networks (ASNs)",
        "n_watcher_countries": "Watcher countries", "n_attacker_as": "Attacker networks (ASNs)",
    },
    "Attack mix (fractions from 0 to 1)": {
        "share_http_spam": "HTTP spam", "share_http_scan": "HTTP scan", "share_http_exploit": "HTTP exploit",
        "share_http_bruteforce": "HTTP brute force", "share_http_crawl": "HTTP crawl",
        "share_ssh_bruteforce": "SSH brute force", "share_pop3_imap_bruteforce": "POP3/IMAP brute force",
        "share_tcp_scan": "TCP scan",
    },
    "Network crowding": {"asn_unique_ips": "Attacker IPs in this network"},
    "Shodan (must be supplied, the app does not query Shodan)": {
        "has_shodan": "Seen by Shodan", "n_tcp_ports": "Open TCP ports", "n_udp_ports": "Open UDP ports",
        "jarm_shared": "IPs sharing JARM fingerprints", "ja3s_shared": "IPs sharing JA3S fingerprints",
        "headers_hash_shared": "IPs sharing HTTP header hash", "no_print_with_ports": "Ports but no fingerprints",
    },
}
LABELS = {name: label for group in GROUPS.values() for name, label in group.items()}
BINARY_FEATURES = {"has_shodan", "no_print_with_ports"}
FRACTION_PREFIX = "share_"


def label_for(name: str) -> str:
    return LABELS.get(name, name)


def help_for(spec: FeatureSpec) -> str:
    """The tooltip for one input: what it means, its unit, the kind of value, and how to read it."""
    if not spec.meaning:
        return spec.description or spec.name
    return (f"{spec.meaning}\n\n**Unit:** {spec.unit}. **Kind:** {spec.kind}. {spec.kind_help}\n\n"
            f"{spec.interpret}\n\nAPI field: `{spec.name}`")


def parse_pasted_row(text: str, names: list[str]) -> dict[str, float]:
    """Read pasted values: a JSON object, or a row of numbers copied from Excel in the API's feature order.

    Raises ValueError with a message that's safe to show.
    """
    text = text.strip()
    if not text:
        raise ValueError("Nothing was pasted.")
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise ValueError("That looks like JSON, but it couldn't be read.") from exc
        missing = [name for name in names if name not in parsed]
        extra = [name for name in parsed if name not in names]
        if missing or extra:
            raise ValueError(f"The JSON doesn't match the form. Missing: {missing or 'none'}. "
                             f"Unknown: {extra or 'none'}.")
        cells = [parsed[name] for name in names]
    else:
        cells = [cell for cell in re.split(r"[\t,;\s]+", text) if cell]
        if len(cells) != len(names):
            raise ValueError(f"Expected {len(names)} numbers, one per feature, but found {len(cells)}. Copy only the "
                             "feature columns, from n_signals to no_print_with_ports.")
    try:
        return {name: float(cell) for name, cell in zip(names, cells, strict=True)}
    except (TypeError, ValueError) as exc:
        raise ValueError("Every value must be a number. Copy only the feature columns, without headers.") from exc


def group_specs(specs: list[FeatureSpec]) -> dict[str, list[FeatureSpec]]:
    """Arrange the API's features into display groups, keeping every one of them."""
    by_name = {spec.name: spec for spec in specs}
    grouped = {group: [by_name[name] for name in names if name in by_name] for group, names in GROUPS.items()}
    other = [spec for spec in specs if spec.name not in LABELS]
    if other:
        grouped["Other"] = other
    return {group: items for group, items in grouped.items() if items}


def validate_inputs(values: dict, specs: list[FeatureSpec]) -> list[str]:
    """Problems that stop a request being sent. The API still validates everything itself."""
    problems = []
    for spec in specs:
        value = values.get(spec.name)
        if value is None:
            problems.append(f"{label_for(spec.name)} is required.")
        elif spec.minimum is not None and value < spec.minimum:
            problems.append(f"{label_for(spec.name)} must be at least {spec.minimum:g}.")
        elif spec.maximum is not None and value > spec.maximum:
            problems.append(f"{label_for(spec.name)} must be at most {spec.maximum:g}.")
    return problems


def consistency_warnings(values: dict, specs: list[FeatureSpec] | None = None) -> list[str]:
    """Combinations the feature pipeline cannot produce. The model still scores them, so these only warn."""
    warnings = []
    for spec in specs or []:
        value = values.get(spec.name)
        if value is not None and spec.kind == "count" and float(value) != int(value):
            warnings.append(f"{label_for(spec.name)} is a whole-number count, but {value:g} was entered.")
    ports = (values.get("n_tcp_ports") or 0) + (values.get("n_udp_ports") or 0)
    if values.get("has_shodan") == 0 and ports > 0:
        warnings.append("Seen by Shodan is No, but open ports are entered. The pipeline sets it to Yes then.")
    if values.get("has_shodan") == 1 and ports == 0:
        warnings.append("Seen by Shodan is Yes, but no open ports are entered. The pipeline sets it to No then.")
    shares = [value for name, value in values.items() if name.startswith(FRACTION_PREFIX) and value is not None]
    if sum(shares) > 1.01:
        warnings.append(f"The attack mix fractions add up to {sum(shares):.2f}, but they cannot exceed 1.")
    days, signals, per_day = values.get("n_active_days"), values.get("n_signals"), values.get("signals_per_day")
    if days and signals and per_day is not None and abs(per_day - signals / days) > 0.01 * max(signals / days, 1):
        warnings.append(f"Signals per active day should equal signals divided by active days ({signals / days:.3f}).")
    return warnings
