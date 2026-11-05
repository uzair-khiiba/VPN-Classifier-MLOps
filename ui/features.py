"""Presentation details for the model's inputs. The inputs themselves come from the API's schema.

This file only adds friendly labels and groups. A feature the API declares but this file does not
know still appears in the form, under "Other", so the form can never hide a required input.
"""

import math

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
# Features that count something: the API accepts only whole numbers for these.
COUNT_FEATURES = {"n_signals", "n_attack_types", "n_watchers", "n_watcher_as", "n_watcher_countries", "n_attacker_as",
                  "n_active_days", "max_signals_per_hour", "asn_unique_ips", "n_tcp_ports", "n_udp_ports",
                  "jarm_shared", "ja3s_shared", "headers_hash_shared"}
FRACTION_PREFIX = "share_"


def label_for(name: str) -> str:
    return LABELS.get(name, name)


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
        elif spec.name in COUNT_FEATURES and not float(value).is_integer():
            problems.append(f"{label_for(spec.name)} is a count, so it must be a whole number.")
    return problems


def consistency_warnings(values: dict) -> list[str]:
    """Combinations the feature pipeline can never produce. The API rejects them (HTTP 422) with the same
    rules, so the form reports them before sending anything."""
    get = values.get
    problems = []
    ports = (get("n_tcp_ports") or 0) + (get("n_udp_ports") or 0)
    if get("has_shodan") == 0 and ports > 0:
        problems.append("Seen by Shodan is No, but open ports are entered. It must be Yes when there are ports.")
    if get("has_shodan") == 1 and ports == 0:
        problems.append("Seen by Shodan is Yes, but no open ports are entered. It must be No without ports.")
    if get("no_print_with_ports") == 1 and ports == 0:
        problems.append("Ports but no fingerprints is Yes, but no open ports are entered.")
    shares = [value for name, value in values.items() if name.startswith(FRACTION_PREFIX) and value is not None]
    if sum(shares) > 1 + 1e-6:
        problems.append(f"The attack mix fractions add up to {sum(shares):.4f}, but they cannot exceed 1.")
    signals = get("n_signals")
    for name in ("max_signals_per_hour", "n_attack_types", "n_watchers", "n_active_days"):
        if signals is not None and get(name) is not None and get(name) > signals:
            problems.append(f"{label_for(name)} cannot be more than {label_for('n_signals')} ({signals:g}).")
    days, per_day = get("n_active_days"), get("signals_per_day")
    if days and signals and per_day is not None and not math.isclose(per_day, signals / days, rel_tol=1e-3,
                                                                     abs_tol=1e-3):
        problems.append(f"Signals per active day must equal signals divided by active days ({signals / days:.4f}).")
    hours = get("active_hours")
    gaps = [get(name) for name in ("mean_gap_s", "median_gap_s") if get(name) is not None]
    if hours is not None and gaps and max(gaps) > hours * 3600 + 1:
        problems.append("A gap between signals cannot be longer than the active hours.")
    return problems
