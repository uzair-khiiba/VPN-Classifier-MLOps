"""What each of the 28 model inputs means. Used by the Excel export and the dashboard form.

The calculations themselves live in src/features.py; this file only describes them.
"""

COUNT, FRACTION, BINARY, DURATION, RATE = "count", "fraction", "binary", "duration", "rate"

KIND_HELP = {
    COUNT: "Whole number: 0, 1, 2, ... (the fingerprint counts are sums, so they can be large)",
    FRACTION: "Decimal from 0 to 1. 0.25 means 25% of this IP's signals",
    BINARY: "Exactly 0 (no) or 1 (yes)",
    DURATION: "Non-negative decimal measured in hours or seconds",
    RATE: "Non-negative decimal: signals per day",
}


def _share(attack: str) -> dict:
    return {"kind": FRACTION, "unit": "fraction of signals, 0 to 1",
            "meaning": f"Share of this IP's signals that have the attack type {attack}.",
            "interpret": "0 = never used this attack; 1 = every signal used it. Shares need not add up to 1, "
                         "because rarer attack types have no column."}


def _shared(print_name: str) -> dict:
    return {"kind": COUNT, "unit": "other IPs, summed per fingerprint", "population": True,
            "meaning": f"For each {print_name} seen on this IP, how many other IPs in the Shodan file have it, "
                       "added up.",
            "interpret": "0 = no fingerprint, or a unique one. Large = shared with many IPs. Depends on the whole "
                         "Shodan file, including test IPs, so it can't be worked out from one IP alone."}


FEATURE_DOCS = {
    "n_signals": {"kind": COUNT, "unit": "signals", "meaning": "Attack signals reported for this IP.",
                  "interpret": "Higher = reported more often. Minimum is 1."},
    "n_attack_types": {"kind": COUNT, "unit": "attack types", "meaning": "Distinct attack types this IP used.",
                       "interpret": "Higher = more varied behaviour. Minimum is 1."},
    "n_watchers": {"kind": COUNT, "unit": "watchers", "meaning": "Distinct CrowdSec watchers that reported this IP.",
                   "interpret": "Higher = seen by more sensors. Minimum is 1."},
    "n_watcher_as": {"kind": COUNT, "unit": "networks (ASNs)", "meaning": "Distinct networks those watchers belong to.",
                     "interpret": "Higher = reported from more networks. Can be 0 if the network is missing."},
    "n_watcher_countries": {"kind": COUNT, "unit": "countries", "meaning": "Distinct countries of those watchers.",
                            "interpret": "Higher = reported more widely. Can be 0 if the country is missing."},
    "n_attacker_as": {"kind": COUNT, "unit": "networks (ASNs)",
                      "meaning": "Distinct networks this attacker IP was recorded in.",
                      "interpret": "Usually 1. Can be 0 if the network is missing."},
    "active_hours": {"kind": DURATION, "unit": "hours", "meaning": "Time between this IP's first and last signal.",
                     "interpret": "0 = all signals at the same moment."},
    "n_active_days": {"kind": COUNT, "unit": "days", "meaning": "Distinct UTC days with at least one signal.",
                      "interpret": "Minimum is 1."},
    "signals_per_day": {"kind": RATE, "unit": "signals per active day",
                        "meaning": "n_signals divided by n_active_days.",
                        "interpret": "Should equal those two values divided. If not, the pipeline couldn't produce "
                                     "this record."},
    "max_signals_per_hour": {"kind": COUNT, "unit": "signals",
                             "meaning": "Signals in this IP's busiest clock hour (UTC).",
                             "interpret": "Higher = sharper bursts. Minimum is 1."},
    "mean_gap_s": {"kind": DURATION, "unit": "seconds", "meaning": "Average time between consecutive signals.",
                   "interpret": "0 if the IP has one signal, or all its signals share a timestamp."},
    "median_gap_s": {"kind": DURATION, "unit": "seconds", "meaning": "Median time between consecutive signals.",
                     "interpret": "Less affected than the mean by one long pause."},
    "asn_unique_ips": {"kind": COUNT, "unit": "attacker IPs", "population": True,
                       "meaning": "Distinct attacker IPs in this IP's network (the largest value if it has several).",
                       "interpret": "Larger = a more crowded network. Depends on the whole signals file, including "
                                    "test IPs, so it can't be worked out from one IP alone."},
    "share_http_spam": _share("http:spam"),
    "share_http_scan": _share("http:scan"),
    "share_http_exploit": _share("http:exploit"),
    "share_http_bruteforce": _share("http:bruteforce"),
    "share_http_crawl": _share("http:crawl"),
    "share_ssh_bruteforce": _share("ssh:bruteforce"),
    "share_pop3_imap_bruteforce": _share("pop3/imap:bruteforce"),
    "share_tcp_scan": _share("tcp:scan"),
    "has_shodan": {"kind": BINARY, "unit": "0 or 1", "meaning": "1 if Shodan recorded at least one open port.",
                   "interpret": "Should be 1 when either port count is above 0, otherwise 0."},
    "n_tcp_ports": {"kind": COUNT, "unit": "ports", "meaning": "Open TCP ports recorded by Shodan.",
                    "interpret": "0 = none recorded, which doesn't prove none are open."},
    "n_udp_ports": {"kind": COUNT, "unit": "ports", "meaning": "Open UDP ports recorded by Shodan.",
                    "interpret": "0 = none recorded, which doesn't prove none are open."},
    "jarm_shared": _shared("JARM (TLS server) fingerprint"),
    "ja3s_shared": _shared("JA3S (TLS server hello) fingerprint"),
    "headers_hash_shared": _shared("HTTP header hash"),
    "no_print_with_ports": {"kind": BINARY, "unit": "0 or 1",
                            "meaning": "1 if Shodan saw open ports but no fingerprints.",
                            "interpret": "Can only be 1 when has_shodan is 1."},
}

POPULATION_FEATURES = [name for name, doc in FEATURE_DOCS.items() if doc.get("population")]
