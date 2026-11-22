"""Request monitoring for the API: request counts, error counts and latency, per endpoint.

Kept in memory, so the numbers cover the current process since it started (a restart or
redeploy resets them). Every request is also written to the log as one line, which is what
the hosting platform's log viewer shows:

    request method=POST path=/predict status=200 latency_ms=4.21

`GET /metrics` returns the totals and the per-endpoint breakdown as JSON.
"""

import threading
import time
from datetime import UTC, datetime

LATENCY_WINDOW = 1000  # recent requests kept per endpoint for the p95


class RequestMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.started = time.monotonic()
            self.started_at = datetime.now(UTC).isoformat(timespec="seconds")
            self.endpoints: dict[str, dict] = {}
            self.predictions = {"VPN/Proxy": 0, "Not VPN/Proxy": 0}

    def record(self, method: str, path: str, status: int, latency_ms: float) -> None:
        key = f"{method} {path}"
        with self._lock:
            stats = self.endpoints.setdefault(key, {"count": 0, "errors": 0, "total_ms": 0.0, "max_ms": 0.0,
                                                    "recent_ms": []})
            stats["count"] += 1
            stats["errors"] += status >= 500
            stats["total_ms"] += latency_ms
            stats["max_ms"] = max(stats["max_ms"], latency_ms)
            stats["recent_ms"].append(latency_ms)
            if len(stats["recent_ms"]) > LATENCY_WINDOW:
                del stats["recent_ms"][0]

    def record_prediction(self, label: str) -> None:
        with self._lock:
            self.predictions[label] = self.predictions.get(label, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            total = sum(s["count"] for s in self.endpoints.values())
            total_ms = sum(s["total_ms"] for s in self.endpoints.values())
            endpoints = {
                key: {"count": s["count"], "errors": s["errors"],
                      "avg_latency_ms": round(s["total_ms"] / s["count"], 2),
                      "p95_latency_ms": round(_percentile(s["recent_ms"], 95), 2),
                      "max_latency_ms": round(s["max_ms"], 2)}
                for key, s in sorted(self.endpoints.items())
            }
            return {
                "started_at": self.started_at,
                "uptime_s": round(time.monotonic() - self.started, 1),
                "total_requests": total,
                "total_errors": sum(s["errors"] for s in self.endpoints.values()),
                "avg_latency_ms": round(total_ms / total, 2) if total else 0.0,
                "predictions": dict(self.predictions),
                "endpoints": endpoints,
            }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * len(ordered)) - 1))
    return ordered[index]


METRICS = RequestMetrics()
