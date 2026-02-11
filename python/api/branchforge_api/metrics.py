"""Lightweight Prometheus-compatible in-process metrics.

No external dependency required — just thread-safe counters and
histograms rendered as Prometheus text exposition format.

Usage::

    from .metrics import inc, observe, render_prometheus

    inc("http_requests_total", {"method": "GET", "status": "200"})
    observe("http_request_duration_ms", 42.3)
    text = render_prometheus()
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Optional


_lock = threading.Lock()
_counters: dict[str, int] = defaultdict(int)
_histograms: dict[str, list[float]] = defaultdict(list)
_gauges: dict[str, float] = {}


def _key(name: str, labels: Optional[dict] = None) -> str:
    if labels:
        lstr = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{lstr}}}"
    return name


# ── Public API ─────────────────────────────────────────────────────

def inc(name: str, labels: Optional[dict] = None, value: int = 1) -> None:
    """Increment a counter."""
    key = _key(name, labels)
    with _lock:
        _counters[key] += value


def observe(name: str, value: float, labels: Optional[dict] = None) -> None:
    """Record a histogram observation."""
    key = _key(name, labels)
    with _lock:
        _histograms[key].append(value)


def gauge(name: str, value: float, labels: Optional[dict] = None) -> None:
    """Set a gauge to a specific value."""
    key = _key(name, labels)
    with _lock:
        _gauges[key] = value


def render_prometheus() -> str:
    """Render all metrics in Prometheus text exposition format."""
    lines: list[str] = []
    with _lock:
        # Counters
        for key, val in sorted(_counters.items()):
            lines.append(f"{key} {val}")
        # Histograms (summary style)
        for key, vals in sorted(_histograms.items()):
            if vals:
                lines.append(f"{key}_count {len(vals)}")
                lines.append(f"{key}_sum {sum(vals):.3f}")
                lines.append(f"{key}_avg {sum(vals) / len(vals):.3f}")
        # Gauges
        for key, val in sorted(_gauges.items()):
            lines.append(f"{key} {val}")
    return "\n".join(lines) + "\n" if lines else ""
