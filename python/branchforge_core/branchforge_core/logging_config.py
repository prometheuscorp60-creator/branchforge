"""Structured JSON logging configuration for BranchForge.

Shared by both the API and worker processes.  In production set
``LOG_FORMAT=json`` (the default) for machine-readable output;
use ``LOG_FORMAT=text`` for local development.

Usage::

    from branchforge_core.logging_config import configure_logging
    configure_logging("api")
"""
from __future__ import annotations

import json
import logging
import os
import sys
import contextvars
from datetime import datetime, timezone

# ── Context propagation ────────────────────────────────────────────
# Allows request IDs (API) or job IDs (worker) to appear in every
# log record from the same async/thread context.

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)
job_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "job_id", default=""
)


# ── JSON Formatter ─────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        # Inject context vars
        rid = request_id_var.get("")
        if rid:
            entry["request_id"] = rid
        jid = job_id_var.get("")
        if jid:
            entry["job_id"] = jid

        # Propagate structured extras (e.g. ``logger.info("x", extra={...})``)
        for key in (
            "request_id", "job_id", "user_id", "duration_ms",
            "event", "status_code", "method", "path",
            "timeout_seconds", "n_candidates", "elapsed_s",
            "peak_memory_mb", "category",
        ):
            val = getattr(record, key, None)
            if val is not None and key not in entry:
                entry[key] = val

        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


# ── Text Formatter (dev) ───────────────────────────────────────────

_TEXT_FMT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_TEXT_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


# ── Public setup function ──────────────────────────────────────────

def configure_logging(service: str = "api") -> None:
    """Configure the root logger for the given service.

    Environment variables
    ---------------------
    LOG_FORMAT : ``json`` (default) or ``text``
    LOG_LEVEL  : standard Python level name, default ``INFO``
    """
    log_format = os.getenv("LOG_FORMAT", "json")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))
    # Remove any pre-existing handlers (e.g. from basicConfig)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FMT, datefmt=_TEXT_DATEFMT))

    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for name in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(f"branchforge.{service}").info(
        "Logging configured",
        extra={"event": "logging.init", "service": service, "log_format": log_format},
    )
