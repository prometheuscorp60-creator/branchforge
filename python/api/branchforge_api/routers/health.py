"""Health, readiness, and metrics endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

router = APIRouter()
logger = logging.getLogger("branchforge.api.health")


@router.get("/health")
def health():
    """Liveness probe — returns 200 if the process is up."""
    return {"ok": True}


@router.get("/health/ready")
def readiness():
    """Readiness probe — verifies DB, Redis, and storage connectivity."""
    checks: dict[str, str] = {}

    # ── Database ────────────────────────────────────────────────
    try:
        from ..db import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    # ── Redis ───────────────────────────────────────────────────
    try:
        from ..queues import redis_conn
        redis_conn.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    # ── Storage ─────────────────────────────────────────────────
    try:
        from branchforge_core.storage_backend import get_storage_backend
        backend = get_storage_backend()
        checks["storage"] = backend.health_check()
    except Exception as e:
        checks["storage"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503

    if not all_ok:
        logger.warning("Readiness check failed: %s", checks)

    return JSONResponse(
        content={"ok": all_ok, "checks": checks},
        status_code=status_code,
    )


@router.get("/metrics")
def metrics():
    """Prometheus-compatible metrics endpoint."""
    from ..metrics import render_prometheus
    return PlainTextResponse(render_prometheus(), media_type="text/plain; charset=utf-8")
