from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from branchforge_core.logging_config import configure_logging, request_id_var

# Configure structured logging before anything else
configure_logging("api")

from .settings import DEV_JWT_SECRET, settings
from .db import init_db
from .routers.health import router as health_router
from .routers.jobs import router as jobs_router
from .routers.candidates import router as candidates_router
from .routers.auth import router as auth_router
from .routers.templates import router as templates_router
from .routers.oauth import router as oauth_router
from .routers.billing import router as billing_router
from .routers.sweeps import router as sweeps_router
from .routers.organizations import router as orgs_router
from .routers.projects import router as projects_router
from .routers.webhooks import router as webhooks_router
from .routers.analytics import router as analytics_router
from .routers.calibration import router as calibration_router
from .routers.reports import router as reports_router

logger = logging.getLogger("branchforge.api")

app = FastAPI(title="BranchForge API", version="0.2.0")


# ── Request ID middleware ──────────────────────────────────────────
# Propagates a unique ID through every request for log correlation.

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = rid
        request_id_var.set(rid)

        start = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - start) * 1000, 1)

        response.headers["X-Request-ID"] = rid

        # Structured access log
        logger.info(
            "%s %s %s %.0fms",
            request.method, request.url.path,
            response.status_code, duration_ms,
            extra={
                "event": "http.request",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "request_id": rid,
            },
        )

        # Lightweight in-process metrics
        from .metrics import inc, observe
        inc("http_requests_total", {"method": request.method, "status": str(response.status_code)})
        observe("http_request_duration_ms", duration_ms)

        return response


class CSRFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in {"POST", "PUT", "DELETE"}:
            # Keep Stripe webhook functional (server-to-server; no browser Origin header).
            if request.url.path != "/api/v1/billing/webhook":
                origin = request.headers.get("origin")
                if origin and origin not in settings.cors_origins and "*" not in settings.cors_origins:
                    return JSONResponse(status_code=403, content={"detail": "Invalid request origin"})

        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CSRFMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    if not settings.dev_no_auth and settings.jwt_secret == DEV_JWT_SECRET:
        raise RuntimeError("JWT_SECRET is using the insecure dev default while DEV_NO_AUTH=false")

    if settings.dev_no_auth and settings.jwt_secret == DEV_JWT_SECRET:
        logger.warning("DEV_NO_AUTH=true with default JWT_SECRET; this configuration is insecure")

    if not settings.stripe_webhook_secret:
        logger.warning("STRIPE_WEBHOOK_SECRET is not set; billing webhook endpoint will return 503")

    init_db()


app.include_router(health_router, prefix="/api/v1", tags=["health"])
app.include_router(jobs_router, prefix="/api/v1", tags=["jobs"])
app.include_router(candidates_router, prefix="/api/v1", tags=["candidates"])
app.include_router(auth_router, prefix="/api/v1", tags=["auth"])
app.include_router(templates_router, prefix="/api/v1", tags=["templates"])
app.include_router(oauth_router, prefix="/api/v1", tags=["oauth"])
app.include_router(billing_router, prefix="/api/v1", tags=["billing"])
app.include_router(sweeps_router, prefix="/api/v1", tags=["sweeps"])
app.include_router(orgs_router, prefix="/api/v1", tags=["organizations"])
app.include_router(projects_router, prefix="/api/v1", tags=["projects"])
app.include_router(webhooks_router, prefix="/api/v1", tags=["webhooks"])
app.include_router(analytics_router, prefix="/api/v1", tags=["analytics"])
app.include_router(calibration_router, prefix="/api/v1", tags=["calibration"])
app.include_router(reports_router, prefix="/api/v1", tags=["reports"])
