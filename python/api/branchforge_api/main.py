from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .settings import settings
from .db import init_db
from .routers.health import router as health_router
from .routers.jobs import router as jobs_router
from .routers.candidates import router as candidates_router
from .routers.auth import router as auth_router

app = FastAPI(title="BranchForge API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    init_db()

app.include_router(health_router, prefix="/api/v1", tags=["health"])
app.include_router(jobs_router, prefix="/api/v1", tags=["jobs"])
app.include_router(candidates_router, prefix="/api/v1", tags=["candidates"])
app.include_router(auth_router, prefix="/api/v1", tags=["auth"])
