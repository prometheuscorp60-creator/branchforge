from __future__ import annotations

from pydantic import BaseModel
import logging
import os

DEV_JWT_SECRET = "branchforge-dev-secret-change-in-prod"


class Settings(BaseModel):
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    cors_origins: list[str] = [o.strip() for o in os.getenv("API_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    data_dir: str = os.getenv("DATA_DIR", "/data")
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "/data/artifacts")
    db_url: str = os.getenv("DB_URL", "postgresql://branchforge:branchforge_dev@postgres:5432/branchforge")

    dev_no_auth: bool = os.getenv("DEV_NO_AUTH", "false").lower() in ("1", "true", "yes")

    job_timeout_seconds: int = int(os.getenv("JOB_TIMEOUT_SECONDS", "1800"))

    # JWT
    jwt_secret: str = os.getenv("JWT_SECRET", DEV_JWT_SECRET)

    # OAuth providers (leave empty to disable)
    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    github_client_id: str = os.getenv("GITHUB_CLIENT_ID", "")
    github_client_secret: str = os.getenv("GITHUB_CLIENT_SECRET", "")

    # Frontend URL for OAuth redirects
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

    # Stripe billing
    stripe_secret_key: str = os.getenv("STRIPE_SECRET_KEY", "")
    stripe_webhook_secret: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    stripe_price_pro: str = os.getenv("STRIPE_PRICE_PRO", "")
    stripe_price_team: str = os.getenv("STRIPE_PRICE_TEAM", "")
    stripe_price_enterprise: str = os.getenv("STRIPE_PRICE_ENTERPRISE", "")

    # Email (Resend)
    resend_api_key: str = os.getenv("RESEND_API_KEY", "")
    email_from: str = os.getenv("EMAIL_FROM", "BranchForge <noreply@branchforge.app>")

    # Storage backend (local | s3)
    storage_backend: str = os.getenv("STORAGE_BACKEND", "local")
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_region: str = os.getenv("S3_REGION", "us-east-1")
    s3_access_key_id: str = os.getenv("S3_ACCESS_KEY_ID", "")
    s3_secret_access_key: str = os.getenv("S3_SECRET_ACCESS_KEY", "")
    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "")


settings = Settings()

logger = logging.getLogger("branchforge.api.settings")
if (not settings.dev_no_auth) and settings.storage_backend.lower() == "local":
    logger.warning(
        "Production mode with STORAGE_BACKEND=local. Artifacts may be lost on restart/scale-out. Use STORAGE_BACKEND=s3 for production.",
        extra={
            "event": "settings.storage_backend.local_in_production",
            "storage_backend": settings.storage_backend,
            "dev_no_auth": settings.dev_no_auth,
        },
    )
