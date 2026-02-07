from __future__ import annotations

from pydantic import BaseModel
import os


class Settings(BaseModel):
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    cors_origins: list[str] = [o.strip() for o in os.getenv("API_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    data_dir: str = os.getenv("DATA_DIR", "/data")
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "/data/artifacts")
    db_url: str = os.getenv("DB_URL", "sqlite:////data/branchforge.sqlite")

    dev_no_auth: bool = os.getenv("DEV_NO_AUTH", "true").lower() in ("1", "true", "yes")


settings = Settings()
