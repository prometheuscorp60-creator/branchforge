from __future__ import annotations

import os
from pydantic import BaseModel


class Settings(BaseModel):
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    db_url: str = os.getenv("DB_URL", "sqlite:////data/branchforge.sqlite")
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "/data/artifacts")

settings = Settings()
