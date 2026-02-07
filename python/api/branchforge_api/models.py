from __future__ import annotations

from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, Column, JSON


class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    email: str = Field(index=True, unique=True)
    api_key: str = Field(index=True, unique=True)
    plan: str = Field(default="free", index=True)  # free|pro|team|enterprise
    month_start: datetime = Field(default_factory=datetime.utcnow)
    jobs_used: int = Field(default=0)


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    status: str = Field(default="queued", index=True)  # queued|running|succeeded|failed
    spec_json: dict = Field(sa_column=Column(JSON))
    job_dir: str
    error: Optional[str] = None


class Candidate(SQLModel, table=True):
    id: str = Field(primary_key=True)
    job_id: str = Field(index=True)
    index: int = Field(index=True)
    label: str = Field(default="Candidate")
    metrics_json: dict = Field(sa_column=Column(JSON))
    artifacts_json: dict = Field(sa_column=Column(JSON))
