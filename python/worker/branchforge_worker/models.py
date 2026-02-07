from __future__ import annotations

from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, Column, JSON


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    status: str = Field(default="queued", index=True)
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
