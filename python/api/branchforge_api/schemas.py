from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class JobCreateResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    candidates: List[Dict[str, Any]] = []


class JobLogsResponse(BaseModel):
    job_id: str
    log: str


class JobSummary(BaseModel):
    job_id: str
    status: str
    created_at: str
    error: Optional[str] = None
    candidate_count: int = 0
    template_name: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: List[JobSummary]
    total: int
    page: int
    per_page: int


class TemplateResponse(BaseModel):
    id: str
    name: str
    description: str
    category: str
    tags: List[str] = []
    plate: Dict[str, Any]
    ports: Dict[str, Any]
    constraints: Dict[str, Any]
    fluid: Dict[str, Any]
    generation: Dict[str, Any]
    heatmap_description: str = ""
