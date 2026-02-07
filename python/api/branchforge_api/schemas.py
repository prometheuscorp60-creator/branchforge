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
