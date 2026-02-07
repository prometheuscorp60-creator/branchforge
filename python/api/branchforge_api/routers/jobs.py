from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlmodel import Session, select

from branchforge_core.schemas import JobSpec

from ..db import get_session
from ..models import Job, Candidate, User
from ..schemas import JobCreateResponse, JobStatusResponse
from ..storage import make_job_dir, save_upload
from ..queues import queue
from .auth import get_current_user

router = APIRouter()


@router.post("/jobs", response_model=JobCreateResponse)
async def create_job(
    spec: str = Form(...),
    heatmap_file: UploadFile = File(...),
    outline_file: Optional[UploadFile] = File(None),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    try:
        spec_dict = json.loads(spec)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in spec: {e}")

    # Quota enforcement (simple but effective)
    if user.plan == "free":
        if datetime.utcnow() - user.month_start > timedelta(days=30):
            user.month_start = datetime.utcnow()
            user.jobs_used = 0
        if user.jobs_used >= 3:
            raise HTTPException(status_code=402, detail="Free plan limit reached (3 jobs / 30 days). Upgrade to Pro.")
        user.jobs_used += 1
        session.add(user)
        session.commit()

    job_id = str(uuid.uuid4())
    job_dir = make_job_dir(job_id)

    # Store uploads
    heatmap_bytes = await heatmap_file.read()
    heatmap_path = save_upload(heatmap_bytes, job_dir, f"heatmap_{heatmap_file.filename or 'heatmap'}")

    if outline_file is not None:
        outline_bytes = await outline_file.read()
        outline_path = save_upload(outline_bytes, job_dir, f"outline_{outline_file.filename or 'outline.dxf'}")
        # set plate to polygon + dxf_path
        spec_dict.setdefault("plate", {})
        spec_dict["plate"]["kind"] = "polygon"
        spec_dict["plate"]["dxf_path"] = outline_path

    # attach heatmap path
    spec_dict.setdefault("heatmap", {})
    spec_dict["heatmap"]["path"] = heatmap_path

    # Validate spec
    try:
        job_spec = JobSpec.model_validate(spec_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Spec validation failed: {e}")

    job = Job(id=job_id, user_id=user.id, status="queued", spec_json=job_spec.model_dump(), job_dir=job_dir)
    session.add(job)
    session.commit()

    # enqueue
    queue.enqueue("branchforge_worker.tasks.run_job", job_id)

    return JobCreateResponse(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    job = session.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Job not found")

    cands = session.exec(select(Candidate).where(Candidate.job_id == job_id).order_by(Candidate.index)).all()
    cand_list = []
    for c in cands:
        cand_list.append({
            "id": c.id,
            "index": c.index,
            "label": c.label,
            "metrics": c.metrics_json,
            "artifacts": c.artifacts_json,
        })

    return JobStatusResponse(job_id=job_id, status=job.status, error=job.error, candidates=cand_list)
