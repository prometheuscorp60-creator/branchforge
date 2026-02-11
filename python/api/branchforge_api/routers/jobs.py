from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlmodel import Session, select

from branchforge_core.schemas import JobSpec
from branchforge_core.storage_backend import get_storage_backend, LocalBackend

from ..db import get_session
from ..models import Job, Candidate, User
from ..schemas import JobCreateResponse, JobStatusResponse, JobLogsResponse, JobListResponse, JobSummary
from ..storage import make_job_dir, save_upload
from ..queues import get_queue_for_plan
from ..billing_utils import check_job_quota, increment_usage, max_candidates
from .auth import get_current_user

router = APIRouter()
logger = logging.getLogger("branchforge.api.jobs")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB
ALLOWED_HEATMAP_MIME_TYPES = {
    "text/csv",
    "application/csv",
    "text/plain",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/tiff",
}
ALLOWED_HEATMAP_EXTENSIONS = {".csv", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff"}


def _validate_upload_size(filename: str, data: bytes) -> None:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' is too large ({len(data)} bytes). Maximum allowed size is 10MB.",
        )


def _validate_heatmap_upload(upload: UploadFile, data: bytes) -> None:
    filename = upload.filename or "heatmap"
    ext = Path(filename).suffix.lower()
    content_type = (upload.content_type or "").lower().strip()

    mime_ok = content_type in ALLOWED_HEATMAP_MIME_TYPES
    ext_ok = ext in ALLOWED_HEATMAP_EXTENSIONS
    if not (mime_ok or ext_ok):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid heatmap file type. Upload a CSV or image file "
                "(csv, png, jpg/jpeg, webp, gif, tif/tiff)."
            ),
        )

    _validate_upload_size(filename, data)


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

    # Quota enforcement via billing_utils
    check_job_quota(user, session)
    increment_usage(user, session)

    # Cap candidates to plan limit
    cap = max_candidates(user)
    if spec_dict.get("generation", {}).get("n_candidates", 0) > cap:
        spec_dict["generation"]["n_candidates"] = cap

    job_id = str(uuid.uuid4())
    job_dir = make_job_dir(job_id)

    # Store uploads
    heatmap_bytes = await heatmap_file.read()
    _validate_heatmap_upload(heatmap_file, heatmap_bytes)
    heatmap_path = save_upload(heatmap_bytes, job_dir, f"heatmap_{heatmap_file.filename or 'heatmap'}")

    if outline_file is not None:
        outline_bytes = await outline_file.read()
        _validate_upload_size(outline_file.filename or "outline", outline_bytes)
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

    # Save job with user_plan snapshot for worker-side enforcement
    job = Job(
        id=job_id,
        user_id=user.id,
        status="queued",
        spec_json=job_spec.model_dump(),
        job_dir=job_dir,
        user_plan=user.plan,
    )
    session.add(job)
    session.commit()

    # Enqueue to plan-appropriate priority queue
    q = get_queue_for_plan(user.plan)
    q.enqueue("branchforge_worker.tasks.run_job", job_id)

    logger.info(
        "Job created",
        extra={
            "event": "job.created",
            "job_id": job_id,
            "user_id": user.id,
            "plan": user.plan,
            "queue": q.name,
            "n_candidates": job_spec.generation.n_candidates if hasattr(job_spec, "generation") else None,
        },
    )

    return JobCreateResponse(job_id=job_id)


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Paginated list of jobs for the current user."""
    query = select(Job).where(Job.user_id == user.id)
    if status:
        query = query.where(Job.status == status)
    query = query.order_by(Job.created_at.desc())

    # Total count
    from sqlmodel import func
    count_query = select(func.count()).select_from(Job).where(Job.user_id == user.id)
    if status:
        count_query = count_query.where(Job.status == status)
    total = session.exec(count_query).one()

    # Paginate
    offset = (page - 1) * per_page
    jobs = session.exec(query.offset(offset).limit(per_page)).all()

    summaries = []
    for job in jobs:
        cand_count = session.exec(
            select(func.count()).select_from(Candidate).where(Candidate.job_id == job.id)
        ).one()
        tpl_name = job.spec_json.get("_template_name") if isinstance(job.spec_json, dict) else None
        summaries.append(JobSummary(
            job_id=job.id,
            status=job.status,
            created_at=job.created_at.isoformat(),
            error=job.error,
            candidate_count=cand_count,
            template_name=tpl_name,
        ))

    return JobListResponse(jobs=summaries, total=total, page=page, per_page=per_page)


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


@router.get("/jobs/{job_id}/logs", response_model=JobLogsResponse)
def job_logs(job_id: str, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    """Return the content of job.log for a given job."""
    job = session.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Job not found")

    backend = get_storage_backend()
    log_key = f"{job.job_dir}/job.log".replace("\\", "/")

    if isinstance(backend, LocalBackend):
        log_path = os.path.join(job.job_dir, "job.log") if os.path.isabs(job.job_dir) else backend.get_local_path(log_key)
        if not os.path.exists(log_path):
            raise HTTPException(status_code=404, detail="Log file not found")
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            log_content = f.read()
    else:
        if not backend.exists(log_key):
            raise HTTPException(status_code=404, detail="Log file not found")
        log_content = backend.read_text(log_key)

    return JobLogsResponse(job_id=job_id, log=log_content)


@router.get("/jobs/{job_id}/manifest")
def job_manifest(job_id: str, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    """Return the manifest.json for a given job."""
    job = session.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Job not found")

    backend = get_storage_backend()
    manifest_key = f"{job.job_dir}/manifest.json".replace("\\", "/")

    if isinstance(backend, LocalBackend):
        manifest_path = os.path.join(job.job_dir, "manifest.json") if os.path.isabs(job.job_dir) else backend.get_local_path(manifest_key)
        if not os.path.exists(manifest_path):
            raise HTTPException(status_code=404, detail="Manifest not found")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        if not backend.exists(manifest_key):
            raise HTTPException(status_code=404, detail="Manifest not found")
        manifest = json.loads(backend.read_text(manifest_key))

    return manifest
