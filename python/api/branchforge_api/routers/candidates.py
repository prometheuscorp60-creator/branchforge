from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from sqlmodel import Session

from branchforge_core.storage_backend import get_storage_backend, LocalBackend

from ..db import get_session
from ..models import Candidate, Job, User
from .auth import get_current_user
from ..settings import settings

router = APIRouter()

_ALLOWED = {
    "preview_png": "image/png",
    "report_pdf": "application/pdf",
    "plate_step": "application/step",
    "channels_step": "application/step",
    "plate_stl": "model/stl",
    "channels_dxf": "application/dxf",
    "json_paths": "application/json",
}

@router.get("/candidates/{candidate_id}/download/{artifact_key}")
def download_candidate_artifact(candidate_id: str, artifact_key: str, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    if artifact_key not in _ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unknown artifact_key. Allowed: {sorted(_ALLOWED.keys())}")

    cand = session.get(Candidate, candidate_id)
    if cand is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    job = session.get(Job, cand.job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Job not found")

    # plan gating: Free cannot download STEP
    if user.plan == "free" and artifact_key in ("plate_step", "channels_step"):
        raise HTTPException(status_code=402, detail="STEP export is Pro+. Upgrade to Pro.")

    rel = cand.artifacts_json.get(artifact_key)
    if not rel:
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Use the storage backend to serve the file
    backend = get_storage_backend()
    remote_key = f"{job.job_dir}/{rel}".replace("\\", "/")

    if isinstance(backend, LocalBackend):
        # Local: serve directly from filesystem
        abs_path = backend.get_local_path(remote_key) if not os.path.isabs(job.job_dir) else os.path.join(job.job_dir, rel)
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404, detail="Artifact missing on disk")
        media_type = _ALLOWED[artifact_key]
        filename = Path(abs_path).name
        return FileResponse(abs_path, media_type=media_type, filename=filename)
    else:
        # S3: redirect to a presigned URL
        if not backend.exists(remote_key):
            raise HTTPException(status_code=404, detail="Artifact not found in storage")
        presigned = backend.get_presigned_url(remote_key, expires_in=3600)
        if presigned:
            return RedirectResponse(presigned)
        raise HTTPException(status_code=500, detail="Failed to generate download URL")
