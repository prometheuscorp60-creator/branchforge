from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session

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

    abs_path = os.path.join(job.job_dir, rel)
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="Artifact missing on disk")

    media_type = _ALLOWED[artifact_key]
    filename = Path(abs_path).name
    return FileResponse(abs_path, media_type=media_type, filename=filename)
