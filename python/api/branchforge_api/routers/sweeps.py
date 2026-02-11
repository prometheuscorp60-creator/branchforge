"""Parametric sweep API endpoints."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlmodel import Session

from branchforge_core.sweep import SweepSpec, SweepResult, SWEEPABLE_PARAMS

from ..db import get_session
from ..models import User
from ..storage import make_job_dir, save_upload
from ..billing_utils import check_job_quota
from .auth import get_current_user

router = APIRouter()


@router.get("/sweeps/parameters")
async def list_sweep_parameters():
    """List all parameters available for sweeping with metadata."""
    return {
        key: {
            "label": info["label"],
            "min": info["min"],
            "max": info["max"],
            "default_start": info["default_start"],
            "default_stop": info["default_stop"],
            "unit": info["unit"],
        }
        for key, info in SWEEPABLE_PARAMS.items()
    }


@router.post("/sweeps", response_model=SweepResult)
async def create_sweep(
    spec: str = Form(...),
    sweep_param: str = Form(...),
    sweep_start: float = Form(...),
    sweep_stop: float = Form(...),
    sweep_steps: int = Form(8),
    heatmap_file: UploadFile = File(...),
    outline_file: Optional[UploadFile] = File(None),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Run a parametric sweep over a design parameter."""
    try:
        spec_dict = json.loads(spec)
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON: {e}")

    if sweep_param not in SWEEPABLE_PARAMS:
        raise HTTPException(400, f"Unknown sweep parameter: {sweep_param}. "
                            f"Available: {list(SWEEPABLE_PARAMS.keys())}")

    if sweep_steps < 2 or sweep_steps > 20:
        raise HTTPException(400, "sweep_steps must be between 2 and 20")

    check_job_quota(user, session)

    sweep_spec = SweepSpec(
        parameter=sweep_param,
        start=sweep_start,
        stop=sweep_stop,
        n_steps=sweep_steps,
    )

    job_dir = make_job_dir(f"sweep_{user.id}")
    heatmap_path = save_upload(heatmap_file, job_dir, "heatmap")
    spec_dict.setdefault("heatmap", {})["path"] = heatmap_path

    if outline_file:
        outline_path = save_upload(outline_file, job_dir, "outline")
        spec_dict.setdefault("plate", {})["dxf_path"] = outline_path

    from branchforge_core.sweep import run_sweep
    result = run_sweep(spec_dict, sweep_spec, job_dir)

    return result
