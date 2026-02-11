"""Templates API: browse, preview, and quickstart jobs from built-in templates."""

from __future__ import annotations

import os
import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlmodel import Session

from branchforge_core.templates import list_templates, get_template
from branchforge_core.template_heatmaps import (
    generate_template_heatmap,
    save_heatmap_csv,
    list_template_heatmaps,
)

from ..schemas import TemplateResponse
from ..database import get_session
from ..auth import get_current_user
from ..models import User, Job
from ..storage import make_job_dir, save_upload
from ..billing_utils import check_plan_quota
from ..queues import get_queue_for_plan

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class QuickstartResponse(BaseModel):
    job_id: str
    template_id: str
    template_name: str
    message: str


# ---------------------------------------------------------------------------
# Browse templates
# ---------------------------------------------------------------------------

def _template_to_response(t) -> TemplateResponse:
    """Convert a branchforge_core TemplateSpec into a TemplateResponse."""
    return TemplateResponse(
        id=t.id,
        name=t.name,
        description=t.description,
        category=t.category,
        tags=t.tags,
        plate=t.plate.model_dump(),
        ports=t.ports.model_dump(),
        constraints=t.constraints.model_dump(),
        fluid=t.fluid.model_dump(),
        generation=t.generation.model_dump(),
        heatmap_description=t.heatmap_description,
    )


@router.get("/templates/categories", response_model=List[str])
def list_categories():
    """Return a sorted list of unique template categories."""
    templates = list_templates()
    categories = sorted({t.category for t in templates})
    return categories


@router.get("/templates", response_model=List[TemplateResponse])
def list_all_templates():
    """Return all available templates."""
    templates = list_templates()
    return [_template_to_response(t) for t in templates]


@router.get("/templates/{template_id}", response_model=TemplateResponse)
def get_single_template(template_id: str):
    """Return a single template by ID."""
    try:
        t = get_template(template_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _template_to_response(t)


# ---------------------------------------------------------------------------
# Quickstart: one-click job from template
# ---------------------------------------------------------------------------

@router.post("/templates/{template_id}/quickstart", response_model=QuickstartResponse)
def quickstart_job(
    template_id: str,
    n_candidates: Optional[int] = Query(None, ge=1, le=50,
        description="Override number of candidates (default from template)"),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Create a job from a template with auto-generated heatmap.

    This is the "one-click GPU plate" experience. No file upload needed.
    The template's heatmap description is turned into a synthetic
    heatmap grid automatically.

    Steps:
      1. Load template spec
      2. Generate heatmap grid from template definition
      3. Save heatmap CSV to job directory
      4. Create job with full spec
      5. Enqueue to appropriate priority queue
    """
    # Load template
    try:
        tmpl = get_template(template_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Check heatmap generator exists
    available_heatmaps = list_template_heatmaps()
    if template_id not in available_heatmaps:
        raise HTTPException(
            400,
            f"Template '{template_id}' does not have an auto-generated heatmap. "
            f"Available: {available_heatmaps}"
        )

    # Check plan quota
    try:
        check_plan_quota(user, session)
    except Exception as e:
        raise HTTPException(status_code=429, detail=str(e))

    # Create job directory
    job_id = str(uuid.uuid4())
    job_dir = make_job_dir(job_id)

    # Generate and save heatmap
    plate_w = tmpl.plate.width_mm
    plate_h = tmpl.plate.height_mm
    grid = generate_template_heatmap(template_id, plate_w, plate_h, resolution=50)

    heatmap_path = os.path.join(job_dir, "heatmap.csv") if isinstance(job_dir, str) and os.path.isabs(job_dir) else f"{job_dir}/heatmap.csv"
    save_heatmap_csv(grid, heatmap_path)

    # Build spec
    gen = tmpl.generation.model_dump()
    if n_candidates is not None:
        gen["n_candidates"] = n_candidates

    spec = {
        "plate": tmpl.plate.model_dump(),
        "ports": tmpl.ports.model_dump(),
        "heatmap": {
            "kind": "csv",
            "path": heatmap_path,
            "total_watts": float(grid.sum()),
            "flip_y": False,
        },
        "constraints": tmpl.constraints.model_dump(),
        "fluid": tmpl.fluid.model_dump(),
        "generation": gen,
        "template_id": template_id,
    }

    # Create job record
    job = Job(
        id=job_id,
        user_id=user.id,
        status="queued",
        spec_json=spec,
        job_dir=job_dir,
        user_plan=user.plan,
    )
    session.add(job)

    # Increment usage
    user.jobs_used += 1
    session.add(user)
    session.commit()

    # Enqueue
    q = get_queue_for_plan(user.plan)
    q.enqueue("branchforge_worker.tasks.run_job", job_id, job_timeout=60 * 30)

    logger.info(
        "Quickstart job created: template=%s job=%s user=%s",
        template_id, job_id, user.id,
        extra={
            "event": "quickstart.created",
            "template_id": template_id,
            "job_id": job_id,
            "user_plan": user.plan,
        },
    )

    return QuickstartResponse(
        job_id=job_id,
        template_id=template_id,
        template_name=tmpl.name,
        message=f"Job created from template '{tmpl.name}'. Poll /jobs/{job_id} for status.",
    )
