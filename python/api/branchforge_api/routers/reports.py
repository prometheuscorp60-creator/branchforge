"""Reports API: generate HTML reports, comparison tables, compliance checks, CFD handoff."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlmodel import Session, select

from ..database import get_session
from ..auth import get_current_user
from ..models import User, Job, Candidate

from branchforge_core.reporting import (
    render_report_html,
    render_comparison_html,
    generate_compliance_checklist,
    generate_comparison_table,
    generate_cfd_handoff,
)
from branchforge_core.calibration import (
    apply_corrections_to_metrics,
    get_default_profile,
    CalibrationProfile,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# HTML report for a single candidate
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/candidates/{candidate_index}/report.html",
            response_class=HTMLResponse)
def candidate_html_report(
    job_id: str,
    candidate_index: int,
    calibration_profile_id: Optional[str] = Query(None, description="Apply calibration profile"),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Generate and return a standalone HTML report for a candidate."""
    job = session.get(Job, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(404, "Job not found")

    stmt = select(Candidate).where(
        Candidate.job_id == job_id,
        Candidate.index == candidate_index,
    )
    candidate = session.exec(stmt).first()
    if not candidate:
        raise HTTPException(404, "Candidate not found")

    metrics = dict(candidate.metrics_json or {})
    spec = job.spec_json or {}

    # Optionally apply calibration
    calibration_info = None
    if calibration_profile_id:
        profile = _resolve_calibration_profile(calibration_profile_id, user, session)
        if profile:
            metrics = apply_corrections_to_metrics(metrics, profile)
            calibration_info = {"name": profile.name, "id": profile.id}

    html = render_report_html(
        metrics=metrics,
        spec=spec,
        candidate_label=candidate.label or f"Candidate {candidate_index}",
        calibration_info=calibration_info,
    )
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Comparison report across candidates
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/comparison.html", response_class=HTMLResponse)
def comparison_html_report(
    job_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Generate an HTML comparison report across all candidates for a job."""
    job = session.get(Job, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(404, "Job not found")

    stmt = select(Candidate).where(Candidate.job_id == job_id).order_by(Candidate.index)
    candidates = session.exec(stmt).all()
    if not candidates:
        raise HTTPException(404, "No candidates found")

    cand_data = []
    for c in candidates:
        cand_data.append({
            "label": c.label or f"Candidate {c.index}",
            "metrics": c.metrics_json or {},
        })

    html = render_comparison_html(cand_data, job_id=job_id)
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# JSON comparison table
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/comparison")
def comparison_json(
    job_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Return a structured comparison table across candidates (JSON)."""
    job = session.get(Job, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(404, "Job not found")

    stmt = select(Candidate).where(Candidate.job_id == job_id).order_by(Candidate.index)
    candidates = session.exec(stmt).all()
    if not candidates:
        raise HTTPException(404, "No candidates found")

    cand_data = [
        {"label": c.label or f"Candidate {c.index}", "metrics": c.metrics_json or {}}
        for c in candidates
    ]

    table = generate_comparison_table(cand_data)
    return table


# ---------------------------------------------------------------------------
# Manufacturing compliance checklist
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/candidates/{candidate_index}/compliance")
def compliance_checklist(
    job_id: str,
    candidate_index: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Return the manufacturing compliance checklist for a candidate."""
    job = session.get(Job, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(404, "Job not found")

    stmt = select(Candidate).where(
        Candidate.job_id == job_id,
        Candidate.index == candidate_index,
    )
    candidate = session.exec(stmt).first()
    if not candidate:
        raise HTTPException(404, "Candidate not found")

    metrics = candidate.metrics_json or {}
    spec = job.spec_json or {}
    checklist = generate_compliance_checklist(metrics, spec)

    n_pass = sum(1 for c in checklist if c["status"] == "pass")
    n_warn = sum(1 for c in checklist if c["status"] == "warn")
    n_fail = sum(1 for c in checklist if c["status"] == "fail")

    return {
        "job_id": job_id,
        "candidate_index": candidate_index,
        "candidate_label": candidate.label,
        "summary": {
            "total": len(checklist),
            "pass": n_pass,
            "warn": n_warn,
            "fail": n_fail,
            "overall": "pass" if n_fail == 0 and n_warn == 0 else "warn" if n_fail == 0 else "fail",
        },
        "checks": checklist,
    }


# ---------------------------------------------------------------------------
# CFD handoff metadata
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/candidates/{candidate_index}/cfd-handoff")
def cfd_handoff(
    job_id: str,
    candidate_index: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Return CFD simulation handoff metadata for a candidate.

    Includes boundary conditions, fluid properties, mesh recommendations,
    and predicted results for comparison.
    """
    job = session.get(Job, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(404, "Job not found")

    stmt = select(Candidate).where(
        Candidate.job_id == job_id,
        Candidate.index == candidate_index,
    )
    candidate = session.exec(stmt).first()
    if not candidate:
        raise HTTPException(404, "Candidate not found")

    metrics = candidate.metrics_json or {}
    spec = job.spec_json or {}
    handoff = generate_cfd_handoff(metrics, spec)

    handoff["job_id"] = job_id
    handoff["candidate_index"] = candidate_index
    handoff["candidate_label"] = candidate.label
    handoff["artifacts"] = candidate.artifacts_json or {}

    return handoff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_calibration_profile(
    profile_id: str,
    user: User,
    session: Session,
) -> Optional[CalibrationProfile]:
    """Look up a calibration profile by ID (built-in or user-specific)."""
    from branchforge_core.calibration import list_default_profiles

    # Check built-in
    for p in list_default_profiles():
        if p.id == profile_id:
            return p

    # Check DB
    from ..models import CalibrationProfileDB
    db_profile = session.get(CalibrationProfileDB, profile_id)
    if db_profile and db_profile.user_id == user.id:
        return CalibrationProfile(
            id=db_profile.id,
            name=db_profile.name,
            friction_factor_multiplier=db_profile.friction_factor_multiplier,
            junction_K_multiplier=db_profile.junction_K_multiplier,
            heat_transfer_multiplier=db_profile.heat_transfer_multiplier,
            minor_loss_multiplier=db_profile.minor_loss_multiplier,
            r_squared_dp=db_profile.r_squared_dp,
            r_squared_thermal=db_profile.r_squared_thermal,
        )

    return None
