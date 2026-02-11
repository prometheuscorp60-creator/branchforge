"""Calibration API: manage calibration profiles and import CFD/experimental data."""

from __future__ import annotations

import json
import uuid
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from ..database import get_session
from ..auth import get_current_user
from ..models import User, CalibrationProfileDB, Candidate

from branchforge_core.calibration import (
    CalibrationPoint,
    CalibrationProfile,
    build_profile_from_points,
    get_default_profile,
    list_default_profiles,
    apply_corrections_to_metrics,
    calibration_confidence,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CalibrationPointIn(BaseModel):
    metric: str = Field(..., description="dp_kpa | uniformity_std | max_temp_C")
    predicted: float
    measured: float
    job_id: Optional[str] = None
    candidate_index: Optional[int] = None
    notes: str = ""


class CalibrationProfileOut(BaseModel):
    id: str
    name: str
    description: str
    process_preset: str
    coolant: str
    friction_factor_multiplier: float
    junction_K_multiplier: float
    heat_transfer_multiplier: float
    minor_loss_multiplier: float
    n_calibration_points: int
    r_squared_dp: float
    r_squared_thermal: float
    fit_rmse_dp_kpa: float
    fit_rmse_thermal_C: float
    active: bool = True
    confidence: Optional[dict] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CalibrateFromJobRequest(BaseModel):
    """Import calibration data from a completed job's candidates."""
    job_id: str
    candidate_index: int
    measured_dp_kpa: Optional[float] = None
    measured_uniformity_std: Optional[float] = None
    measured_max_temp_C: Optional[float] = None
    notes: str = ""


class CalibrationFitRequest(BaseModel):
    """Request to fit a new profile from accumulated points."""
    name: str = "Custom Profile"
    process_preset: str = "CNC"
    coolant: str = "water"
    points: List[CalibrationPointIn] = Field(default_factory=list)


class CalibratedMetricsRequest(BaseModel):
    """Request to apply calibration corrections to metrics."""
    profile_id: str
    metrics: dict


# ---------------------------------------------------------------------------
# Default profiles (built-in, read-only)
# ---------------------------------------------------------------------------

@router.get("/calibration/defaults", response_model=List[CalibrationProfileOut])
def list_defaults():
    """Return all built-in calibration profiles."""
    defaults = list_default_profiles()
    return [
        CalibrationProfileOut(
            id=p.id,
            name=p.name,
            description=p.description,
            process_preset=p.process_preset,
            coolant=p.coolant,
            friction_factor_multiplier=p.friction_factor_multiplier,
            junction_K_multiplier=p.junction_K_multiplier,
            heat_transfer_multiplier=p.heat_transfer_multiplier,
            minor_loss_multiplier=p.minor_loss_multiplier,
            n_calibration_points=0,
            r_squared_dp=0.0,
            r_squared_thermal=0.0,
            fit_rmse_dp_kpa=0.0,
            fit_rmse_thermal_C=0.0,
            confidence=calibration_confidence(p),
        )
        for p in defaults
    ]


# ---------------------------------------------------------------------------
# User profiles CRUD
# ---------------------------------------------------------------------------

@router.get("/calibration/profiles", response_model=List[CalibrationProfileOut])
def list_user_profiles(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """List all calibration profiles for the current user (and their org)."""
    stmt = select(CalibrationProfileDB).where(
        (CalibrationProfileDB.user_id == user.id) |
        (CalibrationProfileDB.org_id == user.org_id)
    ).order_by(CalibrationProfileDB.created_at.desc())
    profiles = session.exec(stmt).all()

    results = []
    for p in profiles:
        cp = _db_to_core(p)
        results.append(CalibrationProfileOut(
            id=p.id,
            name=p.name,
            description=p.description,
            process_preset=p.process_preset,
            coolant=p.coolant,
            friction_factor_multiplier=p.friction_factor_multiplier,
            junction_K_multiplier=p.junction_K_multiplier,
            heat_transfer_multiplier=p.heat_transfer_multiplier,
            minor_loss_multiplier=p.minor_loss_multiplier,
            n_calibration_points=p.n_calibration_points,
            r_squared_dp=p.r_squared_dp,
            r_squared_thermal=p.r_squared_thermal,
            fit_rmse_dp_kpa=p.fit_rmse_dp_kpa,
            fit_rmse_thermal_C=p.fit_rmse_thermal_C,
            active=p.active,
            confidence=calibration_confidence(cp),
            created_at=p.created_at.isoformat() if p.created_at else None,
            updated_at=p.updated_at.isoformat() if p.updated_at else None,
        ))
    return results


@router.get("/calibration/profiles/{profile_id}", response_model=CalibrationProfileOut)
def get_profile(
    profile_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Get a single calibration profile by ID."""
    # Check built-in first
    for p in list_default_profiles():
        if p.id == profile_id:
            return CalibrationProfileOut(
                id=p.id, name=p.name, description=p.description,
                process_preset=p.process_preset, coolant=p.coolant,
                friction_factor_multiplier=p.friction_factor_multiplier,
                junction_K_multiplier=p.junction_K_multiplier,
                heat_transfer_multiplier=p.heat_transfer_multiplier,
                minor_loss_multiplier=p.minor_loss_multiplier,
                n_calibration_points=0, r_squared_dp=0.0, r_squared_thermal=0.0,
                fit_rmse_dp_kpa=0.0, fit_rmse_thermal_C=0.0,
                confidence=calibration_confidence(p),
            )

    db_profile = session.get(CalibrationProfileDB, profile_id)
    if not db_profile or db_profile.user_id != user.id:
        raise HTTPException(404, "Profile not found")

    cp = _db_to_core(db_profile)
    return CalibrationProfileOut(
        id=db_profile.id, name=db_profile.name, description=db_profile.description,
        process_preset=db_profile.process_preset, coolant=db_profile.coolant,
        friction_factor_multiplier=db_profile.friction_factor_multiplier,
        junction_K_multiplier=db_profile.junction_K_multiplier,
        heat_transfer_multiplier=db_profile.heat_transfer_multiplier,
        minor_loss_multiplier=db_profile.minor_loss_multiplier,
        n_calibration_points=db_profile.n_calibration_points,
        r_squared_dp=db_profile.r_squared_dp,
        r_squared_thermal=db_profile.r_squared_thermal,
        fit_rmse_dp_kpa=db_profile.fit_rmse_dp_kpa,
        fit_rmse_thermal_C=db_profile.fit_rmse_thermal_C,
        active=db_profile.active,
        confidence=calibration_confidence(cp),
        created_at=db_profile.created_at.isoformat() if db_profile.created_at else None,
        updated_at=db_profile.updated_at.isoformat() if db_profile.updated_at else None,
    )


@router.delete("/calibration/profiles/{profile_id}")
def delete_profile(
    profile_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Delete a user's calibration profile."""
    db_profile = session.get(CalibrationProfileDB, profile_id)
    if not db_profile or db_profile.user_id != user.id:
        raise HTTPException(404, "Profile not found")
    session.delete(db_profile)
    session.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Import calibration data from a completed job
# ---------------------------------------------------------------------------

@router.post("/calibration/import-from-job", response_model=CalibrationProfileOut)
def import_from_job(
    req: CalibrateFromJobRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Import measured data from a completed job's candidate.

    Compares the measured ΔP / thermal values against the candidate's
    predicted metrics, and creates or updates a calibration profile.
    """
    # Find the candidate
    stmt = select(Candidate).where(
        Candidate.job_id == req.job_id,
        Candidate.index == req.candidate_index,
    )
    candidate = session.exec(stmt).first()
    if not candidate:
        raise HTTPException(404, "Candidate not found")

    metrics = candidate.metrics_json or {}
    points: List[CalibrationPoint] = []

    if req.measured_dp_kpa is not None and "delta_p_kpa" in metrics:
        points.append(CalibrationPoint(
            metric="dp_kpa",
            predicted=metrics["delta_p_kpa"],
            measured=req.measured_dp_kpa,
            job_id=req.job_id,
            candidate_index=req.candidate_index,
            notes=req.notes,
        ))

    if req.measured_uniformity_std is not None and "uniformity_deltaT_C_std" in metrics:
        points.append(CalibrationPoint(
            metric="uniformity_std",
            predicted=metrics["uniformity_deltaT_C_std"],
            measured=req.measured_uniformity_std,
            job_id=req.job_id,
            candidate_index=req.candidate_index,
            notes=req.notes,
        ))

    if req.measured_max_temp_C is not None and "uniformity_deltaT_C_max" in metrics:
        points.append(CalibrationPoint(
            metric="max_temp_C",
            predicted=metrics["uniformity_deltaT_C_max"],
            measured=req.measured_max_temp_C,
            job_id=req.job_id,
            candidate_index=req.candidate_index,
            notes=req.notes,
        ))

    if not points:
        raise HTTPException(400, "No calibration data provided")

    # Determine process preset from job spec
    job_spec = {}
    from ..models import Job
    job = session.get(Job, req.job_id)
    if job:
        job_spec = job.spec_json or {}
    process_preset = (job_spec.get("constraints", {}).get("process_preset", "CNC"))
    coolant = (job_spec.get("fluid", {}).get("coolant", "water"))

    # Find existing profile or create new
    existing = _find_active_profile(session, user.id, process_preset, coolant)

    # Merge points with existing
    all_points = points
    if existing and existing.points_json:
        try:
            old_points = [
                CalibrationPoint(**p)
                for p in json.loads(existing.points_json)
            ]
            all_points = old_points + points
        except Exception:
            pass

    # Fit
    profile = build_profile_from_points(
        all_points,
        name=f"{process_preset}/{coolant} (calibrated)",
        process_preset=process_preset,
        coolant=coolant,
    )

    # Persist
    if existing:
        existing.friction_factor_multiplier = profile.friction_factor_multiplier
        existing.junction_K_multiplier = profile.junction_K_multiplier
        existing.heat_transfer_multiplier = profile.heat_transfer_multiplier
        existing.minor_loss_multiplier = profile.minor_loss_multiplier
        existing.n_calibration_points = profile.n_calibration_points
        existing.r_squared_dp = profile.r_squared_dp
        existing.r_squared_thermal = profile.r_squared_thermal
        existing.fit_rmse_dp_kpa = profile.fit_rmse_dp_kpa
        existing.fit_rmse_thermal_C = profile.fit_rmse_thermal_C
        existing.points_json = profile.points_json
        existing.updated_at = datetime.utcnow()
        existing.name = profile.name
        session.add(existing)
        session.commit()
        session.refresh(existing)
        db_profile = existing
    else:
        db_profile = CalibrationProfileDB(
            id=str(uuid.uuid4()),
            user_id=user.id,
            org_id=user.org_id,
            name=profile.name,
            description=profile.description,
            process_preset=process_preset,
            coolant=coolant,
            friction_factor_multiplier=profile.friction_factor_multiplier,
            junction_K_multiplier=profile.junction_K_multiplier,
            heat_transfer_multiplier=profile.heat_transfer_multiplier,
            minor_loss_multiplier=profile.minor_loss_multiplier,
            n_calibration_points=profile.n_calibration_points,
            r_squared_dp=profile.r_squared_dp,
            r_squared_thermal=profile.r_squared_thermal,
            fit_rmse_dp_kpa=profile.fit_rmse_dp_kpa,
            fit_rmse_thermal_C=profile.fit_rmse_thermal_C,
            points_json=profile.points_json,
        )
        session.add(db_profile)
        session.commit()
        session.refresh(db_profile)

    logger.info("Calibration import: profile=%s points=%d r2_dp=%.3f",
                db_profile.id, db_profile.n_calibration_points, db_profile.r_squared_dp)

    cp = _db_to_core(db_profile)
    return CalibrationProfileOut(
        id=db_profile.id, name=db_profile.name, description=db_profile.description,
        process_preset=db_profile.process_preset, coolant=db_profile.coolant,
        friction_factor_multiplier=db_profile.friction_factor_multiplier,
        junction_K_multiplier=db_profile.junction_K_multiplier,
        heat_transfer_multiplier=db_profile.heat_transfer_multiplier,
        minor_loss_multiplier=db_profile.minor_loss_multiplier,
        n_calibration_points=db_profile.n_calibration_points,
        r_squared_dp=db_profile.r_squared_dp,
        r_squared_thermal=db_profile.r_squared_thermal,
        fit_rmse_dp_kpa=db_profile.fit_rmse_dp_kpa,
        fit_rmse_thermal_C=db_profile.fit_rmse_thermal_C,
        active=db_profile.active,
        confidence=calibration_confidence(cp),
        created_at=db_profile.created_at.isoformat() if db_profile.created_at else None,
        updated_at=db_profile.updated_at.isoformat() if db_profile.updated_at else None,
    )


# ---------------------------------------------------------------------------
# Manual calibration fit
# ---------------------------------------------------------------------------

@router.post("/calibration/fit", response_model=CalibrationProfileOut)
def fit_calibration(
    req: CalibrationFitRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Fit a calibration profile from manually supplied data points."""
    if len(req.points) < 2:
        raise HTTPException(400, "At least 2 calibration points are required")

    core_points = [
        CalibrationPoint(
            metric=p.metric,
            predicted=p.predicted,
            measured=p.measured,
            job_id=p.job_id,
            candidate_index=p.candidate_index,
            notes=p.notes,
        )
        for p in req.points
    ]

    profile = build_profile_from_points(
        core_points,
        name=req.name,
        process_preset=req.process_preset,
        coolant=req.coolant,
    )

    db_profile = CalibrationProfileDB(
        id=str(uuid.uuid4()),
        user_id=user.id,
        org_id=user.org_id,
        name=profile.name,
        description=profile.description,
        process_preset=req.process_preset,
        coolant=req.coolant,
        friction_factor_multiplier=profile.friction_factor_multiplier,
        junction_K_multiplier=profile.junction_K_multiplier,
        heat_transfer_multiplier=profile.heat_transfer_multiplier,
        minor_loss_multiplier=profile.minor_loss_multiplier,
        n_calibration_points=profile.n_calibration_points,
        r_squared_dp=profile.r_squared_dp,
        r_squared_thermal=profile.r_squared_thermal,
        fit_rmse_dp_kpa=profile.fit_rmse_dp_kpa,
        fit_rmse_thermal_C=profile.fit_rmse_thermal_C,
        points_json=profile.points_json,
    )
    session.add(db_profile)
    session.commit()
    session.refresh(db_profile)

    cp = _db_to_core(db_profile)
    return CalibrationProfileOut(
        id=db_profile.id, name=db_profile.name, description=db_profile.description,
        process_preset=db_profile.process_preset, coolant=db_profile.coolant,
        friction_factor_multiplier=db_profile.friction_factor_multiplier,
        junction_K_multiplier=db_profile.junction_K_multiplier,
        heat_transfer_multiplier=db_profile.heat_transfer_multiplier,
        minor_loss_multiplier=db_profile.minor_loss_multiplier,
        n_calibration_points=db_profile.n_calibration_points,
        r_squared_dp=db_profile.r_squared_dp,
        r_squared_thermal=db_profile.r_squared_thermal,
        fit_rmse_dp_kpa=db_profile.fit_rmse_dp_kpa,
        fit_rmse_thermal_C=db_profile.fit_rmse_thermal_C,
        active=db_profile.active,
        confidence=calibration_confidence(cp),
        created_at=db_profile.created_at.isoformat() if db_profile.created_at else None,
        updated_at=db_profile.updated_at.isoformat() if db_profile.updated_at else None,
    )


# ---------------------------------------------------------------------------
# Apply calibration to metrics
# ---------------------------------------------------------------------------

@router.post("/calibration/apply")
def apply_calibration(
    req: CalibratedMetricsRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Apply a calibration profile to a set of candidate metrics.

    Returns the corrected metrics with calibration metadata.
    """
    # Try built-in first
    profile = None
    for p in list_default_profiles():
        if p.id == req.profile_id:
            profile = p
            break

    if profile is None:
        db_profile = session.get(CalibrationProfileDB, req.profile_id)
        if not db_profile:
            raise HTTPException(404, "Profile not found")
        profile = _db_to_core(db_profile)

    corrected = apply_corrections_to_metrics(req.metrics, profile)
    return corrected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_active_profile(
    session: Session,
    user_id: str,
    process_preset: str,
    coolant: str,
) -> Optional[CalibrationProfileDB]:
    """Find the active calibration profile for a user+process+coolant combo."""
    stmt = select(CalibrationProfileDB).where(
        CalibrationProfileDB.user_id == user_id,
        CalibrationProfileDB.process_preset == process_preset,
        CalibrationProfileDB.coolant == coolant,
        CalibrationProfileDB.active == True,
    ).order_by(CalibrationProfileDB.updated_at.desc())
    return session.exec(stmt).first()


def _db_to_core(db: CalibrationProfileDB) -> CalibrationProfile:
    """Convert a DB model to a core CalibrationProfile."""
    return CalibrationProfile(
        id=db.id,
        name=db.name,
        description=db.description,
        process_preset=db.process_preset,
        coolant=db.coolant,
        friction_factor_multiplier=db.friction_factor_multiplier,
        junction_K_multiplier=db.junction_K_multiplier,
        heat_transfer_multiplier=db.heat_transfer_multiplier,
        minor_loss_multiplier=db.minor_loss_multiplier,
        n_calibration_points=db.n_calibration_points,
        r_squared_dp=db.r_squared_dp,
        r_squared_thermal=db.r_squared_thermal,
        fit_rmse_dp_kpa=db.fit_rmse_dp_kpa,
        fit_rmse_thermal_C=db.fit_rmse_thermal_C,
        points_json=db.points_json,
        user_id=db.user_id,
        org_id=db.org_id or "",
    )
