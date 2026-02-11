"""
Calibration harness for BranchForge hydraulic predictions.

Engineers run CFD or experiments and import results; BranchForge learns
correction factors so that preview metrics (ΔP, temperature uniformity)
match reality for each shop's process capabilities.

Key concepts:
  - CalibrationPoint: one (predicted, measured) pair for a metric
  - CalibrationProfile: a named collection of correction factors
  - Fitting: least-squares regression from calibration points to factors

Supported correction targets:
  1. friction_factor_multiplier — scales Darcy-Weisbach friction
  2. junction_K_multiplier — scales Idelchik junction K-factors
  3. heat_transfer_multiplier — scales thermal predictions (Nusselt / HTC)
  4. minor_loss_multiplier — scales contraction/expansion losses
"""

from __future__ import annotations

import json
import math
import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationPoint:
    """One measured vs predicted pair for a calibration metric."""
    metric: str              # "dp_kpa" | "uniformity_std" | "max_temp_C"
    predicted: float         # BranchForge's estimate
    measured: float          # CFD or experimental result
    job_id: Optional[str] = None
    candidate_index: Optional[int] = None
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


@dataclass
class CalibrationProfile:
    """Named set of correction factors derived from calibration data.

    The factors multiply the corresponding physical model terms:
      corrected_dp = predicted_dp * friction_factor_multiplier
      corrected_junction = predicted_junction * junction_K_multiplier
      corrected_htc = predicted_htc * heat_transfer_multiplier
      corrected_minor = predicted_minor * minor_loss_multiplier
    """
    id: str = ""
    name: str = "Default"
    description: str = ""
    process_preset: str = "CNC"     # CNC | AM | ETCHED_LID
    coolant: str = "water"

    # Correction factors (1.0 = no correction)
    friction_factor_multiplier: float = 1.0
    junction_K_multiplier: float = 1.0
    heat_transfer_multiplier: float = 1.0
    minor_loss_multiplier: float = 1.0

    # Confidence & metadata
    n_calibration_points: int = 0
    r_squared_dp: float = 0.0          # goodness of fit for ΔP
    r_squared_thermal: float = 0.0     # goodness of fit for thermal
    fit_rmse_dp_kpa: float = 0.0       # absolute RMSE on ΔP
    fit_rmse_thermal_C: float = 0.0    # absolute RMSE on temperature

    # Point history (serialised for DB storage)
    points_json: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    user_id: str = ""
    org_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationProfile":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in allowed}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Default calibration profiles (industry-standard corrections)
# ---------------------------------------------------------------------------

_DEFAULT_PROFILES: Dict[str, CalibrationProfile] = {
    "CNC_water": CalibrationProfile(
        id="builtin_cnc_water",
        name="CNC Milled / Water",
        description=(
            "Baseline correction for CNC-milled aluminium cold plates with "
            "water coolant.  Friction factor multiplier accounts for typical "
            "surface roughness (Ra ~1.6 μm).  Junction K scaled for "
            "sharp-edged rectangular channels."
        ),
        process_preset="CNC",
        coolant="water",
        friction_factor_multiplier=1.15,   # CNC finish adds ~15% friction
        junction_K_multiplier=1.10,        # sharp edges add ~10% junction loss
        heat_transfer_multiplier=1.05,     # roughness slightly enhances HTC
        minor_loss_multiplier=1.10,
    ),
    "CNC_glycol": CalibrationProfile(
        id="builtin_cnc_glycol",
        name="CNC Milled / Glycol Mix",
        description=(
            "CNC aluminium with ethylene glycol mixture.  Higher viscosity "
            "means laminar flow dominates; friction factor needs less "
            "correction but junction losses increase."
        ),
        process_preset="CNC",
        coolant="ethylene_glycol_50",
        friction_factor_multiplier=1.08,
        junction_K_multiplier=1.20,
        heat_transfer_multiplier=0.95,     # glycol has lower HTC
        minor_loss_multiplier=1.15,
    ),
    "AM_water": CalibrationProfile(
        id="builtin_am_water",
        name="Additive (DMLS) / Water",
        description=(
            "Metal 3D-printed (DMLS/SLM) cold plates.  Surface roughness "
            "is much higher (~Ra 8-15 μm), significantly increasing friction.  "
            "Complex internal geometry reduces junction losses relative to "
            "sharp-edged CNC channels."
        ),
        process_preset="AM",
        coolant="water",
        friction_factor_multiplier=1.45,   # rough AM surfaces
        junction_K_multiplier=0.85,        # smoother transitions in AM
        heat_transfer_multiplier=1.20,     # roughness enhances turbulence & HTC
        minor_loss_multiplier=1.25,
    ),
    "ETCHED_water": CalibrationProfile(
        id="builtin_etched_water",
        name="Etched Lid / Water",
        description=(
            "Chemical-etched microchannel lids.  Very smooth surfaces but "
            "shallow channels with high aspect ratios.  Entrance effects "
            "dominate."
        ),
        process_preset="ETCHED_LID",
        coolant="water",
        friction_factor_multiplier=1.02,   # very smooth
        junction_K_multiplier=1.30,        # abrupt transitions in etched geometry
        heat_transfer_multiplier=0.90,     # developing flow reduces effective HTC
        minor_loss_multiplier=1.35,
    ),
}


def get_default_profile(process_preset: str, coolant: str = "water") -> CalibrationProfile:
    """Return the best matching built-in calibration profile.

    Falls back to CNC/water if no exact match exists.
    """
    key = f"{process_preset}_{coolant}"
    if key in _DEFAULT_PROFILES:
        return _DEFAULT_PROFILES[key]

    # Try matching just the process preset
    for k, v in _DEFAULT_PROFILES.items():
        if k.startswith(process_preset + "_"):
            return v

    # Fall back to CNC/water
    return _DEFAULT_PROFILES.get("CNC_water", CalibrationProfile())


def list_default_profiles() -> List[CalibrationProfile]:
    """Return all built-in calibration profiles."""
    return list(_DEFAULT_PROFILES.values())


# ---------------------------------------------------------------------------
# Fitting engine
# ---------------------------------------------------------------------------

def fit_dp_calibration(
    points: List[CalibrationPoint],
) -> Tuple[float, float, float]:
    """Fit a friction-factor multiplier from ΔP calibration data.

    Uses ordinary least squares: measured_dp ≈ k * predicted_dp.
    The multiplier k is constrained to [0.5, 3.0] for physical sanity.

    Parameters
    ----------
    points : list[CalibrationPoint]
        Each point has .predicted and .measured for metric="dp_kpa"

    Returns
    -------
    (multiplier, r_squared, rmse_kpa) : tuple
    """
    dp_points = [p for p in points if p.metric == "dp_kpa"
                 and p.predicted > 0 and p.measured > 0]

    if len(dp_points) < 2:
        return (1.0, 0.0, 0.0)

    pred = np.array([p.predicted for p in dp_points])
    meas = np.array([p.measured for p in dp_points])

    # OLS through origin: meas = k * pred
    # k = sum(pred * meas) / sum(pred^2)
    k = float(np.sum(pred * meas) / max(np.sum(pred ** 2), 1e-12))
    k = max(0.5, min(3.0, k))

    # R² (coefficient of determination)
    fitted = k * pred
    ss_res = float(np.sum((meas - fitted) ** 2))
    ss_tot = float(np.sum((meas - np.mean(meas)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12) if ss_tot > 1e-12 else 0.0
    r2 = max(0.0, r2)

    rmse = float(np.sqrt(np.mean((meas - fitted) ** 2)))

    return (k, r2, rmse)


def fit_thermal_calibration(
    points: List[CalibrationPoint],
) -> Tuple[float, float, float]:
    """Fit a heat-transfer multiplier from temperature calibration data.

    Uses the same OLS approach: measured_deltaT ≈ k * predicted_deltaT.
    A multiplier < 1 means the real HTC is higher than predicted (better
    cooling); > 1 means worse cooling.

    Parameters
    ----------
    points : list[CalibrationPoint]
        Each point has .predicted and .measured for metric in
        ("uniformity_std", "max_temp_C").

    Returns
    -------
    (multiplier, r_squared, rmse_C) : tuple
    """
    thermal_points = [
        p for p in points
        if p.metric in ("uniformity_std", "max_temp_C")
        and p.predicted > 0 and p.measured > 0
    ]

    if len(thermal_points) < 2:
        return (1.0, 0.0, 0.0)

    pred = np.array([p.predicted for p in thermal_points])
    meas = np.array([p.measured for p in thermal_points])

    k = float(np.sum(pred * meas) / max(np.sum(pred ** 2), 1e-12))
    k = max(0.5, min(3.0, k))

    fitted = k * pred
    ss_res = float(np.sum((meas - fitted) ** 2))
    ss_tot = float(np.sum((meas - np.mean(meas)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12) if ss_tot > 1e-12 else 0.0
    r2 = max(0.0, r2)

    rmse = float(np.sqrt(np.mean((meas - fitted) ** 2)))

    return (k, r2, rmse)


def fit_junction_calibration(
    points: List[CalibrationPoint],
) -> float:
    """Estimate junction K multiplier from ΔP data.

    If friction-corrected ΔP still under-predicts measured ΔP, the
    residual is attributed to junction losses.  This is a heuristic
    split: we assume 30% of total ΔP comes from junctions (typical
    for branching networks with 3-5 bifurcation levels).

    Returns junction_K_multiplier in [0.5, 3.0].
    """
    dp_points = [p for p in points if p.metric == "dp_kpa"
                 and p.predicted > 0 and p.measured > 0]

    if len(dp_points) < 3:
        return 1.0

    pred = np.array([p.predicted for p in dp_points])
    meas = np.array([p.measured for p in dp_points])

    # Overall ratio
    overall_ratio = float(np.mean(meas / pred))

    # If predictions are already close, junction correction is 1.0
    if abs(overall_ratio - 1.0) < 0.05:
        return 1.0

    # Assume 30% of pressure drop is from junctions
    junction_fraction = 0.30
    friction_fraction = 1.0 - junction_fraction

    # If overall_ratio > 1.0: under-predicting → increase junction K
    # Attribute residual proportionally
    junction_multiplier = 1.0 + (overall_ratio - 1.0) / junction_fraction
    return max(0.5, min(3.0, junction_multiplier))


def build_profile_from_points(
    points: List[CalibrationPoint],
    name: str = "Custom",
    process_preset: str = "CNC",
    coolant: str = "water",
    base_profile: Optional[CalibrationProfile] = None,
) -> CalibrationProfile:
    """Build a complete calibration profile from a set of calibration points.

    If a base_profile is provided, corrections are applied on top of
    the base factors.  Otherwise starts from the default profile for
    the given process_preset + coolant.
    """
    if base_profile is None:
        base_profile = get_default_profile(process_preset, coolant)

    # Fit ΔP
    dp_mult, dp_r2, dp_rmse = fit_dp_calibration(points)

    # Fit thermal
    htc_mult, thermal_r2, thermal_rmse = fit_thermal_calibration(points)

    # Fit junction K (from ΔP residuals)
    junc_mult = fit_junction_calibration(points)

    # Apply corrections on top of base
    profile = CalibrationProfile(
        name=name,
        description=f"Calibrated from {len(points)} data points.",
        process_preset=process_preset,
        coolant=coolant,
        friction_factor_multiplier=base_profile.friction_factor_multiplier * dp_mult,
        junction_K_multiplier=base_profile.junction_K_multiplier * junc_mult,
        heat_transfer_multiplier=base_profile.heat_transfer_multiplier * htc_mult,
        minor_loss_multiplier=base_profile.minor_loss_multiplier * dp_mult,
        n_calibration_points=len(points),
        r_squared_dp=dp_r2,
        r_squared_thermal=thermal_r2,
        fit_rmse_dp_kpa=dp_rmse,
        fit_rmse_thermal_C=thermal_rmse,
        points_json=json.dumps([asdict(p) for p in points]),
        updated_at=datetime.datetime.utcnow().isoformat(),
    )

    return profile


# ---------------------------------------------------------------------------
# Apply corrections to predictions
# ---------------------------------------------------------------------------

def apply_dp_correction(
    predicted_dp_kpa: float,
    profile: CalibrationProfile,
) -> float:
    """Apply calibration corrections to a predicted pressure drop.

    The total ΔP is split into friction (70%) and junction/minor (30%)
    components, and each is scaled by its respective multiplier.
    """
    junction_fraction = 0.30
    friction_fraction = 1.0 - junction_fraction

    dp_friction = predicted_dp_kpa * friction_fraction * profile.friction_factor_multiplier
    dp_junction = predicted_dp_kpa * junction_fraction * profile.junction_K_multiplier
    dp_minor = predicted_dp_kpa * 0.0  # already included in junction fraction

    return dp_friction + dp_junction


def apply_thermal_correction(
    predicted_deltaT: float,
    profile: CalibrationProfile,
) -> float:
    """Apply calibration correction to a predicted temperature metric.

    A heat_transfer_multiplier > 1 means better-than-predicted heat
    transfer, so the temperature rise is *reduced*.
    """
    if profile.heat_transfer_multiplier <= 0:
        return predicted_deltaT
    return predicted_deltaT / profile.heat_transfer_multiplier


def apply_corrections_to_metrics(
    metrics: Dict[str, Any],
    profile: CalibrationProfile,
) -> Dict[str, Any]:
    """Apply calibration corrections to a full metrics dictionary.

    Returns a new dict with corrected values and calibration metadata.
    """
    corrected = dict(metrics)

    # Pressure drop correction
    if "delta_p_kpa" in metrics:
        corrected["delta_p_kpa_uncalibrated"] = metrics["delta_p_kpa"]
        corrected["delta_p_kpa"] = round(
            apply_dp_correction(metrics["delta_p_kpa"], profile), 3
        )

    # Temperature uniformity correction
    for key in ("uniformity_deltaT_C_std", "uniformity_deltaT_C_max"):
        if key in metrics:
            corrected[f"{key}_uncalibrated"] = metrics[key]
            corrected[key] = round(
                apply_thermal_correction(metrics[key], profile), 4
            )

    # Add calibration metadata
    corrected["calibration_profile_id"] = profile.id
    corrected["calibration_profile_name"] = profile.name
    corrected["calibration_applied"] = True
    corrected["calibration_r2_dp"] = profile.r_squared_dp
    corrected["calibration_r2_thermal"] = profile.r_squared_thermal

    return corrected


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def calibration_confidence(profile: CalibrationProfile) -> Dict[str, Any]:
    """Assess the confidence level of a calibration profile.

    Returns a dict with overall confidence (0-100), and per-metric details.
    """
    n = profile.n_calibration_points

    # Points-based confidence (diminishing returns: 3-5 points = good, 10+ = great)
    points_score = min(100, n * 20) if n < 5 else min(100, 80 + (n - 5) * 4)

    # R² based confidence
    dp_r2_score = int(profile.r_squared_dp * 100)
    thermal_r2_score = int(profile.r_squared_thermal * 100)

    # Overall
    if n == 0:
        overall = 30  # built-in defaults have moderate confidence
    else:
        overall = int(
            0.3 * points_score +
            0.4 * dp_r2_score +
            0.3 * thermal_r2_score
        )

    tier = (
        "high" if overall >= 75 else
        "moderate" if overall >= 50 else
        "low" if overall >= 25 else
        "uncalibrated"
    )

    return {
        "overall_score": overall,
        "tier": tier,
        "points_score": points_score,
        "dp_r2_score": dp_r2_score,
        "thermal_r2_score": thermal_r2_score,
        "n_points": n,
        "recommendation": (
            "Profile is well-calibrated. Predictions should be within ±10%."
            if tier == "high" else
            "Profile has moderate calibration. Consider adding more data points."
            if tier == "moderate" else
            "Profile needs more calibration data. Predictions may differ ±30% from reality."
            if tier == "low" else
            "Using built-in defaults. Run CFD/experiments and import results to improve accuracy."
        ),
    }
