"""Parametric sweep engine for BranchForge.

Varies a single design parameter across a range and collects metrics
for each step, allowing engineers to map the design space efficiently.

Usage:
    sweep_spec = SweepSpec(
        parameter="fluid.target_deltaT_C",
        start=5.0, stop=20.0, n_steps=8,
    )
    results = run_sweep(base_job_spec, sweep_spec, outputs_root)
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import copy
import logging
import math
from pydantic import BaseModel, Field

from .schemas import JobSpec, CandidateMetrics

logger = logging.getLogger("branchforge.sweep")


# Sweepable parameters with their dotted paths and human labels
SWEEPABLE_PARAMS: Dict[str, Dict[str, Any]] = {
    "fluid.target_deltaT_C": {
        "label": "Target \u0394T (\u00B0C)",
        "min": 1.0, "max": 50.0, "default_start": 5.0, "default_stop": 20.0,
        "unit": "\u00B0C",
    },
    "generation.v_max_m_per_s": {
        "label": "Max velocity (m/s)",
        "min": 0.1, "max": 5.0, "default_start": 0.5, "default_stop": 3.0,
        "unit": "m/s",
    },
    "heatmap.total_watts": {
        "label": "Total heat load (W)",
        "min": 10.0, "max": 10000.0, "default_start": 200.0, "default_stop": 2000.0,
        "unit": "W",
    },
    "constraints.channel_depth_mm": {
        "label": "Channel depth (mm)",
        "min": 0.3, "max": 10.0, "default_start": 1.0, "default_stop": 5.0,
        "unit": "mm",
    },
    "constraints.min_channel_width_mm": {
        "label": "Min channel width (mm)",
        "min": 0.3, "max": 5.0, "default_start": 0.5, "default_stop": 3.0,
        "unit": "mm",
    },
    "constraints.max_channel_width_mm": {
        "label": "Max channel width (mm)",
        "min": 1.0, "max": 20.0, "default_start": 4.0, "default_stop": 12.0,
        "unit": "mm",
    },
    "fluid.inlet_temp_C": {
        "label": "Inlet temperature (\u00B0C)",
        "min": 5.0, "max": 80.0, "default_start": 15.0, "default_stop": 45.0,
        "unit": "\u00B0C",
    },
}


class SweepSpec(BaseModel):
    """Specification for a parametric sweep."""
    parameter: str  # dotted path into JobSpec
    start: float
    stop: float
    n_steps: int = 8


class SweepPoint(BaseModel):
    """Result at one sweep step."""
    param_value: float
    metrics: Optional[CandidateMetrics] = None
    error: Optional[str] = None


class SweepResult(BaseModel):
    """Complete sweep result."""
    parameter: str
    parameter_label: str = ""
    unit: str = ""
    values: List[float] = Field(default_factory=list)
    points: List[SweepPoint] = Field(default_factory=list)
    best_index: int = -1


def _set_nested(obj: dict, dotted_path: str, value: float):
    """Set a value in a nested dict using a dotted key path."""
    parts = dotted_path.split(".")
    d = obj
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _get_nested(obj: dict, dotted_path: str, default=None):
    """Get a value from a nested dict using a dotted key path."""
    parts = dotted_path.split(".")
    d = obj
    for part in parts:
        if isinstance(d, dict):
            d = d.get(part, default)
        else:
            return default
    return d


def generate_sweep_values(start: float, stop: float, n_steps: int) -> List[float]:
    """Generate evenly spaced parameter values."""
    if n_steps <= 1:
        return [start]
    step = (stop - start) / (n_steps - 1)
    return [round(start + i * step, 6) for i in range(n_steps)]


def run_sweep(
    base_spec_dict: dict,
    sweep_spec: SweepSpec,
    outputs_root: str,
    job_id_prefix: str = "sweep",
) -> SweepResult:
    """Execute a parametric sweep.

    For each step, generates ONE candidate with the varied parameter,
    collects its metrics, and returns the set for charting.
    """
    from .generator import generate_job

    param_info = SWEEPABLE_PARAMS.get(sweep_spec.parameter, {})
    result = SweepResult(
        parameter=sweep_spec.parameter,
        parameter_label=param_info.get("label", sweep_spec.parameter),
        unit=param_info.get("unit", ""),
    )

    values = generate_sweep_values(sweep_spec.start, sweep_spec.stop, sweep_spec.n_steps)
    result.values = values

    best_score = float("inf")

    for i, val in enumerate(values):
        logger.info("Sweep step %d/%d: %s = %.4f",
                    i + 1, len(values), sweep_spec.parameter, val)

        spec_dict = copy.deepcopy(base_spec_dict)
        _set_nested(spec_dict, sweep_spec.parameter, val)

        # Force single candidate per sweep step for speed
        if "generation" not in spec_dict:
            spec_dict["generation"] = {}
        spec_dict["generation"]["n_candidates"] = 1

        try:
            job_spec = JobSpec(**spec_dict)
            job_id = f"{job_id_prefix}_{i:02d}"
            candidates = generate_job(job_spec, outputs_root, job_id=job_id)

            if candidates:
                m = candidates[0].metrics
                result.points.append(SweepPoint(param_value=val, metrics=m))

                # Score by pressure drop (lower = better)
                if m.delta_p_kpa < best_score:
                    best_score = m.delta_p_kpa
                    result.best_index = i
            else:
                result.points.append(SweepPoint(
                    param_value=val, error="No candidates produced"))

        except Exception as e:
            logger.error("Sweep step %d failed: %s", i, e)
            result.points.append(SweepPoint(param_value=val, error=str(e)))

    return result
