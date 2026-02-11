"""Plan limits shared between API and worker.

Single source of truth for per-plan resource constraints.
Import from here instead of defining limits in multiple places.
"""
from __future__ import annotations


PLAN_LIMITS: dict[str, dict] = {
    "free": {
        "jobs_per_month": 3,
        "candidates_per_job": 5,
        "step_export": False,
        "priority_queue": False,
        "pdf_reports": False,
        "max_candidate_seconds": 120,
    },
    "pro": {
        "jobs_per_month": 50,
        "candidates_per_job": 20,
        "step_export": True,
        "priority_queue": True,
        "pdf_reports": True,
        "max_candidate_seconds": 300,
    },
    "team": {
        "jobs_per_month": 200,
        "candidates_per_job": 999,
        "step_export": True,
        "priority_queue": True,
        "pdf_reports": True,
        "max_candidate_seconds": 600,
    },
    "enterprise": {
        "jobs_per_month": 999999,
        "candidates_per_job": 999,
        "step_export": True,
        "priority_queue": True,
        "pdf_reports": True,
        "max_candidate_seconds": 900,
    },
}


def get_plan_limits(plan: str) -> dict:
    """Return limits for the given plan, falling back to free."""
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])


def get_queue_name(plan: str) -> str:
    """Return the RQ queue name for the given plan's priority level."""
    if plan in ("enterprise", "team"):
        return "branchforge-high"
    if plan == "pro":
        return "branchforge"
    return "branchforge-low"
