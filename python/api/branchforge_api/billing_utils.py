"""Plan limits, quota enforcement, and Stripe helpers."""
from __future__ import annotations

from datetime import datetime, timedelta
from fastapi import HTTPException
from sqlmodel import Session

from branchforge_core.plan_limits import PLAN_LIMITS, get_plan_limits
from .models import User


def check_job_quota(user: User, session: Session) -> None:
    """Enforce job quota for the user's plan. Raises 402 if limit reached."""
    limits = get_plan_limits(user.plan)

    # Reset counter at month boundary
    if datetime.utcnow() - user.month_start > timedelta(days=30):
        user.month_start = datetime.utcnow()
        user.jobs_used = 0
        session.add(user)
        session.commit()

    if user.jobs_used >= limits["jobs_per_month"]:
        raise HTTPException(
            status_code=402,
            detail=f"{user.plan.title()} plan limit reached "
                   f"({limits['jobs_per_month']} jobs / 30 days). "
                   f"Upgrade your plan for more.",
        )


def increment_usage(user: User, session: Session) -> None:
    """Increment the user's job counter."""
    user.jobs_used += 1
    session.add(user)
    session.commit()


def can_export_step(user: User) -> bool:
    return get_plan_limits(user.plan)["step_export"]


def max_candidates(user: User) -> int:
    return get_plan_limits(user.plan)["candidates_per_job"]
