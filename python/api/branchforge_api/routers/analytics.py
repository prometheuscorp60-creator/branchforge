"""3E: API Rate Limiting + Usage Analytics."""

from __future__ import annotations

import uuid
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from pydantic import BaseModel
from sqlmodel import Session, select, col, func

from ..db import get_session
from ..models import User, UsageEvent, RateLimitBucket
from .auth import get_current_user

logger = logging.getLogger("branchforge.analytics")

router = APIRouter()


# ── Rate Limit Config ────────────────────────────────

RATE_LIMITS: Dict[str, Dict[str, int]] = {
    "free":       {"requests_per_minute": 10,  "requests_per_hour": 100},
    "pro":        {"requests_per_minute": 60,  "requests_per_hour": 1000},
    "team":       {"requests_per_minute": 120, "requests_per_hour": 5000},
    "enterprise": {"requests_per_minute": 300, "requests_per_hour": 50000},
}


# ── Rate Limiting ────────────────────────────────────

def check_rate_limit(user: User, session: Session):
    """Check and enforce per-user rate limits.

    Uses a database-backed sliding window (fallback for when Redis is unavailable).
    In production, this should be replaced with Redis-based token bucket.
    """
    limits = RATE_LIMITS.get(user.plan, RATE_LIMITS["free"])
    rpm_limit = limits["requests_per_minute"]

    bucket = session.exec(
        select(RateLimitBucket).where(RateLimitBucket.user_id == user.id)
    ).first()

    now = datetime.utcnow()

    if not bucket:
        bucket = RateLimitBucket(
            id=str(uuid.uuid4()),
            user_id=user.id,
            window_start=now,
            request_count=1,
        )
        session.add(bucket)
        session.commit()
        return

    window_age = (now - bucket.window_start).total_seconds()

    if window_age > 60:
        # Reset window
        bucket.window_start = now
        bucket.request_count = 1
    else:
        bucket.request_count += 1
        if bucket.request_count > rpm_limit:
            retry_after = int(60 - window_age)
            raise HTTPException(
                429,
                detail=f"Rate limit exceeded ({rpm_limit}/min for {user.plan} plan). "
                       f"Retry after {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )

    session.add(bucket)
    session.commit()


# ── Usage Event Recording ────────────────────────────

def record_usage_event(
    session: Session,
    user_id: str,
    event_type: str,
    metadata: Optional[dict] = None,
    org_id: Optional[str] = None,
):
    """Record a usage event for analytics."""
    event = UsageEvent(
        id=str(uuid.uuid4()),
        user_id=user_id,
        org_id=org_id,
        event_type=event_type,
        metadata_json=metadata,
    )
    session.add(event)
    session.commit()


# ── Schemas ──────────────────────────────────────────

class UsageSummary(BaseModel):
    period: str  # "day" | "week" | "month"
    total_events: int
    events_by_type: Dict[str, int]
    daily_breakdown: List[Dict[str, Any]]

class RateLimitStatus(BaseModel):
    plan: str
    requests_per_minute: int
    requests_per_hour: int
    current_minute_usage: int
    window_resets_in_seconds: int


# ── Endpoints ────────────────────────────────────────

@router.get("/analytics/usage", response_model=UsageSummary)
async def get_usage_analytics(
    period: str = Query("month", regex="^(day|week|month)$"),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    now = datetime.utcnow()
    if period == "day":
        since = now - timedelta(days=1)
    elif period == "week":
        since = now - timedelta(weeks=1)
    else:
        since = now - timedelta(days=30)

    # Fetch events
    owner_filter = UsageEvent.org_id == user.org_id if user.org_id else UsageEvent.user_id == user.id
    events = session.exec(
        select(UsageEvent)
        .where(owner_filter, UsageEvent.created_at >= since)
        .order_by(col(UsageEvent.created_at).asc())
    ).all()

    # Aggregate by type
    by_type: Dict[str, int] = {}
    for e in events:
        by_type[e.event_type] = by_type.get(e.event_type, 0) + 1

    # Daily breakdown
    daily: Dict[str, int] = {}
    for e in events:
        day_key = e.created_at.strftime("%Y-%m-%d")
        daily[day_key] = daily.get(day_key, 0) + 1

    daily_list = [{"date": k, "count": v} for k, v in sorted(daily.items())]

    return UsageSummary(
        period=period,
        total_events=len(events),
        events_by_type=by_type,
        daily_breakdown=daily_list,
    )


@router.get("/analytics/rate-limit", response_model=RateLimitStatus)
async def get_rate_limit_status(
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    limits = RATE_LIMITS.get(user.plan, RATE_LIMITS["free"])

    bucket = session.exec(
        select(RateLimitBucket).where(RateLimitBucket.user_id == user.id)
    ).first()

    current_usage = 0
    resets_in = 60

    if bucket:
        age = (datetime.utcnow() - bucket.window_start).total_seconds()
        if age < 60:
            current_usage = bucket.request_count
            resets_in = max(0, int(60 - age))

    return RateLimitStatus(
        plan=user.plan,
        requests_per_minute=limits["requests_per_minute"],
        requests_per_hour=limits["requests_per_hour"],
        current_minute_usage=current_usage,
        window_resets_in_seconds=resets_in,
    )


@router.get("/analytics/events")
async def list_recent_events(
    limit: int = Query(50, le=200),
    event_type: Optional[str] = None,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    stmt = select(UsageEvent).where(UsageEvent.user_id == user.id)
    if event_type:
        stmt = stmt.where(UsageEvent.event_type == event_type)
    stmt = stmt.order_by(col(UsageEvent.created_at).desc()).limit(limit)

    events = session.exec(stmt).all()
    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "created_at": e.created_at.isoformat(),
            "metadata": e.metadata_json,
        }
        for e in events
    ]
