from __future__ import annotations

from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Column, JSON


class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    email: str = Field(index=True, unique=True)
    api_key: str = Field(index=True, unique=True)
    plan: str = Field(default="free", index=True)  # free|pro|team|enterprise
    month_start: datetime = Field(default_factory=datetime.utcnow)
    jobs_used: int = Field(default=0)

    # OAuth fields
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    oauth_provider: Optional[str] = None  # google|github|None (email+password)
    oauth_id: Optional[str] = None
    password_hash: Optional[str] = None

    # Billing (Stripe)
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None

    # Organization membership
    org_id: Optional[str] = Field(default=None, index=True)
    org_role: Optional[str] = None  # owner|admin|member|viewer

    # Notifications
    notify_on_job_complete: bool = Field(default=False)


class Organization(SQLModel, table=True):
    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    name: str = Field(index=True)
    slug: str = Field(index=True, unique=True)
    owner_id: str = Field(index=True)
    plan: str = Field(default="team")
    max_members: int = Field(default=10)
    avatar_url: Optional[str] = None


class OrgInvite(SQLModel, table=True):
    id: str = Field(primary_key=True)
    org_id: str = Field(index=True)
    email: str = Field(index=True)
    role: str = Field(default="member")
    invited_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accepted: bool = Field(default=False)


class Project(SQLModel, table=True):
    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    name: str
    description: str = ""
    owner_id: str = Field(index=True)  # user or org
    owner_type: str = Field(default="user")  # user|org
    archived: bool = Field(default=False)


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    status: str = Field(default="queued", index=True)  # queued|running|succeeded|failed
    spec_json: dict = Field(sa_column=Column(JSON))
    job_dir: str
    error: Optional[str] = None

    # Phase 3 additions
    project_id: Optional[str] = Field(default=None, index=True)
    version_tag: Optional[str] = None  # e.g., "v1", "v2.1"
    parent_job_id: Optional[str] = Field(default=None, index=True)  # version lineage
    notes: Optional[str] = None

    # Phase 4: worker-side plan enforcement
    user_plan: Optional[str] = Field(default="free")  # snapshot of user.plan at enqueue time


class Candidate(SQLModel, table=True):
    id: str = Field(primary_key=True)
    job_id: str = Field(index=True)
    index: int = Field(index=True)
    label: str = Field(default="Candidate")
    metrics_json: dict = Field(sa_column=Column(JSON))
    artifacts_json: dict = Field(sa_column=Column(JSON))


class DesignSnapshot(SQLModel, table=True):
    """Auto-saved design state for version history timeline."""
    id: str = Field(primary_key=True)
    project_id: str = Field(index=True)
    user_id: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    spec_json: dict = Field(sa_column=Column(JSON))
    label: Optional[str] = None  # user-given name
    job_id: Optional[str] = None  # linked job if generated


class Webhook(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    url: str
    secret: str  # HMAC signing key
    events: list = Field(sa_column=Column(JSON))  # ["job.completed", "job.failed"]
    active: bool = Field(default=True)
    last_triggered_at: Optional[datetime] = None
    failure_count: int = Field(default=0)


class UsageEvent(SQLModel, table=True):
    """Fine-grained usage event for analytics."""
    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    org_id: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    event_type: str = Field(index=True)  # job.created|job.completed|export.step|api.call
    metadata_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))


class RateLimitBucket(SQLModel, table=True):
    """Per-user rate limit state (fallback when Redis unavailable)."""
    id: str = Field(primary_key=True)
    user_id: str = Field(index=True, unique=True)
    window_start: datetime = Field(default_factory=datetime.utcnow)
    request_count: int = Field(default=0)


class StripeWebhookEvent(SQLModel, table=True):
    """Processed Stripe webhook events (idempotency + audit)."""
    id: str = Field(primary_key=True)  # Stripe event id
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    event_type: str = Field(index=True)
    customer_id: Optional[str] = Field(default=None, index=True)
    user_id: Optional[str] = Field(default=None, index=True)
    payload_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))


class PasswordResetToken(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    token: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    expires_at: datetime = Field(index=True)
    used_at: Optional[datetime] = None


class BillingAuditLog(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    event_type: str = Field(index=True)
    details_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))


class CalibrationProfileDB(SQLModel, table=True):
    """Persisted calibration profile with correction factors."""
    __tablename__ = "calibration_profile"

    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    org_id: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    name: str = Field(default="Custom")
    description: str = Field(default="")
    process_preset: str = Field(default="CNC", index=True)
    coolant: str = Field(default="water")

    # Correction factors
    friction_factor_multiplier: float = Field(default=1.0)
    junction_K_multiplier: float = Field(default=1.0)
    heat_transfer_multiplier: float = Field(default=1.0)
    minor_loss_multiplier: float = Field(default=1.0)

    # Fit quality
    n_calibration_points: int = Field(default=0)
    r_squared_dp: float = Field(default=0.0)
    r_squared_thermal: float = Field(default=0.0)
    fit_rmse_dp_kpa: float = Field(default=0.0)
    fit_rmse_thermal_C: float = Field(default=0.0)

    # Raw calibration data (JSON array)
    points_json: Optional[str] = None

    active: bool = Field(default=True)
