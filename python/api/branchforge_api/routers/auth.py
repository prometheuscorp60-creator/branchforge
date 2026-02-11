from __future__ import annotations

import secrets
import time
import uuid
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Header, Query, Request
from pydantic import BaseModel
from sqlmodel import Session, select

from ..db import get_session
from ..models import User, PasswordResetToken
from ..settings import settings
from ..auth_utils import (
    create_jwt,
    decode_jwt,
    hash_password,
    verify_password,
    new_api_key,
)
from ..email import send_template_email, send_email, EmailTemplate

router = APIRouter()
logger = logging.getLogger("branchforge.api.auth")

# Simple in-memory auth rate limiter: 5 attempts/minute per IP per endpoint.
_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_MAX_ATTEMPTS = 5
_rate_limit_buckets: dict[tuple[str, str], list[float]] = {}


def _enforce_rate_limit(request: Request, endpoint: str) -> None:
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    key = (endpoint, client_ip)
    attempts = _rate_limit_buckets.get(key, [])
    attempts = [t for t in attempts if now - t < _RATE_LIMIT_WINDOW_SECONDS]

    if len(attempts) >= _RATE_LIMIT_MAX_ATTEMPTS:
        retry_after = int(_RATE_LIMIT_WINDOW_SECONDS - (now - attempts[0])) if attempts else _RATE_LIMIT_WINDOW_SECONDS
        raise HTTPException(
            status_code=429,
            detail="Too many authentication attempts. Please try again shortly.",
            headers={"Retry-After": str(max(1, retry_after))},
        )

    attempts.append(now)
    _rate_limit_buckets[key] = attempts


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class MessageResponse(BaseModel):
    message: str


class AuthResponse(BaseModel):
    token: str
    api_key: str
    email: str
    plan: str
    display_name: str | None = None


class CreateKeyRequest(BaseModel):
    email: str
    plan: str = "free"


class CreateKeyResponse(BaseModel):
    api_key: str
    plan: str


class MeResponse(BaseModel):
    id: str
    email: str
    plan: str
    jobs_used: int
    display_name: str | None = None
    avatar_url: str | None = None
    oauth_provider: str | None = None
    stripe_customer_id: str | None = None
    stripe_subscription_id: str | None = None


def get_current_user(
    session: Session = Depends(get_session),
    authorization: str | None = Header(default=None),
    x_branchforge_key: str | None = Header(default=None, alias="X-BranchForge-Key"),
    key: str | None = Query(default=None),
):
    if settings.dev_no_auth:
        logger.warning("DEV_NO_AUTH bypass is active; request auto-authenticated as dev user")
        email = "dev@local"
        user = session.exec(select(User).where(User.email == email)).first()
        if user is None:
            user = User(
                id="dev",
                email=email,
                api_key="dev",
                plan="enterprise",
                display_name="Developer",
                month_start=datetime.utcnow(),
                jobs_used=0,
            )
            session.add(user)
            session.commit()
        return user

    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        payload = decode_jwt(token)
        if payload:
            user = session.exec(select(User).where(User.id == payload["sub"])).first()
            if user:
                return user

    api_key = x_branchforge_key or key
    if api_key:
        user = session.exec(select(User).where(User.api_key == api_key)).first()
        if user:
            return user

    raise HTTPException(status_code=401, detail="Authentication required")


@router.post("/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest, request: Request, session: Session = Depends(get_session)):
    _enforce_rate_limit(request, "register")
    existing = session.exec(select(User).where(User.email == req.email)).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user_id = str(uuid.uuid4())
    user = User(
        id=user_id,
        email=req.email,
        api_key=new_api_key(),
        plan="free",
        display_name=req.display_name or req.email.split("@")[0],
        password_hash=hash_password(req.password),
        month_start=datetime.utcnow(),
        jobs_used=0,
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    send_template_email(
        to_email=user.email,
        template=EmailTemplate.WELCOME,
        context={"name": user.display_name, "app_url": settings.frontend_url},
    )

    token = create_jwt(user.id, user.email)
    return AuthResponse(
        token=token,
        api_key=user.api_key,
        email=user.email,
        plan=user.plan,
        display_name=user.display_name,
    )


@router.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest, request: Request, session: Session = Depends(get_session)):
    _enforce_rate_limit(request, "login")
    user = session.exec(select(User).where(User.email == req.email)).first()
    if not user or not user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_jwt(user.id, user.email)
    return AuthResponse(
        token=token,
        api_key=user.api_key,
        email=user.email,
        plan=user.plan,
        display_name=user.display_name,
    )


@router.post("/auth/forgot-password", response_model=MessageResponse)
def forgot_password(req: ForgotPasswordRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == req.email)).first()
    if not user:
        return MessageResponse(message="If that email exists, a reset link has been sent")

    raw_token = secrets.token_urlsafe(32)
    reset = PasswordResetToken(
        id=str(uuid.uuid4()),
        user_id=user.id,
        token=raw_token,
        expires_at=datetime.utcnow() + timedelta(hours=1),
    )
    session.add(reset)
    session.commit()

    reset_url = f"{settings.frontend_url}/forgot-password?token={raw_token}"
    send_email(
        to_email=user.email,
        subject="Reset your BranchForge password",
        html=(
            "<h2>Password reset requested</h2>"
            "<p>Use the link below to reset your password. This link expires in 1 hour.</p>"
            f"<p><a href=\"{reset_url}\">Reset Password</a></p>"
            "<p>If you didn't request this, you can ignore this email.</p>"
        ),
    )

    return MessageResponse(message="If that email exists, a reset link has been sent")


@router.post("/auth/reset-password", response_model=MessageResponse)
def reset_password(req: ResetPasswordRequest, session: Session = Depends(get_session)):
    token_row = session.exec(
        select(PasswordResetToken).where(PasswordResetToken.token == req.token)
    ).first()

    if not token_row:
        raise HTTPException(status_code=400, detail="Invalid reset token")
    if token_row.used_at is not None:
        raise HTTPException(status_code=400, detail="Reset token already used")
    if token_row.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Reset token expired")

    user = session.get(User, token_row.user_id)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid reset token")

    user.password_hash = hash_password(req.new_password)
    token_row.used_at = datetime.utcnow()
    session.add(user)
    session.add(token_row)
    session.commit()

    return MessageResponse(message="Password reset successful")


@router.post("/auth/refresh", response_model=AuthResponse)
def refresh_token(user: User = Depends(get_current_user)):
    token = create_jwt(user.id, user.email)
    return AuthResponse(
        token=token,
        api_key=user.api_key,
        email=user.email,
        plan=user.plan,
        display_name=user.display_name,
    )


@router.post("/auth/create_key", response_model=CreateKeyResponse)
def create_key(req: CreateKeyRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == req.email)).first()
    if user is None:
        user = User(
            id=str(uuid.uuid4()),
            email=req.email,
            api_key=new_api_key(),
            plan=req.plan,
            month_start=datetime.utcnow(),
            jobs_used=0,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
    else:
        user.api_key = new_api_key()
        user.plan = req.plan or user.plan
        session.add(user)
        session.commit()
    return CreateKeyResponse(api_key=user.api_key, plan=user.plan)


@router.get("/auth/me", response_model=MeResponse)
def me(user: User = Depends(get_current_user)):
    return MeResponse(
        id=user.id,
        email=user.email,
        plan=user.plan,
        jobs_used=user.jobs_used,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        oauth_provider=user.oauth_provider,
        stripe_customer_id=user.stripe_customer_id,
        stripe_subscription_id=user.stripe_subscription_id,
    )
