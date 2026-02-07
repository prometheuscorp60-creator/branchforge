from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from ..db import get_session
from ..models import User
from ..settings import settings

router = APIRouter()

class CreateKeyRequest(BaseModel):
    email: str
    plan: str = "free"  # free|pro|team|enterprise

class CreateKeyResponse(BaseModel):
    api_key: str
    plan: str

class MeResponse(BaseModel):
    email: str
    plan: str
    jobs_used: int

def _new_key() -> str:
    return "bf_" + secrets.token_urlsafe(24)

def get_current_user(
    session: Session = Depends(get_session),
    x_branchforge_key: str | None = Header(default=None, alias="X-BranchForge-Key"),
    key: str | None = Query(default=None),
):
    if settings.dev_no_auth:
        # dev singleton user
        email = "dev@local"
        user = session.exec(select(User).where(User.email == email)).first()
        if user is None:
            user = User(
                id="dev",
                email=email,
                api_key="dev",
                plan="enterprise",
                month_start=datetime.utcnow(),
                jobs_used=0,
            )
            session.add(user)
            session.commit()
        return user

    token = x_branchforge_key or key
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")

    user = session.exec(select(User).where(User.api_key == token)).first()
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user

@router.post("/auth/create_key", response_model=CreateKeyResponse)
def create_key(req: CreateKeyRequest, session: Session = Depends(get_session)):
    if not settings.dev_no_auth:
        # keep it simple: in prod you restrict this; for MVP, allow.
        pass
    user = session.exec(select(User).where(User.email == req.email)).first()
    if user is None:
        user = User(
            id=_new_key(),  # ok for MVP
            email=req.email,
            api_key=_new_key(),
            plan=req.plan,
            month_start=datetime.utcnow(),
            jobs_used=0,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
    else:
        # rotate key
        user.api_key = _new_key()
        user.plan = req.plan or user.plan
        session.add(user)
        session.commit()
    return CreateKeyResponse(api_key=user.api_key, plan=user.plan)

@router.get("/auth/me", response_model=MeResponse)
def me(user: User = Depends(get_current_user)):
    return MeResponse(email=user.email, plan=user.plan, jobs_used=user.jobs_used)
