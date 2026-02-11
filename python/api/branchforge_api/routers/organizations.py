"""3A: Organizations and Teams API."""

from __future__ import annotations

import re
import uuid
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from ..db import get_session
from ..models import User, Organization, OrgInvite
from .auth import get_current_user

router = APIRouter()


# ── Schemas ──────────────────────────────────────────

class OrgCreate(BaseModel):
    name: str
    slug: Optional[str] = None

class OrgUpdate(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None

class OrgResponse(BaseModel):
    id: str
    name: str
    slug: str
    owner_id: str
    plan: str
    max_members: int
    avatar_url: Optional[str]
    member_count: int = 0

class MemberResponse(BaseModel):
    user_id: str
    email: str
    display_name: Optional[str]
    avatar_url: Optional[str]
    role: str

class InviteCreate(BaseModel):
    email: str
    role: str = "member"

class InviteResponse(BaseModel):
    id: str
    email: str
    role: str
    accepted: bool


# ── Helpers ──────────────────────────────────────────

def _slugify(name: str) -> str:
    s = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    return s[:48] or 'org'


def _require_org_admin(user: User, org: Organization):
    if user.org_id != org.id or user.org_role not in ("owner", "admin"):
        raise HTTPException(403, "You must be an org admin")


# ── Endpoints ────────────────────────────────────────

@router.post("/orgs", response_model=OrgResponse)
async def create_org(
    body: OrgCreate,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    if user.org_id:
        raise HTTPException(400, "You already belong to an organization")

    slug = body.slug or _slugify(body.name)
    existing = session.exec(select(Organization).where(Organization.slug == slug)).first()
    if existing:
        raise HTTPException(409, f"Slug '{slug}' is already taken")

    org = Organization(
        id=str(uuid.uuid4()),
        name=body.name,
        slug=slug,
        owner_id=user.id,
    )
    session.add(org)

    user.org_id = org.id
    user.org_role = "owner"
    session.add(user)
    session.commit()
    session.refresh(org)

    return OrgResponse(**org.model_dump(), member_count=1)


@router.get("/orgs/me", response_model=Optional[OrgResponse])
async def get_my_org(
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    if not user.org_id:
        return None
    org = session.get(Organization, user.org_id)
    if not org:
        return None

    members = session.exec(select(User).where(User.org_id == org.id)).all()
    return OrgResponse(**org.model_dump(), member_count=len(members))


@router.patch("/orgs/{org_id}", response_model=OrgResponse)
async def update_org(
    org_id: str,
    body: OrgUpdate,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    org = session.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found")
    _require_org_admin(user, org)

    if body.name is not None:
        org.name = body.name
    if body.avatar_url is not None:
        org.avatar_url = body.avatar_url

    session.add(org)
    session.commit()
    session.refresh(org)

    members = session.exec(select(User).where(User.org_id == org.id)).all()
    return OrgResponse(**org.model_dump(), member_count=len(members))


@router.get("/orgs/{org_id}/members", response_model=List[MemberResponse])
async def list_members(
    org_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    if user.org_id != org_id:
        raise HTTPException(403, "You are not a member of this organization")

    members = session.exec(select(User).where(User.org_id == org_id)).all()
    return [
        MemberResponse(
            user_id=m.id, email=m.email,
            display_name=m.display_name, avatar_url=m.avatar_url,
            role=m.org_role or "member",
        )
        for m in members
    ]


@router.post("/orgs/{org_id}/invites", response_model=InviteResponse)
async def create_invite(
    org_id: str,
    body: InviteCreate,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    org = session.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found")
    _require_org_admin(user, org)

    if body.role not in ("member", "admin", "viewer"):
        raise HTTPException(400, "Invalid role")

    invite = OrgInvite(
        id=str(uuid.uuid4()),
        org_id=org_id,
        email=body.email,
        role=body.role,
        invited_by=user.id,
    )
    session.add(invite)
    session.commit()
    session.refresh(invite)

    return InviteResponse(id=invite.id, email=invite.email, role=invite.role, accepted=False)


@router.post("/orgs/invites/{invite_id}/accept")
async def accept_invite(
    invite_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    invite = session.get(OrgInvite, invite_id)
    if not invite:
        raise HTTPException(404, "Invite not found")
    if invite.email != user.email:
        raise HTTPException(403, "This invite is for a different email")
    if invite.accepted:
        raise HTTPException(400, "Already accepted")
    if user.org_id:
        raise HTTPException(400, "You already belong to an organization")

    invite.accepted = True
    user.org_id = invite.org_id
    user.org_role = invite.role
    session.add(invite)
    session.add(user)
    session.commit()

    return {"status": "joined", "org_id": invite.org_id}


@router.delete("/orgs/{org_id}/members/{member_id}")
async def remove_member(
    org_id: str,
    member_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    org = session.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found")
    _require_org_admin(user, org)

    member = session.get(User, member_id)
    if not member or member.org_id != org_id:
        raise HTTPException(404, "Member not found in this organization")
    if member.id == org.owner_id:
        raise HTTPException(400, "Cannot remove the organization owner")

    member.org_id = None
    member.org_role = None
    session.add(member)
    session.commit()

    return {"status": "removed"}
