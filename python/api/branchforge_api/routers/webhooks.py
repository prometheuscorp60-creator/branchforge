"""3D: Webhooks API — HMAC-signed POST notifications for job events."""

from __future__ import annotations

import uuid
import hmac
import hashlib
import json
import secrets
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from ..db import get_session
from ..models import User, Webhook
from .auth import get_current_user

logger = logging.getLogger("branchforge.webhooks")

router = APIRouter()


VALID_EVENTS = [
    "job.created", "job.completed", "job.failed",
    "project.created", "snapshot.created",
]


# ── Schemas ──────────────────────────────────────────

class WebhookCreate(BaseModel):
    url: str
    events: List[str]

class WebhookUpdate(BaseModel):
    url: Optional[str] = None
    events: Optional[List[str]] = None
    active: Optional[bool] = None

class WebhookResponse(BaseModel):
    id: str
    url: str
    events: List[str]
    active: bool
    created_at: str
    last_triggered_at: Optional[str]
    failure_count: int


# ── HMAC signing ─────────────────────────────────────

def sign_payload(payload: dict, secret: str) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()


async def dispatch_webhook(webhook: Webhook, event: str, payload: dict):
    """Fire a webhook — non-blocking. In production, this goes through a queue."""
    import httpx

    sig = sign_payload(payload, webhook.secret)
    headers = {
        "Content-Type": "application/json",
        "X-BranchForge-Event": event,
        "X-BranchForge-Signature": f"sha256={sig}",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(webhook.url, json=payload, headers=headers)
            resp.raise_for_status()
            logger.info("Webhook %s dispatched: %s -> %d", webhook.id, event, resp.status_code)
    except Exception as e:
        logger.warning("Webhook %s failed: %s", webhook.id, e)


async def fire_webhooks_for_event(
    session: Session,
    user_id: str,
    event: str,
    payload: dict,
):
    """Find and fire all active webhooks for a given event."""
    webhooks = session.exec(
        select(Webhook).where(Webhook.user_id == user_id, Webhook.active == True)
    ).all()

    for wh in webhooks:
        if event in wh.events:
            try:
                await dispatch_webhook(wh, event, payload)
                wh.last_triggered_at = datetime.utcnow()
                wh.failure_count = 0
            except Exception:
                wh.failure_count += 1
                if wh.failure_count >= 10:
                    wh.active = False
            session.add(wh)

    session.commit()


# ── Endpoints ────────────────────────────────────────

@router.post("/webhooks", response_model=WebhookResponse)
async def create_webhook(
    body: WebhookCreate,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    for ev in body.events:
        if ev not in VALID_EVENTS:
            raise HTTPException(400, f"Invalid event: {ev}. Valid: {VALID_EVENTS}")

    existing = session.exec(
        select(Webhook).where(Webhook.user_id == user.id)
    ).all()
    if len(existing) >= 10:
        raise HTTPException(400, "Maximum 10 webhooks per user")

    wh = Webhook(
        id=str(uuid.uuid4()),
        user_id=user.id,
        url=body.url,
        secret=secrets.token_hex(32),
        events=body.events,
    )
    session.add(wh)
    session.commit()
    session.refresh(wh)

    return WebhookResponse(
        id=wh.id, url=wh.url, events=wh.events,
        active=wh.active, created_at=wh.created_at.isoformat(),
        last_triggered_at=None, failure_count=0,
    )


@router.get("/webhooks", response_model=List[WebhookResponse])
async def list_webhooks(
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    webhooks = session.exec(
        select(Webhook).where(Webhook.user_id == user.id)
    ).all()
    return [
        WebhookResponse(
            id=wh.id, url=wh.url, events=wh.events,
            active=wh.active, created_at=wh.created_at.isoformat(),
            last_triggered_at=wh.last_triggered_at.isoformat() if wh.last_triggered_at else None,
            failure_count=wh.failure_count,
        )
        for wh in webhooks
    ]


@router.patch("/webhooks/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(
    webhook_id: str,
    body: WebhookUpdate,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    wh = session.get(Webhook, webhook_id)
    if not wh or wh.user_id != user.id:
        raise HTTPException(404, "Webhook not found")

    if body.url is not None:
        wh.url = body.url
    if body.events is not None:
        for ev in body.events:
            if ev not in VALID_EVENTS:
                raise HTTPException(400, f"Invalid event: {ev}")
        wh.events = body.events
    if body.active is not None:
        wh.active = body.active
        if body.active:
            wh.failure_count = 0

    session.add(wh)
    session.commit()
    session.refresh(wh)

    return WebhookResponse(
        id=wh.id, url=wh.url, events=wh.events,
        active=wh.active, created_at=wh.created_at.isoformat(),
        last_triggered_at=wh.last_triggered_at.isoformat() if wh.last_triggered_at else None,
        failure_count=wh.failure_count,
    )


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    wh = session.get(Webhook, webhook_id)
    if not wh or wh.user_id != user.id:
        raise HTTPException(404, "Webhook not found")

    session.delete(wh)
    session.commit()
    return {"status": "deleted"}


@router.post("/webhooks/{webhook_id}/test")
async def test_webhook(
    webhook_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    wh = session.get(Webhook, webhook_id)
    if not wh or wh.user_id != user.id:
        raise HTTPException(404, "Webhook not found")

    test_payload = {
        "event": "test.ping",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {"message": "BranchForge webhook test"},
    }
    await dispatch_webhook(wh, "test.ping", test_payload)
    return {"status": "sent", "payload": test_payload}
