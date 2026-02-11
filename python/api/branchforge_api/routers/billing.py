"""Stripe billing: checkout sessions, customer portal, webhooks."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import Session, select

from ..db import get_session
from ..models import BillingAuditLog, StripeWebhookEvent, User
from ..settings import settings
from ..billing_utils import PLAN_LIMITS
from ..email import EmailTemplate, send_email, send_template_email
from .auth import get_current_user

router = APIRouter()
logger = logging.getLogger("branchforge.api.billing")

stripe.api_key = settings.stripe_secret_key


class CheckoutRequest(BaseModel):
    plan: str


class CheckoutResponse(BaseModel):
    url: str


class PortalResponse(BaseModel):
    url: str


class PlanInfo(BaseModel):
    name: str
    jobs_per_month: int
    candidates_per_job: int
    step_export: bool
    priority_queue: bool
    pdf_reports: bool


class UsageResponse(BaseModel):
    plan: str
    jobs_used: int
    jobs_limit: int
    candidates_per_job: int
    next_billing_date: str | None = None
    plans: list[PlanInfo]


PLAN_PRICE_MAP = {
    "pro": settings.stripe_price_pro,
    "team": settings.stripe_price_team,
    "enterprise": settings.stripe_price_enterprise,
}


def _create_audit_log(session: Session, event_type: str, user_id: str | None, details: dict):
    row = BillingAuditLog(
        id=str(uuid.uuid4()),
        user_id=user_id,
        event_type=event_type,
        details_json=details,
    )
    session.add(row)
    session.commit()


def _subscription_period_end_iso(user: User) -> str | None:
    if not settings.stripe_secret_key or not user.stripe_subscription_id:
        return None
    try:
        sub = stripe.Subscription.retrieve(user.stripe_subscription_id)
        end_ts = sub.get("current_period_end")
        if end_ts:
            return datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()
    except Exception:
        logger.exception("Could not load subscription period end", extra={"user_id": user.id})
    return None


@router.get("/billing/usage", response_model=UsageResponse)
def get_usage(user: User = Depends(get_current_user)):
    limits = PLAN_LIMITS.get(user.plan, PLAN_LIMITS["free"])
    plans = [PlanInfo(name=k, **v) for k, v in PLAN_LIMITS.items()]
    return UsageResponse(
        plan=user.plan,
        jobs_used=user.jobs_used,
        jobs_limit=limits["jobs_per_month"],
        candidates_per_job=limits["candidates_per_job"],
        next_billing_date=_subscription_period_end_iso(user),
        plans=plans,
    )


@router.post("/billing/checkout", response_model=CheckoutResponse)
def create_checkout(
    req: CheckoutRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=501, detail="Billing not configured")

    price_id = PLAN_PRICE_MAP.get(req.plan)
    if not price_id:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {req.plan}")

    if not user.stripe_customer_id:
        customer = stripe.Customer.create(
            email=user.email,
            metadata={"branchforge_user_id": user.id},
        )
        user.stripe_customer_id = customer.id
        session.add(user)
        session.commit()

    checkout = stripe.checkout.Session.create(
        customer=user.stripe_customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{settings.frontend_url}/settings?checkout=success",
        cancel_url=f"{settings.frontend_url}/pricing?checkout=cancelled",
        metadata={"branchforge_user_id": user.id, "plan": req.plan},
    )

    return CheckoutResponse(url=checkout.url)


@router.post("/billing/portal", response_model=PortalResponse)
def create_portal(user: User = Depends(get_current_user)):
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=501, detail="Billing not configured")
    if not user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No billing account found")

    portal = stripe.billing_portal.Session.create(
        customer=user.stripe_customer_id,
        return_url=f"{settings.frontend_url}/settings",
    )
    return PortalResponse(url=portal.url)


@router.post("/billing/webhook")
async def stripe_webhook(request: Request, session: Session = Depends(get_session)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not settings.stripe_webhook_secret:
        raise HTTPException(status_code=503, detail="Stripe webhook not configured")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event_id = event.get("id")
    event_type = event.get("type", "")

    if event_id:
        already = session.get(StripeWebhookEvent, event_id)
        if already:
            return {"status": "ok", "idempotent": True}

    if event_type == "checkout.session.completed":
        _handle_checkout_completed(event["data"]["object"], session, event)
    elif event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
        _handle_subscription_change(event["data"]["object"], session, event)
    elif event_type == "invoice.payment_failed":
        _handle_invoice_payment_failed(event["data"]["object"], session)
    elif event_type == "invoice.paid":
        _handle_invoice_paid(event["data"]["object"], session)
    elif event_type == "customer.subscription.trial_will_end":
        _handle_trial_will_end(event["data"]["object"], session)

    if event_id:
        row = StripeWebhookEvent(
            id=event_id,
            event_type=event_type,
            customer_id=(event.get("data", {}).get("object", {}) or {}).get("customer"),
            payload_json={"created": event.get("created")},
        )
        session.add(row)
        session.commit()

    return {"status": "ok"}


def _find_user_by_customer(customer_id: str | None, session: Session) -> User | None:
    if not customer_id:
        return None
    return session.exec(select(User).where(User.stripe_customer_id == customer_id)).first()


def _handle_checkout_completed(checkout_session: dict, session: Session, event: dict):
    customer_id = checkout_session.get("customer")
    subscription_id = checkout_session.get("subscription")
    plan = checkout_session.get("metadata", {}).get("plan", "pro")
    user_id = checkout_session.get("metadata", {}).get("branchforge_user_id")

    if not user_id:
        return

    user = session.get(User, user_id)
    if not user:
        return

    user.plan = plan
    user.stripe_customer_id = customer_id
    user.stripe_subscription_id = subscription_id
    session.add(user)
    session.commit()

    _create_audit_log(session, "checkout.session.completed", user.id, {"plan": plan, "subscription_id": subscription_id, "event_id": event.get("id")})


def _handle_subscription_change(subscription: dict, session: Session, event: dict):
    customer_id = subscription.get("customer")
    status = subscription.get("status")

    user = _find_user_by_customer(customer_id, session)
    if not user:
        return

    if status in ("canceled", "unpaid"):
        user.plan = "free"
        user.stripe_subscription_id = None
    elif status in ("active", "trialing", "past_due"):
        items = subscription.get("items", {}).get("data", [])
        if items:
            price_id = items[0].get("price", {}).get("id", "")
            for plan_name, pid in PLAN_PRICE_MAP.items():
                if pid and pid == price_id:
                    user.plan = plan_name
                    break

    session.add(user)
    session.commit()
    _create_audit_log(session, "customer.subscription.changed", user.id, {"status": status, "event_id": event.get("id")})


def _handle_invoice_payment_failed(invoice: dict, session: Session):
    customer_id = invoice.get("customer")
    user = _find_user_by_customer(customer_id, session)
    if not user:
        return

    grace_ends = (datetime.utcnow() + timedelta(days=7)).isoformat()
    _create_audit_log(
        session,
        "invoice.payment_failed",
        user.id,
        {
            "invoice_id": invoice.get("id"),
            "amount_due": invoice.get("amount_due"),
            "currency": invoice.get("currency"),
            "grace_period_ends_at": grace_ends,
            "action": "downgrade_if_unpaid_after_grace",
        },
    )

    send_email(
        to_email=user.email,
        subject="Payment issue - action needed",
        html=f"""
        <h2>Payment failed</h2>
        <p>We couldn't process your latest invoice ({invoice.get('id')}).</p>
        <p>Your grace period ends on <strong>{grace_ends}</strong>. Please update your payment method to avoid downgrade.</p>
        <p><a href=\"{settings.frontend_url}/settings\">Manage billing</a></p>
        """,
    )


def _handle_invoice_paid(invoice: dict, session: Session):
    customer_id = invoice.get("customer")
    user = _find_user_by_customer(customer_id, session)
    if not user:
        return

    _create_audit_log(
        session,
        "invoice.paid",
        user.id,
        {
            "invoice_id": invoice.get("id"),
            "amount_paid": invoice.get("amount_paid"),
            "currency": invoice.get("currency"),
            "subscription": invoice.get("subscription"),
        },
    )

    send_template_email(
        to_email=user.email,
        template=EmailTemplate.PAYMENT_RECEIPT,
        context={
            "amount": (invoice.get("amount_paid") or 0) / 100,
            "currency": invoice.get("currency") or "usd",
            "invoice_id": invoice.get("id"),
            "paid_at": datetime.utcnow().isoformat(),
        },
    )


def _handle_trial_will_end(subscription: dict, session: Session):
    customer_id = subscription.get("customer")
    user = _find_user_by_customer(customer_id, session)
    if not user:
        return

    trial_end = subscription.get("trial_end")
    trial_end_iso = None
    if trial_end:
        trial_end_iso = datetime.fromtimestamp(trial_end, tz=timezone.utc).isoformat()

    _create_audit_log(
        session,
        "customer.subscription.trial_will_end",
        user.id,
        {"subscription_id": subscription.get("id"), "trial_end": trial_end_iso},
    )

    send_email(
        to_email=user.email,
        subject="Your trial is ending soon",
        html=f"""
        <h2>Trial ending soon</h2>
        <p>Your subscription trial will end on <strong>{trial_end_iso or 'soon'}</strong>.</p>
        <p>Add or verify billing details to keep access uninterrupted.</p>
        <p><a href=\"{settings.frontend_url}/settings\">Open billing settings</a></p>
        """,
    )
