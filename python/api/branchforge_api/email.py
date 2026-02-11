from __future__ import annotations

import logging
from datetime import datetime

import httpx

from .settings import settings

logger = logging.getLogger("branchforge.api.email")


class EmailTemplate:
    WELCOME = "welcome"
    PASSWORD_RESET = "password_reset"
    JOB_COMPLETED = "job_completed"
    PAYMENT_RECEIPT = "payment_receipt"


def _render_template(template: str, context: dict) -> tuple[str, str]:
    app_url = context.get("app_url") or settings.frontend_url

    if template == EmailTemplate.WELCOME:
        subject = "Welcome to BranchForge"
        html = f"""
        <h2>Welcome to BranchForge</h2>
        <p>Hi {context.get('name', 'there')},</p>
        <p>Your account is ready. Start your first design in minutes.</p>
        <p><a href=\"{app_url}/app\">Open Designer</a></p>
        """
        return subject, html

    if template == EmailTemplate.PASSWORD_RESET:
        subject = "Reset your BranchForge password"
        html = f"""
        <h2>Password reset requested</h2>
        <p>Use the link below to reset your password. This link expires in 1 hour.</p>
        <p><a href=\"{context.get('reset_url', app_url)}\">Reset Password</a></p>
        <p>If you didn't request this, you can ignore this email.</p>
        """
        return subject, html

    if template == EmailTemplate.JOB_COMPLETED:
        subject = "Your BranchForge job is complete"
        html = f"""
        <h2>Job complete</h2>
        <p>Job <strong>{context.get('job_id')}</strong> finished successfully.</p>
        <p>Candidates generated: {context.get('candidate_count', 'N/A')}</p>
        <p><a href=\"{context.get('job_url', app_url + '/dashboard')}\">View results</a></p>
        """
        return subject, html

    if template == EmailTemplate.PAYMENT_RECEIPT:
        subject = "Payment received"
        amount = context.get("amount", "")
        currency = context.get("currency", "").upper()
        paid_at = context.get("paid_at") or datetime.utcnow().isoformat()
        html = f"""
        <h2>Payment receipt</h2>
        <p>Thanks for your payment.</p>
        <ul>
          <li>Amount: {amount} {currency}</li>
          <li>Date: {paid_at}</li>
          <li>Invoice: {context.get('invoice_id', 'N/A')}</li>
        </ul>
        <p>Need help? Reply to this message.</p>
        """
        return subject, html

    raise ValueError(f"Unsupported email template: {template}")


def send_email(to_email: str, subject: str, html: str) -> bool:
    if not settings.resend_api_key:
        logger.warning("RESEND_API_KEY is missing; email not sent", extra={"to": to_email, "subject": subject})
        return False

    try:
        resp = httpx.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {settings.resend_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": settings.email_from,
                "to": [to_email],
                "subject": subject,
                "html": html,
            },
            timeout=20.0,
        )
        if resp.is_success:
            return True

        logger.error(
            "Resend email failed",
            extra={"status_code": resp.status_code, "body": resp.text[:500], "to": to_email},
        )
        return False
    except Exception:
        logger.exception("Resend email exception")
        return False


def send_template_email(to_email: str, template: str, context: dict) -> bool:
    subject, html = _render_template(template, context)
    return send_email(to_email, subject, html)
