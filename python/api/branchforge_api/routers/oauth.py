"""OAuth2 callback handlers for Google and GitHub sign-in."""
from __future__ import annotations

import uuid
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlmodel import Session, select

from ..db import get_session
from ..models import User
from ..settings import settings
from ..auth_utils import create_jwt, new_api_key

router = APIRouter()


# ── Google OAuth ─────────────────────────────────────────────────

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


@router.get("/oauth/google")
def google_login():
    """Redirect user to Google consent screen."""
    if not settings.google_client_id:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")

    redirect_uri = f"{settings.frontend_url}/oauth/google/callback"
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    url = f"{GOOGLE_AUTH_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    return RedirectResponse(url)


@router.get("/oauth/google/callback")
def google_callback(
    code: str = Query(...),
    session: Session = Depends(get_session),
):
    """Exchange Google auth code for token, upsert user, return JWT redirect."""
    if not settings.google_client_id:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")

    redirect_uri = f"{settings.frontend_url}/oauth/google/callback"

    # Exchange code for tokens
    token_resp = httpx.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
    )
    if token_resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to exchange Google auth code")

    access_token = token_resp.json().get("access_token")

    # Fetch user info
    userinfo_resp = httpx.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if userinfo_resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch Google user info")

    info = userinfo_resp.json()
    email = info.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Google account has no email")

    user = _upsert_oauth_user(
        session,
        email=email,
        provider="google",
        oauth_id=info.get("id", ""),
        display_name=info.get("name", email.split("@")[0]),
        avatar_url=info.get("picture"),
    )

    token = create_jwt(user.id, user.email)
    return RedirectResponse(f"{settings.frontend_url}/oauth/complete?token={token}")


# ── GitHub OAuth ─────────────────────────────────────────────────

GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


@router.get("/oauth/github")
def github_login():
    """Redirect user to GitHub consent screen."""
    if not settings.github_client_id:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")

    redirect_uri = f"{settings.frontend_url}/oauth/github/callback"
    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": redirect_uri,
        "scope": "user:email",
    }
    url = f"{GITHUB_AUTH_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    return RedirectResponse(url)


@router.get("/oauth/github/callback")
def github_callback(
    code: str = Query(...),
    session: Session = Depends(get_session),
):
    """Exchange GitHub auth code for token, upsert user, return JWT redirect."""
    if not settings.github_client_id:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")

    # Exchange code for token
    token_resp = httpx.post(
        GITHUB_TOKEN_URL,
        headers={"Accept": "application/json"},
        data={
            "client_id": settings.github_client_id,
            "client_secret": settings.github_client_secret,
            "code": code,
        },
    )
    if token_resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to exchange GitHub auth code")

    access_token = token_resp.json().get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="No access token from GitHub")

    gh_headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    # Fetch user profile
    user_resp = httpx.get(GITHUB_USER_URL, headers=gh_headers)
    if user_resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch GitHub user info")
    gh_user = user_resp.json()

    # Fetch primary email
    email = gh_user.get("email")
    if not email:
        emails_resp = httpx.get(GITHUB_EMAILS_URL, headers=gh_headers)
        if emails_resp.status_code == 200:
            for e in emails_resp.json():
                if e.get("primary") and e.get("verified"):
                    email = e["email"]
                    break
    if not email:
        raise HTTPException(status_code=400, detail="GitHub account has no verified email")

    user = _upsert_oauth_user(
        session,
        email=email,
        provider="github",
        oauth_id=str(gh_user.get("id", "")),
        display_name=gh_user.get("name") or gh_user.get("login", email.split("@")[0]),
        avatar_url=gh_user.get("avatar_url"),
    )

    token = create_jwt(user.id, user.email)
    return RedirectResponse(f"{settings.frontend_url}/oauth/complete?token={token}")


# ── Helpers ──────────────────────────────────────────────────────

def _upsert_oauth_user(
    session: Session,
    email: str,
    provider: str,
    oauth_id: str,
    display_name: str,
    avatar_url: str | None,
) -> User:
    """Find or create a user from OAuth info."""
    user = session.exec(select(User).where(User.email == email)).first()
    if user is None:
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            api_key=new_api_key(),
            plan="free",
            display_name=display_name,
            avatar_url=avatar_url,
            oauth_provider=provider,
            oauth_id=oauth_id,
            month_start=datetime.utcnow(),
            jobs_used=0,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
    else:
        # Update profile from OAuth provider
        user.display_name = user.display_name or display_name
        user.avatar_url = avatar_url or user.avatar_url
        user.oauth_provider = user.oauth_provider or provider
        user.oauth_id = user.oauth_id or oauth_id
        session.add(user)
        session.commit()
    return user
