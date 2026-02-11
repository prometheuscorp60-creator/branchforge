"""JWT creation / verification and password hashing utilities."""
from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
import bcrypt

JWT_SECRET = os.getenv("JWT_SECRET", "branchforge-dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "72"))


def create_jwt(user_id: str, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": now + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": now,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> dict | None:
    try:
        return jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "email", "exp", "iat"]},
        )
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, jwt.MissingRequiredClaimError):
        return None


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def new_api_key() -> str:
    return "bf_" + secrets.token_urlsafe(24)
