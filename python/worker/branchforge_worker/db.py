from __future__ import annotations

from sqlmodel import SQLModel, create_engine, Session
from .settings import settings

engine = create_engine(
    settings.db_url,
    connect_args={"check_same_thread": False} if settings.db_url.startswith("sqlite") else {},
)

def get_session():
    return Session(engine)
