from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlmodel import SQLModel, Session, create_engine

from .settings import settings

logger = logging.getLogger("branchforge.api.db")

_connect_args = {}
if settings.db_url.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(
    settings.db_url,
    connect_args=_connect_args,
    pool_pre_ping=True,
)


def _run_alembic_migrations() -> None:
    python_dir = Path(__file__).resolve().parents[2]  # .../python
    alembic_ini = python_dir / "alembic.ini"
    alembic_script_dir = python_dir / "alembic"

    alembic_cfg = Config(str(alembic_ini))
    alembic_cfg.set_main_option("script_location", str(alembic_script_dir))
    alembic_cfg.set_main_option("sqlalchemy.url", settings.db_url)

    command.upgrade(alembic_cfg, "head")

    script = ScriptDirectory.from_config(alembic_cfg)
    head_revision = script.get_current_head()
    with engine.connect() as conn:
        current_revision = MigrationContext.configure(conn).get_current_revision()

    logger.info(
        "Alembic migrations applied",
        extra={
            "event": "db.migrations.applied",
            "current_revision": current_revision,
            "head_revision": head_revision,
        },
    )


def init_db():
    """Initialize database schema.

    Uses Alembic migrations for non-SQLite environments.
    Falls back to SQLModel create_all for SQLite/test environments.
    """
    if settings.db_url.startswith("sqlite"):
        SQLModel.metadata.create_all(engine)
        logger.info(
            "SQLite/test DB detected; using SQLModel.create_all fallback",
            extra={"event": "db.create_all.fallback"},
        )
        return

    _run_alembic_migrations()


def get_session():
    with Session(engine) as session:
        yield session
