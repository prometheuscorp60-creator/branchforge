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


def _resolve_db_url() -> str:
    db_url = settings.db_url
    if not db_url.startswith("sqlite"):
        return db_url

    # Ensure SQLite parent dir exists; fall back to /tmp if configured path is not writable.
    sqlite_path = db_url.replace("sqlite:///", "", 1)
    if sqlite_path and sqlite_path != ":memory:" and not sqlite_path.startswith("file:"):
        db_file = Path(sqlite_path)
        try:
            db_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            fallback = Path("/tmp/branchforge.db")
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback_url = f"sqlite:///{fallback.as_posix()}"
            logger.warning(
                "SQLite path not writable; falling back to /tmp/branchforge.db",
                extra={
                    "event": "db.sqlite.fallback_tmp",
                    "configured_db_url": db_url,
                    "fallback_db_url": fallback_url,
                },
            )
            return fallback_url

    return db_url


EFFECTIVE_DB_URL = _resolve_db_url()

_connect_args = {}
if EFFECTIVE_DB_URL.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(
    EFFECTIVE_DB_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,
)


def _run_alembic_migrations() -> None:
    python_dir = Path(__file__).resolve().parents[2]  # .../python
    alembic_ini = python_dir / "alembic.ini"
    alembic_script_dir = python_dir / "alembic"

    alembic_cfg = Config(str(alembic_ini))
    alembic_cfg.set_main_option("script_location", str(alembic_script_dir))
    alembic_cfg.set_main_option("sqlalchemy.url", EFFECTIVE_DB_URL)

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
    if EFFECTIVE_DB_URL.startswith("sqlite"):
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
