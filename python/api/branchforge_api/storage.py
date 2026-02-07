from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Tuple

from .settings import settings


def make_job_dir(job_id: str) -> str:
    base = Path(settings.artifacts_dir)
    base.mkdir(parents=True, exist_ok=True)
    job_dir = base / job_id
    (job_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (job_dir / "candidates").mkdir(parents=True, exist_ok=True)
    return str(job_dir)


def save_upload(file_bytes: bytes, job_dir: str, filename: str) -> str:
    path = Path(job_dir) / "inputs" / filename
    path.write_bytes(file_bytes)
    return str(path)
