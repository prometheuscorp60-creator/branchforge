"""Storage helpers for job artifacts.

Wraps the storage backend abstraction so existing callers (jobs router,
worker tasks) can keep using ``make_job_dir`` and ``save_upload`` without
caring whether artifacts land on local disk or S3.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from branchforge_core.storage_backend import get_storage_backend, LocalBackend


def _backend():
    return get_storage_backend()


def make_job_dir(job_id: str) -> str:
    """Create (or identify) the directory/prefix for a job's artifacts.

    For local backend: creates the actual directory tree and returns the path.
    For S3 backend: returns the key prefix (no directory creation needed).
    """
    backend = _backend()

    if isinstance(backend, LocalBackend):
        job_dir = Path(backend.base_dir) / job_id
        (job_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (job_dir / "candidates").mkdir(parents=True, exist_ok=True)
        return str(job_dir)
    else:
        # S3: job_dir is the key prefix; actual files are written to a
        # local temp dir first, then uploaded by the worker after generation.
        return job_id


def make_local_work_dir(job_id: str) -> str:
    """Create a local temp directory for the worker to generate into.

    For local backend: same as make_job_dir (artifacts dir IS the work dir).
    For S3 backend: creates a temp dir that will be uploaded post-generation.
    """
    backend = _backend()

    if isinstance(backend, LocalBackend):
        return make_job_dir(job_id)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix=f"bf-{job_id[:8]}-"))
        (work_dir / "inputs").mkdir(exist_ok=True)
        (work_dir / "candidates").mkdir(exist_ok=True)
        return str(work_dir)


def save_upload(file_bytes: bytes, job_dir: str, filename: str) -> str:
    """Persist an uploaded file (heatmap, outline DXF, etc.)."""
    backend = _backend()

    if isinstance(backend, LocalBackend):
        path = Path(job_dir) / "inputs" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file_bytes)
        return str(path)
    else:
        # Write locally first so the worker can read it
        local_path = Path(job_dir) / "inputs" / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(file_bytes)

        # Also upload to S3 immediately so it's persisted
        key = f"{os.path.basename(job_dir)}/inputs/{filename}"
        backend.upload_file(str(local_path), key)
        return str(local_path)


def upload_job_artifacts(job_id: str, local_dir: str) -> list[str]:
    """Upload all artifacts from a local work directory to the backend.

    Called by the worker after generation completes.  For local backend
    this is a no-op (artifacts are already in the right place).
    """
    backend = _backend()
    return backend.upload_directory(local_dir, job_id)
