"""Storage backend abstraction for BranchForge artifact persistence.

Supports two backends:

* **local** — Filesystem storage (default, for dev and single-node deploys).
* **s3**    — S3-compatible storage (AWS S3, Cloudflare R2, MinIO).

The generator always writes to a local temp/work directory.  After
generation completes, the worker uploads artifacts to the configured
backend.  The API reads/serves artifacts via the same backend.

Configuration (environment variables)::

    STORAGE_BACKEND=local          # local | s3
    ARTIFACTS_DIR=/data/artifacts  # base dir for local backend

    # S3 backend:
    S3_BUCKET=branchforge-artifacts
    S3_REGION=us-east-1
    S3_ACCESS_KEY_ID=...
    S3_SECRET_ACCESS_KEY=...
    S3_ENDPOINT_URL=               # for R2/MinIO

Usage::

    from branchforge_core.storage_backend import get_storage_backend
    backend = get_storage_backend()
    backend.upload_file("/tmp/plate.step", "abc123/candidates/cand_00/plate.step")
"""
from __future__ import annotations

import abc
import os
import shutil
from pathlib import Path
from typing import Optional


class StorageBackend(abc.ABC):
    """Abstract interface for artifact storage."""

    @abc.abstractmethod
    def upload_file(self, local_path: str, remote_key: str) -> None:
        """Upload a local file to the storage backend."""

    @abc.abstractmethod
    def download_file(self, remote_key: str, local_path: str) -> None:
        """Download a file from the backend to a local path."""

    @abc.abstractmethod
    def get_presigned_url(self, remote_key: str, expires_in: int = 3600) -> Optional[str]:
        """Return a temporary URL for direct download, or None if not supported."""

    @abc.abstractmethod
    def read_bytes(self, remote_key: str) -> bytes:
        """Read file contents as bytes."""

    @abc.abstractmethod
    def read_text(self, remote_key: str, encoding: str = "utf-8") -> str:
        """Read file contents as text."""

    @abc.abstractmethod
    def exists(self, remote_key: str) -> bool:
        """Check if a key exists in the backend."""

    @abc.abstractmethod
    def upload_directory(self, local_dir: str, prefix: str) -> list[str]:
        """Recursively upload a directory.  Returns list of uploaded keys."""

    @abc.abstractmethod
    def health_check(self) -> str:
        """Return 'ok' or an error description."""


# ═══════════════════════════════════════════════════════════════════
# Local filesystem backend
# ═══════════════════════════════════════════════════════════════════

class LocalBackend(StorageBackend):
    """Filesystem-based storage — the default for development."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _abs(self, key: str) -> str:
        return os.path.join(self.base_dir, key)

    def upload_file(self, local_path: str, remote_key: str) -> None:
        target = self._abs(remote_key)
        if os.path.abspath(local_path) != os.path.abspath(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(local_path, target)

    def download_file(self, remote_key: str, local_path: str) -> None:
        shutil.copy2(self._abs(remote_key), local_path)

    def get_presigned_url(self, remote_key: str, expires_in: int = 3600) -> Optional[str]:
        return None  # Local backend uses FileResponse directly

    def read_bytes(self, remote_key: str) -> bytes:
        with open(self._abs(remote_key), "rb") as f:
            return f.read()

    def read_text(self, remote_key: str, encoding: str = "utf-8") -> str:
        with open(self._abs(remote_key), "r", encoding=encoding, errors="replace") as f:
            return f.read()

    def exists(self, remote_key: str) -> bool:
        return os.path.exists(self._abs(remote_key))

    def upload_directory(self, local_dir: str, prefix: str) -> list[str]:
        # For local backend, if the dir is already in base_dir, no-op.
        abs_local = os.path.abspath(local_dir)
        abs_target = os.path.abspath(self._abs(prefix))
        if abs_local == abs_target:
            return []  # Already in place
        uploaded: list[str] = []
        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                src = os.path.join(root, fname)
                rel = os.path.relpath(src, local_dir)
                key = f"{prefix}/{rel}".replace("\\", "/")
                self.upload_file(src, key)
                uploaded.append(key)
        return uploaded

    def health_check(self) -> str:
        try:
            if os.path.isdir(self.base_dir) and os.access(self.base_dir, os.W_OK):
                return "ok"
            return f"directory not writable: {self.base_dir}"
        except Exception as e:
            return f"error: {e}"

    def get_local_path(self, remote_key: str) -> str:
        """Local-only convenience: return the absolute path for FileResponse."""
        return self._abs(remote_key)


# ═══════════════════════════════════════════════════════════════════
# S3-compatible backend (AWS S3 / Cloudflare R2 / MinIO)
# ═══════════════════════════════════════════════════════════════════

class S3Backend(StorageBackend):
    """S3-compatible object storage backend."""

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        access_key_id: str = "",
        secret_access_key: str = "",
        endpoint_url: Optional[str] = None,
    ) -> None:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage backend. "
                "Install it with: pip install boto3"
            )

        kwargs: dict = {"region_name": region}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            **kwargs,
        )
        self.bucket = bucket

    def upload_file(self, local_path: str, remote_key: str) -> None:
        self.s3.upload_file(local_path, self.bucket, remote_key)

    def download_file(self, remote_key: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket, remote_key, local_path)

    def get_presigned_url(self, remote_key: str, expires_in: int = 3600) -> Optional[str]:
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": remote_key},
            ExpiresIn=expires_in,
        )

    def read_bytes(self, remote_key: str) -> bytes:
        obj = self.s3.get_object(Bucket=self.bucket, Key=remote_key)
        return obj["Body"].read()

    def read_text(self, remote_key: str, encoding: str = "utf-8") -> str:
        return self.read_bytes(remote_key).decode(encoding)

    def exists(self, remote_key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=remote_key)
            return True
        except Exception:
            return False

    def upload_directory(self, local_dir: str, prefix: str) -> list[str]:
        uploaded: list[str] = []
        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel = os.path.relpath(local_path, local_dir)
                key = f"{prefix}/{rel}".replace("\\", "/")
                self.upload_file(local_path, key)
                uploaded.append(key)
        return uploaded

    def health_check(self) -> str:
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            return "ok"
        except Exception as e:
            return f"error: {e}"


# ═══════════════════════════════════════════════════════════════════
# Factory — lazily cached singleton
# ═══════════════════════════════════════════════════════════════════

_backend: Optional[StorageBackend] = None


def get_storage_backend() -> StorageBackend:
    """Return the configured storage backend (cached singleton)."""
    global _backend
    if _backend is not None:
        return _backend

    backend_type = os.getenv("STORAGE_BACKEND", "local")

    if backend_type == "s3":
        _backend = S3Backend(
            bucket=os.environ["S3_BUCKET"],
            region=os.getenv("S3_REGION", "us-east-1"),
            access_key_id=os.environ.get("S3_ACCESS_KEY_ID", ""),
            secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY", ""),
            endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
        )
    else:
        _backend = LocalBackend(os.getenv("ARTIFACTS_DIR", "/data/artifacts"))

    return _backend
