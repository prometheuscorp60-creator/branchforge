"""Compatibility shim for legacy auth imports.

Routers may import `branchforge_api.auth` while auth endpoints live in
`branchforge_api.routers.auth`.
"""

from .routers.auth import get_current_user  # noqa: F401
