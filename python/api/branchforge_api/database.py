"""Compatibility shim for legacy imports.

Some routers still import `branchforge_api.database` while the canonical
module name is now `branchforge_api.db`.
"""

from .db import *  # noqa: F401,F403
