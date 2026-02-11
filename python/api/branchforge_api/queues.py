"""RQ queue configuration with plan-based priority.

Three queues in priority order (RQ serves high first):

* ``branchforge-high``  — Enterprise and Team plans
* ``branchforge``       — Pro plan (default)
* ``branchforge-low``   — Free plan
"""
from __future__ import annotations

from redis import Redis
from rq import Queue

from branchforge_core.plan_limits import get_queue_name
from .settings import settings

redis_conn = Redis.from_url(settings.redis_url)

# Priority queues — worker listens in this order
queue_high = Queue("branchforge-high", connection=redis_conn, default_timeout=60 * 30)
queue_default = Queue("branchforge", connection=redis_conn, default_timeout=60 * 30)
queue_low = Queue("branchforge-low", connection=redis_conn, default_timeout=60 * 30)

# Legacy alias — callers that haven't been updated yet
queue = queue_default

_QUEUE_MAP = {
    "branchforge-high": queue_high,
    "branchforge": queue_default,
    "branchforge-low": queue_low,
}


def get_queue_for_plan(plan: str) -> Queue:
    """Return the appropriate queue for the user's subscription plan."""
    name = get_queue_name(plan)
    return _QUEUE_MAP.get(name, queue_default)
