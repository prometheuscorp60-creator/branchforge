from __future__ import annotations

from redis import Redis
from rq import Queue

from .settings import settings

redis_conn = Redis.from_url(settings.redis_url)
queue = Queue("branchforge", connection=redis_conn, default_timeout=60*30)  # 30 min
