from __future__ import annotations

from redis import Redis
from rq import Worker, Queue, Connection

from .settings import settings

listen = ["branchforge"]

def main():
    redis_conn = Redis.from_url(settings.redis_url)
    with Connection(redis_conn):
        worker = Worker([Queue(name) for name in listen])
        worker.work(with_scheduler=False)

if __name__ == "__main__":
    main()
