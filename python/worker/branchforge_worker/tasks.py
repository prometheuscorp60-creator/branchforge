from __future__ import annotations

import uuid
import traceback

from sqlmodel import select

from branchforge_core.schemas import JobSpec
from branchforge_core.generator import generate_job

from .db import get_session
from .models import Job, Candidate


def run_job(job_id: str) -> None:
    session = get_session()
    try:
        job = session.get(Job, job_id)
        if job is None:
            return
        job.status = "running"
        job.error = None
        session.add(job)
        session.commit()

        spec = JobSpec.model_validate(job.spec_json)

        candidates = generate_job(spec, job.job_dir)

        # wipe old candidates if any (idempotence)
        old = session.exec(select(Candidate).where(Candidate.job_id == job_id)).all()
        for oc in old:
            session.delete(oc)
        session.commit()

        for c in candidates:
            cand = Candidate(
                id=str(uuid.uuid4()),
                job_id=job_id,
                index=c.index,
                label=c.label,
                metrics_json=c.metrics.model_dump(),
                artifacts_json=c.artifacts.model_dump(),
            )
            session.add(cand)
        session.commit()

        job.status = "succeeded"
        session.add(job)
        session.commit()
    except Exception as e:
        # mark failed
        tb = traceback.format_exc()
        try:
            job = session.get(Job, job_id)
            if job is not None:
                job.status = "failed"
                job.error = f"{e}\n\n{tb}"
                session.add(job)
                session.commit()
        except Exception:
            pass
    finally:
        session.close()
