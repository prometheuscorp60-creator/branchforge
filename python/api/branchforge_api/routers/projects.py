"""3B: Projects and Workspaces + 3C: Design Version History API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, select, col

from ..db import get_session
from ..models import User, Project, Job, DesignSnapshot
from .auth import get_current_user

router = APIRouter()


# ── Schemas ──────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    description: str = ""

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    archived: Optional[bool] = None

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    owner_id: str
    owner_type: str
    created_at: str
    updated_at: str
    archived: bool
    job_count: int = 0
    latest_job_status: Optional[str] = None

class SnapshotCreate(BaseModel):
    spec_json: dict
    label: Optional[str] = None
    job_id: Optional[str] = None

class SnapshotResponse(BaseModel):
    id: str
    project_id: str
    user_id: str
    created_at: str
    label: Optional[str]
    job_id: Optional[str]

class VersionTimelineEntry(BaseModel):
    id: str
    created_at: str
    label: Optional[str]
    job_id: Optional[str]
    job_status: Optional[str] = None
    version_tag: Optional[str] = None


# ── Projects ─────────────────────────────────────────

@router.post("/projects", response_model=ProjectResponse)
async def create_project(
    body: ProjectCreate,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    project = Project(
        id=str(uuid.uuid4()),
        name=body.name,
        description=body.description,
        owner_id=user.org_id or user.id,
        owner_type="org" if user.org_id else "user",
    )
    session.add(project)
    session.commit()
    session.refresh(project)

    return ProjectResponse(
        **project.model_dump(),
        created_at=project.created_at.isoformat(),
        updated_at=project.updated_at.isoformat(),
    )


@router.get("/projects", response_model=List[ProjectResponse])
async def list_projects(
    archived: bool = Query(False),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    owner_id = user.org_id or user.id
    stmt = (
        select(Project)
        .where(Project.owner_id == owner_id, Project.archived == archived)
        .order_by(col(Project.updated_at).desc())
    )
    projects = session.exec(stmt).all()

    results = []
    for p in projects:
        jobs = session.exec(
            select(Job).where(Job.project_id == p.id)
            .order_by(col(Job.created_at).desc())
        ).all()
        latest_status = jobs[0].status if jobs else None
        results.append(ProjectResponse(
            **p.model_dump(),
            created_at=p.created_at.isoformat(),
            updated_at=p.updated_at.isoformat(),
            job_count=len(jobs),
            latest_job_status=latest_status,
        ))

    return results


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    owner_id = user.org_id or user.id
    if project.owner_id != owner_id:
        raise HTTPException(403, "Access denied")

    jobs = session.exec(select(Job).where(Job.project_id == project_id)).all()
    latest_status = jobs[0].status if jobs else None

    return ProjectResponse(
        **project.model_dump(),
        created_at=project.created_at.isoformat(),
        updated_at=project.updated_at.isoformat(),
        job_count=len(jobs),
        latest_job_status=latest_status,
    )


@router.patch("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    body: ProjectUpdate,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    owner_id = user.org_id or user.id
    if project.owner_id != owner_id:
        raise HTTPException(403, "Access denied")

    if body.name is not None:
        project.name = body.name
    if body.description is not None:
        project.description = body.description
    if body.archived is not None:
        project.archived = body.archived

    project.updated_at = datetime.utcnow()
    session.add(project)
    session.commit()
    session.refresh(project)

    return ProjectResponse(
        **project.model_dump(),
        created_at=project.created_at.isoformat(),
        updated_at=project.updated_at.isoformat(),
    )


# ── Design Version History (3C) ─────────────────────

@router.post("/projects/{project_id}/snapshots", response_model=SnapshotResponse)
async def create_snapshot(
    project_id: str,
    body: SnapshotCreate,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    snapshot = DesignSnapshot(
        id=str(uuid.uuid4()),
        project_id=project_id,
        user_id=user.id,
        spec_json=body.spec_json,
        label=body.label,
        job_id=body.job_id,
    )
    session.add(snapshot)

    project.updated_at = datetime.utcnow()
    session.add(project)
    session.commit()
    session.refresh(snapshot)

    return SnapshotResponse(
        **snapshot.model_dump(),
        created_at=snapshot.created_at.isoformat(),
    )


@router.get("/projects/{project_id}/timeline", response_model=List[VersionTimelineEntry])
async def get_timeline(
    project_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # Combine snapshots and jobs into a unified timeline
    snapshots = session.exec(
        select(DesignSnapshot)
        .where(DesignSnapshot.project_id == project_id)
        .order_by(col(DesignSnapshot.created_at).desc())
    ).all()

    jobs = session.exec(
        select(Job)
        .where(Job.project_id == project_id)
        .order_by(col(Job.created_at).desc())
    ).all()

    job_map = {j.id: j for j in jobs}

    timeline = []

    # Add snapshots
    for s in snapshots:
        job_status = None
        if s.job_id and s.job_id in job_map:
            job_status = job_map[s.job_id].status

        timeline.append(VersionTimelineEntry(
            id=s.id,
            created_at=s.created_at.isoformat(),
            label=s.label,
            job_id=s.job_id,
            job_status=job_status,
        ))

    # Add jobs without snapshots
    snapshot_job_ids = {s.job_id for s in snapshots if s.job_id}
    for j in jobs:
        if j.id not in snapshot_job_ids:
            timeline.append(VersionTimelineEntry(
                id=j.id,
                created_at=j.created_at.isoformat(),
                label=f"Job {j.id[:8]}",
                job_id=j.id,
                job_status=j.status,
                version_tag=j.version_tag,
            ))

    timeline.sort(key=lambda e: e.created_at, reverse=True)
    return timeline


@router.get("/projects/{project_id}/snapshots/{snapshot_id}")
async def get_snapshot_spec(
    project_id: str,
    snapshot_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    snapshot = session.get(DesignSnapshot, snapshot_id)
    if not snapshot or snapshot.project_id != project_id:
        raise HTTPException(404, "Snapshot not found")

    return {
        "id": snapshot.id,
        "created_at": snapshot.created_at.isoformat(),
        "label": snapshot.label,
        "spec_json": snapshot.spec_json,
        "job_id": snapshot.job_id,
    }
