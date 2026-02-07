# Architecture

BranchForge is intentionally split into three services:

- **API** (FastAPI): accepts uploads + specs, persists jobs/candidates in SQLite, serves downloads.
- **Worker** (RQ): runs the generator + CAD export on background jobs.
- **Web** (React): heatmap → candidate list → downloads.

Why this matters commercially:
- You can sell the *workflow* first. CFD stays in the customer's stack.
- CAD export is compute-heavy and unstable — keep it off the request thread.
- You can scale workers independently to control COGS.

## Data flow

1. User uploads heatmap (CSV or image) and (optionally) DXF outline.
2. API writes inputs under `/data/artifacts/<job_id>/inputs/`.
3. API enqueues an RQ job (`branchforge_worker.tasks.run_job(job_id)`).
4. Worker loads the JobSpec from SQLite, runs `branchforge_core.generator.generate_job`.
5. Worker writes artifacts under `/data/artifacts/<job_id>/candidates/`.
6. Worker inserts Candidate rows into SQLite.
7. Web UI polls job status and renders candidates + download links.

## Where the Nature rules live

`python/branchforge_core/branchforge_core/junctions.py`

- `RHO_THRESHOLD = 0.6`
- `CHI_TRIFURCATION = 0.83`
- `omega_from_theta` uses Ω = 4π sin²((π − θ)/4)
- `thick_angle_from_rho` morphs from sprout (θ=180°) toward symmetric branching (θ≈120°)

This is the core “differentiator layer” you can keep refining with calibration data.
