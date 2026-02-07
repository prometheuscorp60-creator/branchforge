# BranchForge (BranchForge MVP)

**Heat map → STEP in ~60 seconds** (locally; depends on machine).

This repo is a **commercial-style, shippable MVP** of *BranchForge*: a web app that converts a cold-plate outline + heat map + port locations + manufacturing constraints into a manufacturable **branching** supply/return cooling network with:

- **Ranked design candidates**
- **Pressure-drop estimate**
- **Temperature-uniformity estimate**
- **Exports**: STEP (plate & channel-void), STL, DXF, plus a PDF report

It bakes in the *Nature* / *string-theory mapping* insight as a **junction rule library**:
- **Sprouting regime** for thin branches (ρ < 0.6): main path stays straight, thin branch sprouts orthogonally.
- **Branching regime** for thicker branches (ρ ≥ 0.6): steering angle increases with ρ and approaches symmetric branching near ρ≈1.
- **Trifurcation emergence** for sufficiently “thick” links (χ ≳ 0.83): two close bifurcations collapse into a trifurcation.

These thresholds and the Ω↔θ mapping are taken from:
- Meng et al., *Surface optimization governs the local design of physical networks*, Nature (2026). (Open access via PMC.)

---

## Quick start (Docker)

### 1) Requirements
- Docker + Docker Compose

### 2) Run
```bash
cp .env.example .env
docker compose up --build
```

Then open:
- Web UI: http://localhost:5173
- API docs (OpenAPI): http://localhost:8000/docs

### API key / plans (MVP gating)
By default `.env.example` runs with **DEV_NO_AUTH=true**, which means:
- No API key required
- All downloads enabled

If you set `DEV_NO_AUTH=false`, requests must include `X-BranchForge-Key`.
You can create a key via the UI (Designer page) or via:
```bash
curl -X POST http://localhost:8000/api/v1/auth/create_key \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","plan":"pro"}'
```

Free plan is limited to **3 jobs per 30 days** and cannot download STEP.

### 3) Generate a design
In the web UI:
1) Choose a rectangular plate or upload a DXF outline
2) Upload a heatmap (CSV grid or grayscale image)
3) Pick inlet/outlet locations
4) Set constraints / preset
5) Click **Generate**

Results:
- A recommended candidate + 9 alternates
- Download STEP/STL/DXF and the PDF report

Artifacts are written to `./data/artifacts/`.

---

## Architecture (opinionated)

- **API**: FastAPI (Python)
- **Queue**: Redis + RQ
- **Worker**: Python, does heavy generation + CAD exports
- **Geometry**: Shapely (2D routing footprint) + CadQuery (STEP)
- **Frontend**: React (Vite)

Why this split:
- The API stays snappy and cheap.
- The worker can scale separately (more CPU/RAM).
- CAD export stays off the request thread.

---

## What is “manufacturable” here?

This MVP outputs a **2.5D CNC-friendly channel footprint**:
- We compute centerline paths.
- We buffer them into a channel footprint polygon.
- We extrude down to channel depth from the top surface.
- We subtract the channel void from the plate.

That gives a CAD solid a shop can actually use.
For AM / etched-lid workflows you’d add presets (v2).

---

## Folder layout

```
branchforge/
  docker-compose.yml
  .env.example
  data/                 # runtime volumes
  python/
    api/                # FastAPI service
    worker/             # RQ worker + generator
    branchforge_core/   # shared algorithm + exports
  web/                  # React app
```

---

## Dev notes

### Running without Docker
It’s possible, but Docker is the supported path for this MVP because CadQuery/OCC
can be annoying on bare metal.

---

## License

This code is provided as a **commercial-style MVP template**.
You are responsible for any licensing/compliance if you commercialize it.

Important note:
- The academic `min-surf-netw` repo is GPL-3.0 and Wolfram/Mathematica-based.
  This MVP does **not** use it and does **not** copy code from it.
  It only implements the empirically reported thresholds and relations described
  in the paper.
