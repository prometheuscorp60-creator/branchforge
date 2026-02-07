# CadQuery + OpenCascade prebuilt environment (huge but reliable)
FROM ghcr.io/cadquery/cadquery:latest

WORKDIR /app

# Copy core + worker
COPY ../branchforge_core /app/branchforge_core
COPY . /app/worker

RUN pip install --no-cache-dir -e /app/branchforge_core[cad] \
    && pip install --no-cache-dir -e /app/worker

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "branchforge_worker.worker"]
