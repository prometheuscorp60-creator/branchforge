FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy core + api
COPY ../branchforge_core /app/branchforge_core
COPY . /app/api

# Install
RUN pip install --no-cache-dir -e /app/branchforge_core \
    && pip install --no-cache-dir -e /app/api

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "branchforge_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
