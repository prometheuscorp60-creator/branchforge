FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy core library first (changes less often)
COPY branchforge_core /app/branchforge_core
RUN pip install --no-cache-dir -e /app/branchforge_core

# Copy and install API
COPY api /app/api
RUN pip install --no-cache-dir -e /app/api

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "branchforge_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
