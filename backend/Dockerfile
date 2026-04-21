# =====================================================================
# Multi-stage Docker build for the FastAPI backend.
#   - Stage 1 ("builder") installs deps into a virtualenv.
#   - Stage 2 ("runtime") copies only the venv + app code, runs as non-root.
# =====================================================================

ARG PYTHON_VERSION=3.11-slim-bookworm

# ---------- Builder ----------
FROM python:${PYTHON_VERSION} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install .

# ---------- Runtime ----------
FROM python:${PYTHON_VERSION} AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system app && useradd --system --gid app --home-dir /app app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY backend ./backend
COPY params.yaml ./params.yaml

RUN mkdir -p models/champion data/reference reports/drift && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/api/v1/monitoring/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
