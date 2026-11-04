# =============================================================================
# LLM Gateway — Docker Image
# Multi-stage build: builder installs deps, runtime image is lean.
# =============================================================================

# ---- Stage 1: dependency builder ----
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System libs required by psycopg[binary] and hiredis
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


# ---- Stage 2: lean runtime ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=config.settings.production

WORKDIR /app

# Only runtime libs (libpq for psycopg runtime, curl for healthcheck probes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built site-packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Non-root user for security
RUN addgroup --system gateway && adduser --system --ingroup gateway gateway \
    && chown -R gateway:gateway /app
USER gateway

EXPOSE 8000

# Production: 4 uvicorn workers. For dev use docker-compose override.
CMD ["uvicorn", "config.asgi:application", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--access-log", \
     "--log-level", "info"]
