"""
gateway/health/router.py
========================
Health check endpoints — the most important operational endpoints in any service.

Route map (mounted at /api/v1/health by gateway/api.py):
  GET /api/v1/health/       → Full health report (Phase 1 required endpoint)
  GET /api/v1/health/live   → Kubernetes liveness probe
  GET /api/v1/health/ready  → Kubernetes readiness probe

Phase 1 Required Response Shape for GET /api/v1/health/:
  {
    "status":    "healthy",
    "postgres":  true,
    "redis":     true,
    "timestamp": "2024-01-15T12:00:00.000000Z",
    "components": { ... }    ← bonus observability data
  }

Async Design:
  asyncio.gather() runs _check_redis() and _check_database() CONCURRENTLY.
  Total response time ≈ max(redis_rtt, db_rtt), NOT redis_rtt + db_rtt.
  This is the event-loop advantage: both I/O-bound probes start immediately
  and the event loop switches between them while each awaits its network reply.

Blocking ORM Bridge:
  Django's raw SQL cursor (connection.cursor()) is synchronous even in 5.x
  with psycopg3. Wrapping it with @sync_to_async submits the blocking call
  to Django's thread pool (ThreadPoolExecutor), yielding the event loop
  thread until the DB query completes. This prevents a 10 ms SELECT from
  blocking all other concurrent requests.
"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from asgiref.sync import sync_to_async
from ninja import Router

from gateway.health.schemas import ComponentHealth, HealthResponse, ServiceStatus
from gateway.utils.redis import get_redis_client

logger = logging.getLogger(__name__)

router = Router()


# ---------------------------------------------------------------------------
# Internal probe helpers
# Each returns a ComponentHealth and NEVER raises — errors are encoded
# into the schema so one failing component does not block others' reporting.
# ---------------------------------------------------------------------------


async def _check_redis() -> ComponentHealth:
    """
    Ping Redis and measure round-trip latency.

    PING is the lightest Redis command — O(1), no data touched.
    Uses the module-level singleton async client (lazy-initialised).

    Returns:
        ComponentHealth with status=OK on success, DOWN on any exception.
    """
    start = time.perf_counter()     # perf_counter has nanosecond resolution
    try:
        client = await get_redis_client()
        pong = await client.ping()  # Awaits the network round-trip
        latency_ms = (time.perf_counter() - start) * 1000

        return ComponentHealth(
            status=ServiceStatus.OK if pong else ServiceStatus.DEGRADED,
            latency_ms=round(latency_ms, 3),
            detail=None if pong else "Redis PING returned unexpected response.",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("Redis health check failed: %s", exc)
        return ComponentHealth(
            status=ServiceStatus.DOWN,
            latency_ms=round(latency_ms, 3),
            detail=str(exc),
        )


@sync_to_async
def _sync_db_ping() -> float:
    """
    Execute 'SELECT 1' via Django ORM cursor and return elapsed ms.

    @sync_to_async decorator:
      - Submits this entire function to Django's thread pool executor
      - The event loop thread is NOT blocked while the SQL round-trip happens
      - Django manages the pool size (default: min(32, CPU*5+4) threads)

    Why not use async ORM directly?
      Django 5.x async ORM (afilter, aget) works for model queries but
      raw cursor.execute() is still synchronous — hence sync_to_async here.
    """
    from django.db import connection  # Deferred import avoids circular refs

    start = time.perf_counter()
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        cursor.fetchone()           # Must consume the result to close the cursor
    return (time.perf_counter() - start) * 1000


async def _check_database() -> ComponentHealth:
    """
    Ping PostgreSQL and measure round-trip latency.

    Delegates the blocking ORM call to Django's thread pool via
    _sync_db_ping (decorated with sync_to_async).
    """
    try:
        latency_ms = await _sync_db_ping()
        return ComponentHealth(
            status=ServiceStatus.OK,
            latency_ms=round(latency_ms, 3),
        )
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        return ComponentHealth(
            status=ServiceStatus.DOWN,
            latency_ms=0.0,
            detail=str(exc),
        )


def _aggregate_status(
    db_ok: bool,
    redis_ok: bool,
    db_health: ComponentHealth,
    redis_health: ComponentHealth,
) -> str:
    """
    Compute human-readable aggregate status string.

    Priority: DOWN > DEGRADED > healthy
    Returns string literals matching the Phase 1 spec ("healthy"/"degraded"/"down").
    """
    if not db_ok or not redis_ok:
        # If both are DOWN → "down"; if any is degraded → "degraded"
        statuses = {db_health.status, redis_health.status}
        if ServiceStatus.DOWN in statuses:
            return "down"
        return "degraded"
    return "healthy"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response=HealthResponse,
    summary="Full Health Report",
    description=(
        "Returns the live health status of PostgreSQL and Redis, measured "
        "concurrently via asyncio.gather(). "
        "Response matches Phase 1 specification exactly."
    ),
    operation_id="health_check",
)
async def health_check(request: Any) -> HealthResponse:
    """
    GET /api/v1/health/

    Concurrency mechanics:
      asyncio.gather() schedules both coroutines on the running event loop.
      While _check_redis() awaits the TCP round-trip to Redis, the event
      loop runs _check_database() (which runs in the thread pool). Both
      complete in roughly max(redis_rtt, db_rtt) wall-clock time.

    Returns the Phase 1 required JSON:
      {
        "status":    "healthy" | "degraded" | "down",
        "postgres":  true | false,
        "redis":     true | false,
        "timestamp": "2024-01-15T12:00:00.000000Z",
        "components": { "database": {...}, "redis": {...} }
      }
    """
    # Run both probes concurrently — total time ≈ max(rtt_db, rtt_redis)
    redis_health, db_health = await asyncio.gather(
        _check_redis(),
        _check_database(),
        )


    postgres_ok = db_health.status == ServiceStatus.OK
    redis_ok    = redis_health.status == ServiceStatus.OK

    return HealthResponse(
        status=_aggregate_status(postgres_ok, redis_ok, db_health, redis_health),
        postgres=postgres_ok,
        redis=redis_ok,
        # ISO 8601 UTC timestamp with microseconds + Z suffix
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        components={
            "database": db_health,
            "redis":    redis_health,
        },
    )


@router.get(
    "/live",
    summary="Liveness Probe",
    description=(
        "Kubernetes liveness probe — returns 200 if the ASGI process is alive. "
        "Does NOT check downstream dependencies. If this fails, K8s restarts the pod."
    ),
    operation_id="liveness",
)
async def liveness(request: Any) -> dict[str, str]:
    """
    GET /api/v1/health/live

    Process-level check only. Should never return non-200 unless the
    Uvicorn worker itself is broken. Intentionally has no external I/O.
    """
    return {"status": "alive"}


@router.get(
    "/ready",
    response={200: HealthResponse, 503: HealthResponse},
    summary="Readiness Probe",
    description=(
        "Kubernetes readiness probe — returns 200 only when ALL dependencies "
        "are reachable. Returns 503 if any component is DOWN/DEGRADED, "
        "signalling K8s to stop routing traffic to this pod."
    ),
    operation_id="readiness",
)
async def readiness(request: Any) -> tuple[int, HealthResponse]:
    """
    GET /api/v1/health/ready

    Django Ninja supports returning (http_status, schema_instance) tuples
    when the `response` mapping declares multiple status codes.
    """
    redis_health, db_health = await asyncio.gather(
        _check_redis(),
        _check_database(),
    )

    postgres_ok = db_health.status == ServiceStatus.OK
    redis_ok    = redis_health.status == ServiceStatus.OK
    status_str  = _aggregate_status(postgres_ok, redis_ok, db_health, redis_health)

    response = HealthResponse(
        status=status_str,
        postgres=postgres_ok,
        redis=redis_ok,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        components={
            "database": db_health,
            "redis":    redis_health,
        },
    )

    http_code = 200 if status_str == "healthy" else 503
    return http_code, response
