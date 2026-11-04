"""
gateway/health/schemas.py
=========================
Pydantic v2 schemas for health check API responses.

Why Pydantic schemas instead of raw dicts?
  1. Django Ninja auto-generates OpenAPI docs from these models — no manual spec.
  2. Pydantic v2 validates outgoing data at the boundary — bugs in internal
     logic cannot leak malformed responses to clients.
  3. Full IDE autocompletion and mypy type-checking across the codebase.
  4. `model_json_schema()` produces the JSON Schema embedded in openapi.json.

Design:
  - ServiceStatus is a str Enum so it serialises as a plain string in JSON
    (e.g. "ok") rather than an integer code.
  - ComponentHealth captures per-dependency status + latency, making it
    trivial to add new components (e.g., S3, Langfuse) in later phases.
  - HealthResponse is the top-level shape — used for both the full health
    endpoint and the exact shape required by the Phase 1 spec.
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ServiceStatus(str, Enum):
    """Canonical status values for any infrastructure component."""

    OK       = "ok"
    DEGRADED = "degraded"
    DOWN     = "down"


class ComponentHealth(BaseModel):
    """Health state of a single dependency (PostgreSQL, Redis, etc.)."""

    status: ServiceStatus = Field(
        ...,
        description="Current health status of this component.",
        examples=[ServiceStatus.OK],
    )
    latency_ms: float = Field(
        ...,
        description="Round-trip latency to this component in milliseconds.",
        ge=0,
        examples=[1.42],
    )
    detail: str | None = Field(
        default=None,
        description="Human-readable error detail. Present only on DEGRADED/DOWN.",
        examples=["Connection refused: redis://localhost:6379/0"],
    )


class HealthResponse(BaseModel):
    """
    Top-level health check response.

    Phase 1 required JSON shape:
      {
        "status":    "healthy" | "degraded" | "down",
        "postgres":  true | false,
        "redis":     true | false,
        "timestamp": "2024-01-15T12:00:00.000000Z"
      }

    We return both the simplified boolean flags (for spec compliance) and the
    richer `components` dict (for production observability).
    """

    status: str = Field(
        ...,
        description='Aggregate status: "healthy", "degraded", or "down".',
        examples=["healthy"],
    )
    postgres: bool = Field(
        ...,
        description="True if PostgreSQL is reachable.",
    )
    redis: bool = Field(
        ...,
        description="True if Redis is reachable.",
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of when the check was performed.",
        examples=["2024-01-15T12:00:00.000000Z"],
    )
    # Richer per-component breakdown (latency + detail) for observability
    components: dict[str, ComponentHealth] = Field(
        ...,
        description="Per-component health breakdown with latency measurements.",
    )
