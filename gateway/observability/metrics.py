"""
gateway/observability/metrics.py
================================
Prometheus metrics collector and recorder for the LLM Gateway.

Tracks request counters, latency histograms (with buckets for p50/p95/p99),
token counters, cost accumulators, cache hits/misses, and circuit breaker gauges.
"""
import logging
from typing import Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)

# Dedicated registry for all LLM gateway metrics to prevent collisions.
gateway_registry = CollectorRegistry(auto_describe=True)

# 1. Total request count partitioned by tenant, model, provider, status code, and strategy
llm_gateway_requests_total = Counter(
    "llm_gateway_requests_total",
    "Total LLM gateway requests processed",
    ["tenant_id", "model", "provider", "status_code", "routing_strategy"],
    registry=gateway_registry,
)

# 2. Request duration histogram with standard latency buckets for p50, p90, p95, p99 quantiles
llm_gateway_request_duration_seconds = Histogram(
    "llm_gateway_request_duration_seconds",
    "LLM gateway request duration in seconds",
    ["tenant_id", "model", "provider"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0],
    registry=gateway_registry,
)

# 3. Token counter partitioned by token type (prompt, completion, total)
llm_gateway_tokens_total = Counter(
    "llm_gateway_tokens_total",
    "Total tokens processed by LLM gateway",
    ["tenant_id", "model", "provider", "token_type"],
    registry=gateway_registry,
)

# 4. Total FinOps cost incurred in USD
llm_gateway_cost_usd_total = Counter(
    "llm_gateway_cost_usd_total",
    "Total FinOps cost in USD incurred across requests",
    ["tenant_id", "model", "provider"],
    registry=gateway_registry,
)

# 5. Cache telemetry counter (hit, miss, bypass)
llm_gateway_cache_requests_total = Counter(
    "llm_gateway_cache_requests_total",
    "Total prompt cache requests by hit/miss/bypass status",
    ["tenant_id", "status"],
    registry=gateway_registry,
)

# 6. Upstream circuit breaker state gauge (0=CLOSED, 1=HALF_OPEN, 2=OPEN)
llm_gateway_circuit_breaker_state = Gauge(
    "llm_gateway_circuit_breaker_state",
    "Circuit breaker state for upstream provider (0=CLOSED, 1=HALF_OPEN, 2=OPEN)",
    ["provider"],
    registry=gateway_registry,
)


def record_request_metrics(
    tenant_id: str,
    model: str,
    provider: str,
    status_code: int,
    routing_strategy: str,
    duration_seconds: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Record request duration, counters, tokens, and FinOps cost in Prometheus."""
    tenant_id_str = str(tenant_id)
    model_str = str(model)
    provider_str = str(provider)
    status_str = str(status_code)
    strategy_str = str(routing_strategy)

    try:
        # Increment request counter
        llm_gateway_requests_total.labels(
            tenant_id=tenant_id_str,
            model=model_str,
            provider=provider_str,
            status_code=status_str,
            routing_strategy=strategy_str,
        ).inc()

        # Record duration in histogram
        llm_gateway_request_duration_seconds.labels(
            tenant_id=tenant_id_str,
            model=model_str,
            provider=provider_str,
        ).observe(max(0.0001, float(duration_seconds)))

        # Record prompt, completion, and total tokens
        if prompt_tokens > 0:
            llm_gateway_tokens_total.labels(
                tenant_id=tenant_id_str,
                model=model_str,
                provider=provider_str,
                token_type="prompt",
            ).inc(prompt_tokens)

        if completion_tokens > 0:
            llm_gateway_tokens_total.labels(
                tenant_id=tenant_id_str,
                model=model_str,
                provider=provider_str,
                token_type="completion",
            ).inc(completion_tokens)

        total_tokens = prompt_tokens + completion_tokens
        if total_tokens > 0:
            llm_gateway_tokens_total.labels(
                tenant_id=tenant_id_str,
                model=model_str,
                provider=provider_str,
                token_type="total",
            ).inc(total_tokens)

        # Record cost
        if cost_usd > 0:
            llm_gateway_cost_usd_total.labels(
                tenant_id=tenant_id_str,
                model=model_str,
                provider=provider_str,
            ).inc(float(cost_usd))
    except Exception as exc:
        logger.warning("Error updating Prometheus request metrics: %s", exc)


def record_cache_metric(tenant_id: str, status: str) -> None:
    """Record cache hit, miss, or bypass event."""
    try:
        llm_gateway_cache_requests_total.labels(
            tenant_id=str(tenant_id),
            status=status.lower(),
        ).inc()
    except Exception as exc:
        logger.warning("Error updating Prometheus cache metric: %s", exc)


def update_circuit_breaker_metric(provider: str, state: str) -> None:
    """Update circuit breaker gauge: CLOSED=0, HALF_OPEN=1, OPEN=2."""
    state_map = {"closed": 0, "half_open": 1, "open": 2}
    val = state_map.get(state.lower(), 0)
    try:
        llm_gateway_circuit_breaker_state.labels(provider=provider.lower()).set(val)
    except Exception as exc:
        logger.warning("Error updating Prometheus circuit breaker gauge: %s", exc)


def get_prometheus_metrics_text() -> bytes:
    """Generate latest Prometheus scrape payload as bytes."""
    return generate_latest(gateway_registry)
