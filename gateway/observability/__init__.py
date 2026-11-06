"""
gateway/observability
=====================
Observability, Prometheus metrics, and distributed tracing for LLM Gateway.
"""
from gateway.observability.latency import LatencyTracker
from gateway.observability.metrics import (
    CONTENT_TYPE_LATEST,
    gateway_registry,
    get_prometheus_metrics_text,
    llm_gateway_cache_requests_total,
    llm_gateway_circuit_breaker_state,
    llm_gateway_cost_usd_total,
    llm_gateway_request_duration_seconds,
    llm_gateway_requests_total,
    llm_gateway_tokens_total,
    record_cache_metric,
    record_request_metrics,
    update_circuit_breaker_metric,
)
from gateway.observability.tracer import GatewayTracer, gateway_tracer

__all__ = [
    "gateway_registry",
    "get_prometheus_metrics_text",
    "record_request_metrics",
    "record_cache_metric",
    "update_circuit_breaker_metric",
    "llm_gateway_requests_total",
    "llm_gateway_request_duration_seconds",
    "llm_gateway_tokens_total",
    "llm_gateway_cost_usd_total",
    "llm_gateway_cache_requests_total",
    "llm_gateway_circuit_breaker_state",
    "CONTENT_TYPE_LATEST",
    "LatencyTracker",
    "GatewayTracer",
    "gateway_tracer",
]
