"""
tests/test_phase11.py
=====================
Comprehensive test suite for Phase 11: Observability & Tracing:
  1. Direct Prometheus Metrics Recording & Exposition (counters, histograms, gauges)
  2. Empirical Latency Percentile Calculations (p50, p90, p95, p99 quantiles)
  3. Distributed Tracing & Trace ID Injection (X-Trace-Id propagation)
  4. Public Prometheus Scrape Endpoints (GET /metrics and GET /api/v1/metrics)
  5. Authenticated Tenant Observability Summary (GET /api/v1/observability/summary)
  6. Streaming Chat Completion Observability & Trace Header Verification
  7. Circuit Breaker State Metric Gauge Synchronization
"""
import asyncio
from decimal import Decimal
import os
import sys
import time
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
import httpx

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")
django.setup()

from config.asgi import application
from gateway.adapters.base import close_shared_http_client
from gateway.models import APIKey, Tenant
from gateway.observability import (
    LatencyTracker,
    gateway_tracer,
    get_prometheus_metrics_text,
    record_cache_metric,
    record_request_metrics,
    update_circuit_breaker_metric,
)
from gateway.routing.circuit_breaker import circuit_breaker
from gateway.utils.redis import close_redis_client, get_redis_client
from tests.test_helpers import KNOWN_TEST_KEY, ensure_seed_data


@pytest.mark.django_db(transaction=True)
async def test_full_phase11_suite():
    print("\n--- Running Phase 11 Observability & Tracing Tests ---")
    tenant = await ensure_seed_data()
    redis = await get_redis_client()
    tenant_id_str = str(tenant.id)

    # Reset circuit breakers and latency history for clean test run
    for p in ["mock", "groq", "openai", "gemini"]:
        await circuit_breaker.reset(p)
    await LatencyTracker.clear(tenant_id_str)
    await LatencyTracker.clear()

    # -------------------------------------------------------------------------
    # 1. Direct Prometheus Metrics Recording
    # -------------------------------------------------------------------------
    print("\n[Step 1] Verifying Prometheus metrics recording and exposition...")
    record_request_metrics(
        tenant_id=tenant_id_str,
        model="mock/test-model",
        provider="mock",
        status_code=200,
        routing_strategy="latency_optimized",
        duration_seconds=0.125,
        prompt_tokens=15,
        completion_tokens=35,
        cost_usd=0.000045,
    )
    record_cache_metric(tenant_id_str, "hit")
    record_cache_metric(tenant_id_str, "miss")
    update_circuit_breaker_metric("mock", "CLOSED")
    update_circuit_breaker_metric("groq", "OPEN")

    metrics_text = get_prometheus_metrics_text().decode("utf-8")
    assert "llm_gateway_requests_total" in metrics_text
    assert "llm_gateway_request_duration_seconds" in metrics_text
    assert "llm_gateway_tokens_total" in metrics_text
    assert "llm_gateway_cost_usd_total" in metrics_text
    assert "llm_gateway_cache_requests_total" in metrics_text
    assert "llm_gateway_circuit_breaker_state" in metrics_text
    assert 'provider="mock"' in metrics_text
    assert 'provider="groq"' in metrics_text
    print("[PASS] Direct Prometheus metrics successfully recorded and exported.")

    # -------------------------------------------------------------------------
    # 2. Empirical Latency Percentile Tracker (p50, p90, p95, p99)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying LatencyTracker empirical quantiles...")
    await LatencyTracker.clear(tenant_id_str)

    # Feed 10 deterministic latencies: 10ms, 20ms, ..., 100ms
    for latency in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        await LatencyTracker.record_latency(tenant_id_str, latency, model="mock/test-model")

    percentiles = await LatencyTracker.get_percentiles(tenant_id_str)
    assert percentiles["samples_count"] == 10
    # Median of [10..100] is 55.0
    assert 50.0 <= percentiles["p50_ms"] <= 60.0
    assert percentiles["p90_ms"] >= 85.0
    assert percentiles["p95_ms"] >= 90.0
    assert percentiles["p99_ms"] >= 95.0
    print(f"[PASS] Latency quantiles verified: p50={percentiles['p50_ms']}ms, p90={percentiles['p90_ms']}ms, "
          f"p95={percentiles['p95_ms']}ms, p99={percentiles['p99_ms']}ms")

    # -------------------------------------------------------------------------
    # 3. Distributed Tracing & Trace ID Generation
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying distributed tracing and trace export...")
    trace_id = gateway_tracer.generate_trace_id()
    assert trace_id.startswith("tr_")
    assert len(trace_id) > 10

    # Test export_trace without crashing (handles Langfuse or structured logging)
    await gateway_tracer.export_trace(
        trace_id=trace_id,
        tenant_id=tenant_id_str,
        model="mock/test-model",
        provider="mock",
        prompt_tokens=10,
        completion_tokens=20,
        latency_ms=45.2,
        cost_usd=0.00001,
        status="success",
    )
    print(f"[PASS] Distributed trace {trace_id} verified.")

    # -------------------------------------------------------------------------
    # 4. End-to-End Scrape Endpoints (GET /metrics and GET /api/v1/metrics)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying HTTP Prometheus scrape endpoints...")
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # A. Root Prometheus Scrape Endpoint
        resp_root = await client.get("/metrics")
        assert resp_root.status_code == 200
        assert "text/plain" in resp_root.headers.get("content-type", "")
        assert "llm_gateway_requests_total" in resp_root.text

        # B. Ninja Sub-Route Prometheus Scrape Endpoint
        resp_ninja = await client.get("/api/v1/metrics")
        assert resp_ninja.status_code == 200
        assert "text/plain" in resp_ninja.headers.get("content-type", "")
        assert "llm_gateway_requests_total" in resp_ninja.text
        print("[PASS] HTTP Prometheus scrape endpoints (root and API) verified.")

        # ---------------------------------------------------------------------
        # 5. End-to-End Chat Completion Tracing & Headers
        # ---------------------------------------------------------------------
        print("\n[Step 5] Verifying trace propagation and observability headers on chat completions...")
        # Custom Trace ID propagation
        custom_trace = f"tr_test_{int(time.time())}"
        resp_custom = await client.post(
            "/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {KNOWN_TEST_KEY}",
                "X-Trace-Id": custom_trace,
                "Cache-Control": "no-cache",
            },
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Observability trace test"}],
            },
        )
        assert resp_custom.status_code == 200
        assert resp_custom.headers.get("x-trace-id") == custom_trace, (
            f"Expected {custom_trace}, got {resp_custom.headers.get('x-trace-id')}"
        )

        # Auto-generated Trace ID verification
        resp_auto = await client.post(
            "/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {KNOWN_TEST_KEY}",
                "Cache-Control": "no-cache",
            },
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Auto trace test"}],
            },
        )
        assert resp_auto.status_code == 200
        auto_trace = resp_auto.headers.get("x-trace-id")
        assert auto_trace and auto_trace.startswith("tr_"), f"Invalid trace header: {auto_trace}"
        print(f"[PASS] Chat completion trace propagation verified: {auto_trace}")

        # ---------------------------------------------------------------------
        # 6. SSE Streaming Chat Completion Trace Header Verification
        # ---------------------------------------------------------------------
        print("\n[Step 6] Verifying trace propagation on streaming chat completion...")
        resp_stream = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Streaming observability test"}],
                "stream": True,
            },
        )
        assert resp_stream.status_code == 200
        stream_trace = resp_stream.headers.get("x-trace-id")
        assert stream_trace and stream_trace.startswith("tr_")
        assert "text/event-stream" in resp_stream.headers.get("content-type", "")
        print(f"[PASS] Streaming completion trace header verified: {stream_trace}")

        # ---------------------------------------------------------------------
        # 7. Authenticated Observability Summary Endpoint (GET /api/v1/observability/summary)
        # ---------------------------------------------------------------------
        print("\n[Step 7] Verifying GET /api/v1/observability/summary endpoint...")
        resp_summary = await client.get(
            "/api/v1/observability/summary",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
        )
        assert resp_summary.status_code == 200, f"Error: {resp_summary.text}"
        summary_data = resp_summary.json()

        assert summary_data["tenant_id"] == tenant_id_str
        assert "total_requests" in summary_data
        assert "total_tokens" in summary_data
        assert "prompt_tokens" in summary_data
        assert "completion_tokens" in summary_data
        assert "total_cost_usd" in summary_data
        assert "latency_percentiles" in summary_data

        p_data = summary_data["latency_percentiles"]
        assert "p50_ms" in p_data
        assert "p90_ms" in p_data
        assert "p95_ms" in p_data
        assert "p99_ms" in p_data
        assert "samples_count" in p_data
        assert p_data["samples_count"] > 0
        print(f"[PASS] Observability summary verified: requests={summary_data['total_requests']}, "
              f"tokens={summary_data['total_tokens']}, samples={p_data['samples_count']}")

        # ---------------------------------------------------------------------
        # 8. Circuit Breaker Metric Gauge Synchronization
        # ---------------------------------------------------------------------
        print("\n[Step 8] Verifying circuit breaker metric gauge synchronization...")
        reset_resp = await client.post(
            "/api/v1/routing/circuits/mock/reset",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
        )
        assert reset_resp.status_code == 200
        scrape_after_reset = (await client.get("/metrics")).text
        assert 'llm_gateway_circuit_breaker_state{provider="mock"} 0.0' in scrape_after_reset
        print("[PASS] Circuit breaker metric gauge synchronization verified.")

    # Teardown connection pools
    await close_redis_client()
    await close_shared_http_client()
    print("\n[SUCCESS] All Phase 11 Observability & Tracing tests passed cleanly!")
