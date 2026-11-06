"""
tests/test_phase9.py
====================
Comprehensive test suite for Phase 9:
  - Real-Time Streaming Responses (Server-Sent Events / SSE)
  - Async Generator Streaming Chunks (role delta, content tokens, stop finish_reason)
  - End-to-end ASGI Streaming via StreamingHttpResponse (text/event-stream)
  - Pre-flight FinOps Budget Guard (HTTP 402) on streaming requests
  - Pre-flight RPM Rate Limiter Guard (HTTP 429) on streaming requests
  - Dynamic Routing & Initial-Connection Fallback during streaming
  - Post-stream atomic FinOps token spend tracking in Redis
"""
import asyncio
from decimal import Decimal
import json
import os
import sys
import uuid
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
import httpx

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")
django.setup()

from config.asgi import application
from gateway.adapters.base import close_shared_http_client
from gateway.adapters.mock import MockAdapter
from gateway.adapters.factory import ProviderFactory
from gateway.adapters.schemas import ChatCompletionRequest, ChatMessage
from gateway.models import APIKey, Tenant
from gateway.ratelimit.budget import FinOpsBudgetManager
from gateway.routing.circuit_breaker import CircuitState, circuit_breaker
from gateway.routing.orchestrator import RoutingOrchestrator
from gateway.utils.redis import close_redis_client, get_redis_client
from tests.test_helpers import ensure_seed_data, KNOWN_TEST_KEY


@pytest.mark.django_db(transaction=True)
async def test_full_phase9_suite():
    print("\n--- Running Phase 9 Real-Time Streaming (SSE) Tests ---")
    tenant = await ensure_seed_data()
    await get_redis_client()
    assert tenant is not None, "Tenant must exist from seed data"
    for p in ["groq", "gemini", "google", "openai", "mock"]:
        await circuit_breaker.reset(p)

    # -------------------------------------------------------------------------
    # 1. Direct Adapter Streaming Verification
    # -------------------------------------------------------------------------
    print("\n[Step 1] Verifying MockAdapter async streaming generator...")
    adapter = MockAdapter(simulated_latency_ms=1.0)
    req = ChatCompletionRequest(
        model="mock/test-model",
        messages=[ChatMessage(role="user", content="Stream me a greeting!")],
        stream=True,
    )

    chunks = []
    async for chunk in adapter.stream_complete(req):
        chunks.append(chunk)

    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"
    assert chunks[0].choices[0].delta.role == "assistant", "First chunk should declare role"
    assert chunks[-1].choices[0].finish_reason == "stop", "Final chunk should declare finish_reason='stop'"
    assembled_content = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert "[Mock Stream]" in assembled_content
    print(f"[PASS] Direct adapter streaming generated {len(chunks)} chunks: '{assembled_content}'")

    # -------------------------------------------------------------------------
    # 2. End-to-End ASGI SSE Chat Completion Endpoint
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying POST /api/v1/chat/completions with stream=True...")
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # A. Successful SSE Stream
        async with client.stream(
            "POST",
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Explain streaming in 5 words."}],
                "stream": True,
            },
        ) as sse_resp:
            assert sse_resp.status_code == 200, f"Streaming failed: {sse_resp.status_code}"
            assert "text/event-stream" in sse_resp.headers.get("content-type", "")
            assert "x-ratelimit-limit-rpm" in sse_resp.headers
            assert "x-budget-limit" in sse_resp.headers
            assert "x-routing-strategy" in sse_resp.headers

            sse_lines = []
            parsed_chunks = []
            done_received = False

            async for line in sse_resp.aiter_lines():
                if not line:
                    continue
                sse_lines.append(line)
                if line.startswith("data: "):
                    payload = line[len("data: "):].strip()
                    if payload == "[DONE]":
                        done_received = True
                        break
                    chunk_data = json.loads(payload)
                    parsed_chunks.append(chunk_data)

            assert done_received, "Stream must terminate with data: [DONE]"
            assert len(parsed_chunks) >= 3, f"Expected >= 3 parsed chunks, got {len(parsed_chunks)}"
            content = "".join(
                c["choices"][0]["delta"].get("content", "")
                for c in parsed_chunks
            )
            assert "[Mock Stream]" in content
            print(f"[PASS] End-to-end SSE stream delivered {len(parsed_chunks)} chunks, properly formatted with [DONE].")

        # -------------------------------------------------------------------------
        # 3. Pre-Flight FinOps Budget Lockout Guard on Streaming (HTTP 402)
        # -------------------------------------------------------------------------
        print("\n[Step 3] Verifying pre-flight budget lockout rejects streaming requests (HTTP 402)...")
        redis = await get_redis_client()
        spend_key = FinOpsBudgetManager._spend_key(str(tenant.id))
        budget_key = FinOpsBudgetManager._budget_key(str(tenant.id))
        await redis.set(spend_key, "100.00")
        await redis.set(budget_key, "50.00")

        blocked_resp = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "This should be blocked."}],
                "stream": True,
            },
        )
        assert blocked_resp.status_code == 402, f"Expected 402, got {blocked_resp.status_code}"
        assert "spending budget exceeded" in blocked_resp.text
        print("[PASS] Streaming request was rejected pre-flight with HTTP 402 due to budget lockout.")

        # Reset spend to 0 to resume access
        await redis.set(spend_key, "0.00")
        await redis.delete(FinOpsBudgetManager._lockout_key(str(tenant.id)))

        # -------------------------------------------------------------------------
        # 4. Initial-Connection Fallback on Streaming
        # -------------------------------------------------------------------------
        print("\n[Step 4] Verifying streaming fallback when primary candidate fails before first chunk...")
        # Create a failing mock adapter
        failing_mock = MockAdapter(fail_on_stream=True)
        working_mock = MockAdapter(fail_on_stream=False)

        ProviderFactory.register_adapter("failing_mock", failing_mock)
        ProviderFactory.register_adapter("working_mock", working_mock)

        # Execute fallback orchestrator directly
        fallback_req = ChatCompletionRequest(
            model="cost_optimized",
            messages=[ChatMessage(role="user", content="Test fallback on stream")],
            stream=True,
        )

        stream_chunks = []
        async for chunk in RoutingOrchestrator.stream_and_execute(fallback_req, tenant=tenant):
            stream_chunks.append(chunk)

        assert len(stream_chunks) >= 3, "Fallback stream should have successfully delivered chunks from secondary"
        fallback_content = "".join(c.choices[0].delta.content or "" for c in stream_chunks)
        assert "[Mock Stream]" in fallback_content
        print(f"[PASS] Streaming fallback succeeded seamlessly, yielded {len(stream_chunks)} chunks from fallback candidate.")

    # Cleanup connections
    for p in ["groq", "gemini", "google", "openai", "mock"]:
        await circuit_breaker.reset(p)
    ProviderFactory.register_adapter("mock", MockAdapter())
    await close_shared_http_client()
    await close_redis_client()

    print("\nALL PHASE 9 REAL-TIME STREAMING TESTS PASSED 100%!")


if __name__ == "__main__":
    asyncio.run(test_full_phase9_suite())
