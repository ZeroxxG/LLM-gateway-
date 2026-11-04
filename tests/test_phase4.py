"""
tests/test_phase4.py
====================
Comprehensive test suite for Phase 4:
  - DynamicRouter strategy detection & candidate chain generation
  - Redis-backed Circuit Breaker state machine (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
  - Automatic fallback execution when primary provider fails
  - Live inspection & manual reset endpoints for circuit breakers
  - End-to-end chat completions with dynamic routing strategies ('cost_optimized', 'fast')
"""
import asyncio
from decimal import Decimal
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
import httpx

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")
django.setup()

from config.asgi import application
from gateway.adapters.base import BaseProviderAdapter, close_shared_http_client
from gateway.adapters.exceptions import ProviderAPIError
from gateway.adapters.factory import ProviderFactory
from gateway.adapters.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from gateway.models import APIKey, ModelPricing, Tenant
from gateway.routing.circuit_breaker import CircuitState, RedisCircuitBreaker, circuit_breaker
from gateway.routing.orchestrator import RoutingOrchestrator
from gateway.routing.router import DynamicRouter, RoutingStrategy


class FailingMockAdapter(BaseProviderAdapter):
    """Failing adapter used to trigger circuit trips and fallbacks."""

    provider_name: str = "failing"

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise ProviderAPIError("Simulated upstream provider outage (HTTP 503)", provider=self.provider_name, status_code=503)


async def test_full_phase4_suite():
    print("\n--- Running Phase 4 Dynamic Routing, Circuit Breaker & Fallback Tests ---")

    # 1. Dynamic Router Strategy Detection
    assert DynamicRouter.detect_strategy("cost_optimized") == RoutingStrategy.COST_OPTIMIZED
    assert DynamicRouter.detect_strategy("cheapest") == RoutingStrategy.COST_OPTIMIZED
    assert DynamicRouter.detect_strategy("fast") == RoutingStrategy.LATENCY_OPTIMIZED
    assert DynamicRouter.detect_strategy("premium") == RoutingStrategy.QUALITY_OPTIMIZED
    assert DynamicRouter.detect_strategy("groq/llama-3.1-8b") == RoutingStrategy.DIRECT
    print("[PASS] DynamicRouter strategy detection verified.")

    # 2. Candidate Chain Resolution
    cost_chain = await DynamicRouter.resolve_candidate_chain("cost_optimized")
    assert len(cost_chain) >= 2
    assert "mock/test-model" in cost_chain

    direct_chain = await DynamicRouter.resolve_candidate_chain("groq/llama-3.1-8b")
    assert direct_chain[0] == "groq/llama-3.1-8b"
    assert "google/gemini-1.5-flash" in direct_chain or "mock/test-model" in direct_chain
    print(f"[PASS] Candidate chains resolved correctly (Cost chain: {cost_chain[:3]}).")

    # 3. Redis Circuit Breaker State Machine Tests
    test_cb = RedisCircuitBreaker(failure_threshold=2, recovery_timeout=3.0, failure_window=15)
    provider_test = "test_flaky_provider"

    # Reset test provider
    await test_cb.reset(provider_test)
    assert await test_cb.get_state(provider_test) == CircuitState.CLOSED
    assert await test_cb.can_execute(provider_test) is True
    print("[PASS] Circuit Breaker initial state is CLOSED.")

    # Record 1 failure -> below threshold (remains CLOSED)
    s1 = await test_cb.record_failure(provider_test, reason="First timeout")
    assert s1 == CircuitState.CLOSED
    assert await test_cb.can_execute(provider_test) is True

    # Record 2nd failure -> reaches threshold (trips to OPEN)
    s2 = await test_cb.record_failure(provider_test, reason="Second timeout")
    assert s2 == CircuitState.OPEN
    assert await test_cb.get_state(provider_test) == CircuitState.OPEN
    assert await test_cb.can_execute(provider_test) is False
    print("[PASS] Circuit Breaker successfully tripped from CLOSED -> OPEN after 2 failures.")

    # Verify recovery timeout transition: wait 3.2s -> transitions to HALF_OPEN
    await asyncio.sleep(3.2)
    state_after_cooldown = await test_cb.get_state(provider_test)
    assert state_after_cooldown == CircuitState.HALF_OPEN
    assert await test_cb.can_execute(provider_test) is True
    print("[PASS] Circuit Breaker transitioned OPEN -> HALF_OPEN after recovery cooldown.")

    # Success in HALF_OPEN resets circuit to CLOSED
    await test_cb.record_success(provider_test)
    assert await test_cb.get_state(provider_test) == CircuitState.CLOSED
    print("[PASS] Successful request in HALF_OPEN recovered circuit to CLOSED.")

    # Clean up test keys
    await test_cb.reset(provider_test)

    # 4. Fallback Execution Test via RoutingOrchestrator
    # Register temporary failing adapter for "failing_provider"
    ProviderFactory.register_adapter("failing", FailingMockAdapter())
    await circuit_breaker.reset("failing")

    # Construct request with a primary model mapped to failing provider and fallback to mock
    DynamicRouter.detect_strategy("failing/broken-model")
    # Temporarily monkeypatch fallback chain
    from gateway.routing.router import DEFAULT_FALLBACK_CHAINS
    DEFAULT_FALLBACK_CHAINS["failing/broken-model"] = ["mock/test-model"]

    tenant = await Tenant.objects.aget(name="Acme Global (Test Org)")
    req = ChatCompletionRequest(
        model="failing/broken-model",
        messages=[ChatMessage(role="user", content="Will you fall back?")],
        temperature=0.5,
    )

    fallback_response = await RoutingOrchestrator.route_and_execute(request=req, tenant=tenant)
    assert fallback_response.provider == "mock"
    assert "mock" in fallback_response.model
    print(f"[PASS] Automatic fallback executed! Requested 'failing/broken-model' -> Served by '{fallback_response.provider}'.")

    # Verify that the failing provider circuit recorded the error
    failing_status = await circuit_breaker.get_circuit_status("failing")
    assert failing_status["consecutive_failures"] >= 1
    print(f"[PASS] Circuit breaker tracked failure on primary: {failing_status['consecutive_failures']} failure(s).")
    await circuit_breaker.reset("failing")

    # 5. End-to-End ASGI HTTP Client Testing
    valid_key = "gw_live_e6ed9d77b3841b97dadd97bb030f4f5a"
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # A. Query live circuit breakers endpoint
        res = await client.get("/api/v1/routing/circuits")
        assert res.status_code == 200
        circuits_data = res.json()
        assert len(circuits_data) >= 4
        providers_in_circuits = [c["provider"] for c in circuits_data]
        assert "groq" in providers_in_circuits
        assert "openai" in providers_in_circuits
        assert "gemini" in providers_in_circuits
        assert "mock" in providers_in_circuits
        print(f"[PASS] GET /api/v1/routing/circuits returned {len(circuits_data)} monitored circuits.")

        # B. Test Chat Completion with Dynamic Strategy: 'cost_optimized'
        res_cost = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_key}"},
            json={
                "model": "cost_optimized",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
            },
        )
        assert res_cost.status_code == 200, f"Error: {res_cost.text}"
        cost_data = res_cost.json()
        assert cost_data["object"] == "chat.completion"
        assert cost_data["provider"] in ("groq", "gemini", "mock", "openai")
        print(f"[PASS] Dynamic 'cost_optimized' request routed to '{cost_data['model']}' ({cost_data['provider']}) with cost ${cost_data['cost_usd']}.")

        # C. Reset a circuit breaker via API
        res_reset = await client.post(
            "/api/v1/routing/circuits/groq/reset",
            headers={"Authorization": f"Bearer {valid_key}"},
        )
        assert res_reset.status_code == 200
        assert res_reset.json()["state"] == "CLOSED"
        print("[PASS] POST /api/v1/routing/circuits/groq/reset succeeded.")

    # Clean up connections
    await close_shared_http_client()
    from gateway.utils.redis import close_redis_client
    await close_redis_client()

    print("\nALL PHASE 4 DYNAMIC ROUTING & CIRCUIT BREAKER TESTS PASSED 100%!")


if __name__ == "__main__":
    asyncio.run(test_full_phase4_suite())
