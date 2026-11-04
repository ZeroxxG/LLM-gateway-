"""
tests/test_phase3.py
====================
Integration test suite for Phase 3:
  - Provider Adapter execution (Mock, Groq, OpenAI, Gemini)
  - Schema conversions and message normalization
  - ProviderFactory routing and FinOps token cost enrichment
  - End-to-end ASGI execution of POST /api/v1/chat/completions
  - Tenant budget limits enforcement (HTTP 402)
"""
import asyncio
from decimal import Decimal
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
import httpx

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")
django.setup()

from config.asgi import application
from gateway.adapters.base import close_shared_http_client
from gateway.adapters.exceptions import ModelNotSupportedError
from gateway.adapters.factory import ProviderFactory
from gateway.adapters.gemini import GeminiAdapter
from gateway.adapters.mock import MockAdapter
from gateway.adapters.schemas import ChatCompletionRequest, ChatMessage
from gateway.models import APIKey, ModelPricing, ModelTier, Tenant


async def test_full_phase3_suite():
    print("\n--- Running Phase 3 Provider Adapters & Execution Tests ---")

    # 1. Fetch test tenant and API key
    tenant = await Tenant.objects.aget(name="Acme Global (Test Org)")
    api_key_obj = await APIKey.objects.filter(tenant=tenant, is_active=True).afirst()
    assert api_key_obj is not None, "Test API key must exist"
    valid_key = "gw_live_e6ed9d77b3841b97dadd97bb030f4f5a"

    # Ensure mock pricing entry exists in DB for FinOps cost calculation test
    mock_pricing, _ = await ModelPricing.objects.aupdate_or_create(
        model_id="mock/test-model",
        defaults={
            "provider": "mock",
            "input_cost_per_million": Decimal("0.1000"),
            "output_cost_per_million": Decimal("0.2000"),
            "tier": ModelTier.SMALL,
            "is_active": True,
        },
    )

    # 2. Test ProviderFactory Resolution
    provider_groq, _ = await ProviderFactory.resolve_provider("groq/llama-3.1-8b")
    assert provider_groq == "groq"

    provider_openai, _ = await ProviderFactory.resolve_provider("openai/gpt-4o-mini")
    assert provider_openai == "openai"

    provider_gemini, _ = await ProviderFactory.resolve_provider("google/gemini-1.5-flash")
    assert provider_gemini == "gemini"

    provider_mock, pricing_mock = await ProviderFactory.resolve_provider("mock/test-model")
    assert provider_mock == "mock"
    assert pricing_mock is not None
    print("[PASS] ProviderFactory resolution verified for all providers.")

    # 3. Test Unsupported Model Error
    try:
        await ProviderFactory.resolve_provider("nonexistent-vendor/unknown-model")
        assert False, "Should raise ModelNotSupportedError"
    except ModelNotSupportedError:
        print("[PASS] Unsupported model correctly caught by ModelNotSupportedError.")

    # 4. Test Gemini Schema Translation
    gemini_adapter = GeminiAdapter(api_key="test_key")
    messages = [
        ChatMessage(role="system", content="You are a helpful coding assistant."),
        ChatMessage(role="user", content="Hello, world!"),
        ChatMessage(role="assistant", content="Hi there!"),
        ChatMessage(role="user", content="Explain recursion."),
    ]
    sys_inst, contents = gemini_adapter._convert_messages(messages)
    assert sys_inst is not None
    assert sys_inst["parts"][0]["text"] == "You are a helpful coding assistant."
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"
    assert contents[2]["role"] == "user"
    print("[PASS] Gemini message format translation verified.")

    # 5. Direct MockAdapter Completion Execution
    mock_adapter = MockAdapter(simulated_latency_ms=10.0)
    req = ChatCompletionRequest(
        model="mock/test-model",
        messages=[ChatMessage(role="user", content="What is the speed of light?")],
        temperature=0.7,
        max_tokens=256,
    )
    direct_res = await mock_adapter.complete(req)
    assert direct_res.choices[0].message.role == "assistant"
    assert "What is the speed of light?" in direct_res.choices[0].message.content
    assert direct_res.usage.total_tokens > 0
    assert direct_res.latency_ms >= 5.0
    print(f"[PASS] Direct MockAdapter completed successfully in {direct_res.latency_ms}ms.")

    # 6. End-to-End ASGI HTTP Client Testing
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # A. Unauthenticated request to /chat/completions -> 401
        res = await client.post(
            "/api/v1/chat/completions",
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Ping"}],
            },
        )
        assert res.status_code == 401, f"Expected 401, got {res.status_code}"
        print("[PASS] POST /chat/completions rejects unauthenticated request (401).")

        # B. Invalid key -> 401
        res = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": "Bearer gw_live_badkey0000000000000000000000"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Ping"}],
            },
        )
        assert res.status_code == 401, f"Expected 401, got {res.status_code}"
        print("[PASS] POST /chat/completions rejects invalid bearer token (401).")

        # C. Valid request with Bearer key -> 200 OK
        initial_spend = (await Tenant.objects.aget(pk=tenant.pk)).current_spend_usd
        res = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_key}"},
            json={
                "model": "mock/test-model",
                "messages": [
                    {"role": "system", "content": "You are a concise mathematician."},
                    {"role": "user", "content": "What is 17 * 19?"},
                ],
                "temperature": 0.5,
                "max_tokens": 128,
            },
        )
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "mock/test-model"
        assert data["provider"] == "mock"
        assert len(data["choices"]) == 1
        assert "What is 17 * 19?" in data["choices"][0]["message"]["content"]
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert Decimal(str(data["cost_usd"])) > Decimal("0.0000")
        print(f"[PASS] POST /chat/completions executed successfully: cost=${data['cost_usd']}, latency={data['latency_ms']}ms.")

        # Verify tenant current_spend_usd was updated atomically
        updated_tenant = await Tenant.objects.aget(pk=tenant.pk)
        assert updated_tenant.current_spend_usd > initial_spend
        print(f"[PASS] FinOps atomic spend update confirmed: ${initial_spend} -> ${updated_tenant.current_spend_usd}.")

        # D. FinOps Budget Enforcement Test
        # Create a tenant with $0.00 budget and active key
        broke_tenant, _ = await Tenant.objects.aupdate_or_create(
            name="Exhausted Budget Corp",
            defaults={
                "monthly_budget_usd": Decimal("10.0000"),
                "current_spend_usd": Decimal("10.0000"),  # budget maxed out
                "is_active": True,
            },
        )
        from gateway.security import generate_api_key
        raw_broke_key, prefix, hashed = generate_api_key()
        await APIKey.objects.acreate(
            tenant=broke_tenant,
            key_prefix=prefix,
            hashed_key=hashed,
            name="Exhausted Key",
            is_active=True,
        )

        res_budget = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_broke_key}"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "This should be blocked"}],
            },
        )
        assert res_budget.status_code == 402, f"Expected 402 Payment Required, got {res_budget.status_code}"
        assert "spending budget exceeded" in res_budget.json()["detail"]
        print("[PASS] FinOps spending budget limit enforced (HTTP 402 Payment Required).")

    # Cleanup connections
    await close_shared_http_client()
    from gateway.utils.redis import close_redis_client
    await close_redis_client()

    print("\nALL PHASE 3 PROVIDER ADAPTER & FINOPS TESTS PASSED 100%!")


if __name__ == "__main__":
    asyncio.run(test_full_phase3_suite())
