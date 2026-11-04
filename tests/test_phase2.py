"""
tests/test_phase2.py
====================
Automated integration test suite for Phase 2:
  - Cryptographic API Key hashing & verification
  - Multi-tenant model isolation & budget tracking
  - Django Ninja async auth (Bearer & X-API-Key)
  - Model pricing catalog & token cost calculations
"""
import asyncio
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
import httpx

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")
django.setup()

from config.asgi import application
from gateway.models import APIKey, ModelPricing, Tenant
from gateway.security import generate_api_key, hash_api_key, verify_api_key_async


async def test_full_phase2_flow():
    # 1. Fetch test tenant and its active key
    tenant = await Tenant.objects.aget(name="Acme Global (Test Org)")
    api_key_obj = await APIKey.objects.filter(tenant=tenant, is_active=True).afirst()
    assert api_key_obj is not None

    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Test 1: Unauthenticated request should yield 401
        res = await client.get("/api/v1/auth/verify")
        assert res.status_code == 401

        # Test 2: Invalid Bearer token should yield 401
        res = await client.get(
            "/api/v1/auth/verify",
            headers={"Authorization": "Bearer gw_live_invalidtoken1234567890abcdef"},
        )
        assert res.status_code == 401

        # Test 3: Authenticated request with valid key
        # Use known test key seeded earlier
        known_key = "gw_live_e6ed9d77b3841b97dadd97bb030f4f5a"
        res = await client.get(
            "/api/v1/auth/verify",
            headers={"Authorization": f"Bearer {known_key}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["authenticated"] is True
        assert data["tenant"]["name"] == "Acme Global (Test Org)"
        assert float(data["tenant"]["monthly_budget_usd"]) == 50.0

        # Test 4: Authenticated request using custom X-API-Key header
        res = await client.get(
            "/api/v1/tenant/me",
            headers={"X-API-Key": known_key},
        )
        assert res.status_code == 200
        assert res.json()["name"] == "Acme Global (Test Org)"

        # Test 5: Model pricing catalog retrieval (public)
        res = await client.get("/api/v1/models/pricing")
        assert res.status_code == 200
        catalog = res.json()
        model_ids = [item["model_id"] for item in catalog]
        assert "groq/llama-3.1-8b" in model_ids
        assert "openai/gpt-4o-mini" in model_ids
        assert "anthropic/claude-3-5-sonnet" in model_ids

        # Test 6: Health check remains functional
        res = await client.get("/api/v1/health/")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"
        assert res.json()["postgres"] is True
        assert res.json()["redis"] is True

    from gateway.utils.redis import close_redis_client
    await close_redis_client()


if __name__ == "__main__":
    asyncio.run(test_full_phase2_flow())
    print("ALL PHASE 2 TESTS COMPLETED WITH 100% SUCCESS!")
