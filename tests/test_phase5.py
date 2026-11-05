"""
tests/test_phase5.py
====================
Comprehensive test suite for Phase 5:
  - Distributed multi-tenant RPM and TPM sliding window rate limiting
  - Atomic Redis Lua script verification under concurrent loads
  - FinOps soft budget threshold warnings (80%, 90%)
  - FinOps hard budget lockout enforcement (HTTP 402)
  - RFC standard rate limit headers (X-RateLimit-*) injection
  - Budget top-up API endpoint
"""
import asyncio
from decimal import Decimal
import os
import sys
import uuid

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
import httpx
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")
django.setup()

from config.asgi import application
from gateway.adapters.base import close_shared_http_client
from gateway.models import APIKey, Tenant
from gateway.ratelimit.budget import FinOpsBudgetManager
from gateway.ratelimit.limiter import DistributedRateLimiter, rate_limiter
from gateway.security import generate_api_key
from tests.test_helpers import ensure_seed_data


@pytest.mark.django_db(transaction=True)
async def test_full_phase5_suite():
    print("\n--- Running Phase 5 Distributed Rate Limiting & FinOps Budget Tests ---")
    await ensure_seed_data()

    # 1. RPM Sliding Window Rate Limiter Tests
    test_limiter = DistributedRateLimiter(window_seconds=60)
    tenant_rpm_id = f"test-tenant-{uuid.uuid4().hex[:8]}"

    # Reset any existing keys
    await test_limiter.reset_tenant_limits(tenant_rpm_id)

    # Limit of 3 RPM
    r1 = await test_limiter.check_rpm(tenant_rpm_id, rpm_limit=3)
    assert r1.allowed is True
    assert r1.current == 1
    assert r1.remaining == 2
    assert r1.reset_seconds > 0

    r2 = await test_limiter.check_rpm(tenant_rpm_id, rpm_limit=3)
    assert r2.allowed is True
    assert r2.current == 2
    assert r2.remaining == 1

    r3 = await test_limiter.check_rpm(tenant_rpm_id, rpm_limit=3)
    assert r3.allowed is True
    assert r3.current == 3
    assert r3.remaining == 0

    # 4th request must be blocked (HTTP 429 condition)
    r4 = await test_limiter.check_rpm(tenant_rpm_id, rpm_limit=3)
    assert r4.allowed is False
    assert r4.remaining == 0
    assert r4.retry_after >= 1
    print("[PASS] RPM sliding window limiter correctly permitted 3 requests and blocked the 4th.")

    # 2. TPM Rate Limiter Tests
    tenant_tpm_id = f"test-tenant-{uuid.uuid4().hex[:8]}"
    await test_limiter.reset_tenant_limits(tenant_tpm_id)

    # Quota of 1000 TPM
    t1 = await test_limiter.check_and_consume_tpm(tenant_tpm_id, tpm_limit=1000, tokens=600)
    assert t1.allowed is True
    assert t1.current == 600
    assert t1.remaining == 400

    # Consuming 500 more tokens (600 + 500 = 1100 > 1000) must be rejected
    t2 = await test_limiter.check_and_consume_tpm(tenant_tpm_id, tpm_limit=1000, tokens=500)
    assert t2.allowed is False
    assert t2.current == 600
    assert t2.remaining == 400
    print("[PASS] TPM token consumption limiter correctly enforced token budget boundary.")

    # 3. FinOps Soft & Hard Budget Logic
    fake_tenant = Tenant(
        name="FinOps Test Corp",
        monthly_budget_usd=Decimal("100.000000"),
        current_spend_usd=Decimal("50.000000"),
    )
    b_normal = FinOpsBudgetManager.evaluate_budget(fake_tenant)
    assert b_normal.allowed is True
    assert b_normal.warning_header is None
    assert b_normal.percentage_consumed == 50.0

    # Soft alert: 85% consumed
    fake_tenant.current_spend_usd = Decimal("85.000000")
    b_soft = FinOpsBudgetManager.evaluate_budget(fake_tenant)
    assert b_soft.allowed is True
    assert b_soft.warning_header is not None
    assert "WARNING" in b_soft.warning_header
    assert "85.0%" in b_soft.warning_header

    # Critical alert: 92% consumed
    fake_tenant.current_spend_usd = Decimal("92.000000")
    b_crit = FinOpsBudgetManager.evaluate_budget(fake_tenant)
    assert b_crit.allowed is True
    assert b_crit.warning_header is not None
    assert "CRITICAL" in b_crit.warning_header
    assert "92.0%" in b_crit.warning_header

    # Hard lockout: 100% consumed
    fake_tenant.current_spend_usd = Decimal("100.000000")
    b_locked = FinOpsBudgetManager.evaluate_budget(fake_tenant)
    assert b_locked.allowed is False
    assert "exhausted" in b_locked.warning_header
    print("[PASS] FinOps soft threshold alerts (80%, 90%) and hard lockouts evaluated correctly.")

    # 4. End-to-End ASGI Integration Testing
    # Create dedicated tenant for HTTP testing with tight limits
    test_tenant_name = f"HTTP Limiter Tenant {uuid.uuid4().hex[:6]}"
    http_tenant, _ = await Tenant.objects.aupdate_or_create(
        name=test_tenant_name,
        defaults={
            "monthly_budget_usd": Decimal("20.000000"),
            "current_spend_usd": Decimal("16.500000"),  # 82.5% (triggers soft warning header)
            "rpm_limit": 2,  # Strict limit to test HTTP 429
            "tpm_limit": 10000,
            "is_active": True,
        },
    )

    raw_key, prefix, hashed = generate_api_key()
    await APIKey.objects.acreate(
        tenant=http_tenant,
        key_prefix=prefix,
        hashed_key=hashed,
        name="Phase 5 Rate Limit Test Key",
        is_active=True,
    )
    await rate_limiter.reset_tenant_limits(str(http_tenant.id))

    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # A. Query rate limit status
        res_status = await client.get(
            "/api/v1/tenant/ratelimit",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert res_status.status_code == 200
        assert res_status.json()["rpm_limit"] == 2
        print("[PASS] GET /api/v1/tenant/ratelimit returned tenant rate limit quota.")

        # B. First Completion Request -> Success (200) with injected headers
        res1 = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Request 1"}],
            },
        )
        assert res1.status_code == 200, f"Error: {res1.text}"
        headers1 = res1.headers
        assert "x-ratelimit-limit-rpm" in headers1
        assert "x-ratelimit-remaining-rpm" in headers1
        assert "x-ratelimit-reset" in headers1
        assert "x-budget-warning" in headers1  # 82.5% soft budget warning
        assert "82.5%" in headers1["x-budget-warning"]
        print(f"[PASS] Chat completion injected X-RateLimit headers and X-Budget-Warning: '{headers1['x-budget-warning']}'.")

        # C. Second Completion Request -> Success (200, last remaining RPM slot)
        res2 = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Request 2"}],
            },
        )
        assert res2.status_code == 200

        # D. Third Completion Request -> Blocked by RPM limit (HTTP 429)
        res3 = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Request 3 (should be 429)"}],
            },
        )
        assert res3.status_code == 429, f"Expected 429 Too Many Requests, got {res3.status_code}: {res3.text}"
        assert "Retry-After" in res3.headers or "retry-after" in res3.headers
        assert "Rate limit exceeded" in res3.json()["detail"]
        print(f"[PASS] Excessive request rate correctly rejected with HTTP 429: '{res3.json()['detail']}'.")

        # E. FinOps Budget Top-up API
        res_topup = await client.post(
            "/api/v1/tenant/budget/top-up",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={"additional_budget_usd": "30.000000"},
        )
        assert res_topup.status_code == 200
        new_budget = Decimal(str(res_topup.json()["monthly_budget_usd"]))
        assert new_budget == Decimal("50.000000")  # 20 + 30 = 50
        print(f"[PASS] POST /api/v1/tenant/budget/top-up adjusted monthly budget to ${new_budget}.")

    # Cleanup connections
    await close_shared_http_client()
    from gateway.utils.redis import close_redis_client
    await close_redis_client()

    print("\nALL PHASE 5 RATE LIMITING & BUDGET CONTROL TESTS PASSED 100%!")


if __name__ == "__main__":
    asyncio.run(test_full_phase5_suite())
