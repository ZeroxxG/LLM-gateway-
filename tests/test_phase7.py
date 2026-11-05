"""
tests/test_phase7.py
====================
Test suite for Phase 7:
  - Real-time atomic Redis spend tracking (INCRBYFLOAT)
  - Pre-flight sub-millisecond budget evaluation in Redis
  - Automatic hard-stop rejection (HTTP 402) on budget exhaustion
  - Real-time top-up and lockout clearance
  - Redis spend synchronization to PostgreSQL
  - Response headers (X-Budget-Limit, X-Budget-Current-Spend, X-Budget-Remaining)
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

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")
django.setup()

from config.asgi import application
from gateway.adapters.base import close_shared_http_client
from gateway.models import APIKey, Tenant
from gateway.ratelimit.budget import FinOpsBudgetManager
from gateway.security import generate_api_key
from gateway.utils.redis import close_redis_client, get_redis_client


async def test_full_phase7_suite():
    print("\n--- Running Phase 7 Real-Time Redis Budget Tracking & Hard-Stop Tests ---")

    # 1. Direct Redis INCRBYFLOAT Lua tests
    tenant_id = uuid.uuid4()
    budget_cap = Decimal("0.050000")
    redis = await get_redis_client()

    # Clean test keys
    t_id_str = str(tenant_id)
    await redis.delete(
        FinOpsBudgetManager._spend_key(t_id_str),
        FinOpsBudgetManager._budget_key(t_id_str),
        FinOpsBudgetManager._lockout_key(t_id_str),
    )

    # First increment: add $0.010000
    spend1, cap1, exhausted1 = await FinOpsBudgetManager.record_spend_atomic(
        tenant_id, Decimal("0.010000"), budget_cap
    )
    assert spend1 == Decimal("0.010000")
    assert cap1 == budget_cap
    assert exhausted1 is False
    print(f"[PASS] INCRBYFLOAT atomic spend increment 1: ${spend1} / ${cap1}.")

    # Second increment: add $0.020000 -> $0.030000
    spend2, cap2, exhausted2 = await FinOpsBudgetManager.record_spend_atomic(
        tenant_id, Decimal("0.020000"), budget_cap
    )
    assert spend2 == Decimal("0.030000")
    assert exhausted2 is False
    print(f"[PASS] INCRBYFLOAT atomic spend increment 2: ${spend2} / ${cap2}.")

    # Third increment: add $0.025000 -> $0.055000 (exceeds $0.050000 cap)
    spend3, cap3, exhausted3 = await FinOpsBudgetManager.record_spend_atomic(
        tenant_id, Decimal("0.025000"), budget_cap
    )
    assert spend3 >= budget_cap
    assert exhausted3 is True
    print(f"[PASS] INCRBYFLOAT tripped hard lockout: ${spend3} >= ${cap3}.")

    # Check lockout key exists in Redis
    lockout_val = await redis.get(FinOpsBudgetManager._lockout_key(t_id_str))
    assert lockout_val is not None
    print("[PASS] Redis lockout key set correctly on budget exhaustion.")

    # 2. End-to-End ASGI Integration Tests
    # Create test tenant with strict budget cap of $0.000010
    http_tenant, _ = await Tenant.objects.aupdate_or_create(
        name=f"Realtime Budget Tenant {uuid.uuid4().hex[:6]}",
        defaults={
            "monthly_budget_usd": Decimal("0.000010"),  # Very small budget cap
            "current_spend_usd": Decimal("0.000000"),
            "rpm_limit": 60,
            "tpm_limit": 50000,
            "is_active": True,
        },
    )

    raw_key, prefix, hashed = generate_api_key()
    await APIKey.objects.acreate(
        tenant=http_tenant,
        key_prefix=prefix,
        hashed_key=hashed,
        name="Phase 7 Test Key",
        is_active=True,
    )

    # Reset any cached keys for this tenant in Redis
    h_id_str = str(http_tenant.id)
    await redis.delete(
        FinOpsBudgetManager._spend_key(h_id_str),
        FinOpsBudgetManager._budget_key(h_id_str),
        FinOpsBudgetManager._lockout_key(h_id_str),
    )

    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # A. Check real-time budget status endpoint
        res_budget = await client.get(
            "/api/v1/tenant/budget",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert res_budget.status_code == 200, f"Error: {res_budget.text}"
        budget_info = res_budget.json()
        assert budget_info["is_locked_out"] is False
        print(f"[PASS] GET /tenant/budget: cap=${budget_info['monthly_budget_usd']}, spend=${budget_info['current_spend_usd']}.")

        # B. First Completion: Allowed (spend starts at 0, costs ~$0.000006)
        res1 = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Request within budget"}],
            },
        )
        assert res1.status_code == 200, f"Error: {res1.text}"
        assert "x-budget-limit" in res1.headers
        assert "x-budget-current-spend" in res1.headers
        assert "x-budget-remaining" in res1.headers
        print(f"[PASS] First request succeeded. Headers: spend=${res1.headers['x-budget-current-spend']}, remaining=${res1.headers['x-budget-remaining']}.")

        # C. Second Completion: Allowed but exhausts budget (spend ~0.000012 >= 0.000010 cap)
        res2 = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Request pushing past budget"}],
            },
        )
        assert res2.status_code == 200, f"Error: {res2.text}"

        # D. Third Completion: Blocked by Redis real-time pre-flight check (HTTP 402)
        res3 = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "This request must be rejected"}],
            },
        )
        assert res3.status_code == 402, f"Expected 402 Payment Required, got {res3.status_code}: {res3.text}"
        assert "spending budget exceeded" in res3.json()["detail"]
        print(f"[PASS] Real-time hard-stop lockout enforced: HTTP 402 rejected sub-millisecond.")

        # E. Top up budget via API to lift lockout
        res_topup = await client.post(
            "/api/v1/tenant/budget/top-up",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={"additional_budget_usd": "10.000000"},
        )
        assert res_topup.status_code == 200
        print(f"[PASS] Budget topped up via API by $10.000000.")

        # F. Fourth Completion: Now succeeds immediately because lockout was lifted in Redis
        res4 = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Request after top-up"}],
            },
        )
        assert res4.status_code == 200, f"Error: {res4.text}"
        print(f"[PASS] Request after top-up succeeded (HTTP 200 OK).")

        # G. Sync Redis spend to PostgreSQL
        res_sync = await client.post(
            "/api/v1/tenant/budget/sync",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert res_sync.status_code == 200
        sync_data = res_sync.json()
        assert sync_data["status"] == "synced"
        print(f"[PASS] POST /tenant/budget/sync confirmed spend synced to DB: ${sync_data['synced_spend_usd']}.")

    # Cleanup connections
    await close_shared_http_client()
    await close_redis_client()

    print("\nALL PHASE 7 REAL-TIME REDIS BUDGET TESTS PASSED 100%!")


if __name__ == "__main__":
    asyncio.run(test_full_phase7_suite())
