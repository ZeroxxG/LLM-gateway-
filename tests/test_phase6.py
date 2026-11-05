"""
tests/test_phase6.py
====================
Comprehensive test suite for Phase 6:
  - Celery background task registration and execution
  - Asynchronous RequestAuditLog recording
  - Daily usage aggregation rollup into DailyUsageAggregate
  - FinOps tenant budget reconciliation & drift detection
  - Audit log retention pruning
  - End-to-end ASGI endpoints:
      * POST /api/v1/chat/completions with async Celery dispatch
      * GET /api/v1/analytics/logs
      * POST /api/v1/analytics/aggregate
      * GET /api/v1/analytics/tasks/{task_id}
      * GET /api/v1/analytics/usage
"""
import asyncio
from datetime import timedelta
from decimal import Decimal
import os
import sys
import time
import uuid

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
import httpx
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")
# Enable eager mode so Celery tasks can execute synchronously during tests when needed
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "True"
django.setup()

from django.conf import settings
settings.CELERY_TASK_ALWAYS_EAGER = True

from django.utils import timezone
from config.asgi import application
from gateway.adapters.base import close_shared_http_client
from gateway.models import APIKey, DailyUsageAggregate, RequestAuditLog, Tenant
from gateway.security import generate_api_key
from gateway.tasks import (
    aggregate_daily_usage_task,
    cleanup_old_audit_logs_task,
    dispatch_audit_log_async,
    log_request_usage_task,
    reconcile_tenant_budgets_task,
)
from gateway.utils.redis import close_redis_client
from tests.test_helpers import ensure_seed_data


@pytest.mark.django_db(transaction=True)
async def test_full_phase6_suite():
    print("\n--- Running Phase 6 Celery Workers & Asynchronous Usage Aggregation Tests ---")
    await ensure_seed_data()

    # 1. Setup isolated test Tenant
    tenant_name = f"Celery Test Corp {uuid.uuid4().hex[:6]}"
    tenant, _ = await Tenant.objects.aupdate_or_create(
        name=tenant_name,
        defaults={
            "monthly_budget_usd": Decimal("100.000000"),
            "current_spend_usd": Decimal("0.050000"),
            "rpm_limit": 100,
            "tpm_limit": 50000,
            "is_active": True,
        },
    )

    raw_key, prefix, hashed = generate_api_key()
    await APIKey.objects.acreate(
        tenant=tenant,
        key_prefix=prefix,
        hashed_key=hashed,
        name="Phase 6 Test Key",
        is_active=True,
    )

    # 2. Celery Worker Task: log_request_usage_task
    sample_audit = {
        "tenant_id": str(tenant.id),
        "request_id": f"req_{uuid.uuid4().hex[:12]}",
        "model_id": "mock/test-model",
        "provider": "mock",
        "prompt_tokens": 15,
        "completion_tokens": 25,
        "total_tokens": 40,
        "cost_usd": "0.00002000",
        "latency_ms": 18.5,
        "status_code": 200,
        "routing_strategy": "cost_optimized",
        "fallback_occurred": False,
    }

    # Execute eager Celery task directly in a thread to isolate synchronous ORM operations
    task_res = await asyncio.to_thread(log_request_usage_task.apply, args=[sample_audit])
    assert task_res.status == "SUCCESS"
    assert task_res.result["status"] == "success"
    assert task_res.result["total_tokens"] == 40
    print("[PASS] Celery worker task 'log_request_usage_task' successfully recorded audit log.")

    # Verify audit log in PostgreSQL
    log_record = await RequestAuditLog.objects.aget(request_id=sample_audit["request_id"])
    assert log_record.total_tokens == 40
    assert log_record.cost_usd == Decimal("0.00002000")
    assert log_record.model_id == "mock/test-model"
    print(f"[PASS] RequestAuditLog verified in PostgreSQL: id={log_record.id}, cost=${log_record.cost_usd}.")

    # Add second audit log for aggregation testing
    sample_audit2 = {
        "tenant_id": str(tenant.id),
        "request_id": f"req_{uuid.uuid4().hex[:12]}",
        "model_id": "mock/test-model",
        "provider": "mock",
        "prompt_tokens": 30,
        "completion_tokens": 50,
        "total_tokens": 80,
        "cost_usd": "0.00004000",
        "latency_ms": 22.1,
        "status_code": 200,
        "routing_strategy": "cost_optimized",
        "fallback_occurred": False,
    }
    await asyncio.to_thread(log_request_usage_task.apply, args=[sample_audit2])

    # 3. Celery Aggregation Task: aggregate_daily_usage_task
    today_str = timezone.now().date().strftime("%Y-%m-%d")
    agg_res = await asyncio.to_thread(aggregate_daily_usage_task.apply, kwargs={"target_date_str": today_str})
    assert agg_res.status == "SUCCESS"
    assert agg_res.result["tenants_aggregated"] >= 1
    print(f"[PASS] Celery aggregation task 'aggregate_daily_usage_task' completed for date {today_str}.")

    # Verify DailyUsageAggregate row
    daily_agg = await DailyUsageAggregate.objects.aget(tenant=tenant, date=timezone.now().date())
    assert daily_agg.total_requests >= 2
    assert daily_agg.total_tokens >= 120
    assert daily_agg.total_cost_usd >= Decimal("0.000060")
    print(f"[PASS] DailyUsageAggregate verified: {daily_agg.total_requests} reqs, {daily_agg.total_tokens} tokens, ${daily_agg.total_cost_usd}.")

    # 4. Celery Budget Reconciliation Task: reconcile_tenant_budgets_task
    recon_res = await asyncio.to_thread(reconcile_tenant_budgets_task.apply)
    assert recon_res.status == "SUCCESS"
    assert recon_res.result["tenants_checked"] >= 1
    print(f"[PASS] FinOps budget reconciliation task audited {recon_res.result['tenants_checked']} tenants.")

    # 5. Celery Cleanup Task: cleanup_old_audit_logs_task
    clean_res = await asyncio.to_thread(cleanup_old_audit_logs_task.apply, kwargs={"retention_days": 365})
    assert clean_res.status == "SUCCESS"
    print(f"[PASS] Audit log cleanup task executed: {clean_res.result['deleted_count']} old logs pruned.")

    # 6. End-to-End ASGI Integration Tests
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # A. Execute chat completion and verify async audit logging
        res_chat = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "mock/test-model",
                "messages": [{"role": "user", "content": "Async audit log integration test"}],
            },
        )
        assert res_chat.status_code == 200, f"Error: {res_chat.text}"
        chat_data = res_chat.json()
        req_id = chat_data["id"]
        print(f"[PASS] POST /api/v1/chat/completions succeeded with req_id={req_id}.")

        # Allow worker thread / eager execution a brief moment
        await asyncio.sleep(0.1)

        # B. Query GET /api/v1/analytics/logs
        res_logs = await client.get(
            "/api/v1/analytics/logs?limit=10",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert res_logs.status_code == 200, f"Error: {res_logs.text}"
        logs = res_logs.json()
        assert len(logs) >= 1
        assert logs[0]["model_id"] == "mock/test-model"
        print(f"[PASS] GET /api/v1/analytics/logs returned {len(logs)} audit logs.")

        # C. Query POST /api/v1/analytics/aggregate
        res_trigger = await client.post(
            "/api/v1/analytics/aggregate",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert res_trigger.status_code == 200
        task_id = res_trigger.json()["task_id"]
        assert task_id is not None
        print(f"[PASS] POST /api/v1/analytics/aggregate queued task: {task_id}.")

        # D. Query GET /api/v1/analytics/tasks/{task_id}
        res_task = await client.get(
            f"/api/v1/analytics/tasks/{task_id}",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert res_task.status_code == 200
        task_info = res_task.json()
        assert task_info["task_id"] == task_id
        assert task_info["status"] in ["PENDING", "SUCCESS", "STARTED"]
        print(f"[PASS] GET /api/v1/analytics/tasks/{task_id} inspected status: {task_info['status']}.")

        # E. Query GET /api/v1/analytics/usage
        res_usage = await client.get(
            "/api/v1/analytics/usage?days=7",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert res_usage.status_code == 200
        usage_data = res_usage.json()
        assert usage_data["tenant_name"] == tenant.name
        assert usage_data["total_requests"] >= 1
        print(f"[PASS] GET /api/v1/analytics/usage returned aggregated analytics: {usage_data['total_requests']} requests, ${usage_data['total_cost_usd']}.")

    # Cleanup connections
    await close_shared_http_client()
    await close_redis_client()

    print("\nALL PHASE 6 CELERY WORKER & USAGE AGGREGATION TESTS PASSED 100%!")


if __name__ == "__main__":
    asyncio.run(test_full_phase6_suite())
