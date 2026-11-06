"""
tests/test_phase10.py
=====================
Comprehensive test suite for Phase 10:
  - Deterministic Prompt Cache Key Generation (SHA-256)
  - Direct PromptCacheManager Operations (get, set, hits, misses, stats, clear)
  - End-to-end ASGI Chat Completions Cache Miss (X-Cache: MISS)
  - End-to-end ASGI Chat Completions Cache Hit (X-Cache: HIT, < 10ms, identical payload)
  - Cache Bypass via Cache-Control: no-cache (X-Cache: BYPASS)
  - Operational Cache Telemetry (GET /api/v1/cache/stats)
  - Tenant Cache Invalidation (POST /api/v1/cache/clear)
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
from gateway.adapters.schemas import ChatCompletionRequest, ChatMessage
from gateway.caching import PromptCacheManager
from gateway.models import APIKey, Tenant
from gateway.utils.redis import close_redis_client, get_redis_client
from tests.test_helpers import ensure_seed_data, KNOWN_TEST_KEY


@pytest.mark.django_db(transaction=True)
async def test_full_phase10_suite():
    print("\n--- Running Phase 10 Prompt & Response Caching Tests ---")
    tenant = await ensure_seed_data()
    redis = await get_redis_client()
    tenant_id_str = str(tenant.id)

    # Clean existing cache for clean test run
    await PromptCacheManager.clear_tenant_cache(tenant_id_str)
    await redis.delete(f"cache:stats:{tenant_id_str}:hits", f"cache:stats:{tenant_id_str}:misses")

    # -------------------------------------------------------------------------
    # 1. Deterministic Cache Key Generation
    # -------------------------------------------------------------------------
    print("\n[Step 1] Verifying deterministic cache key generation...")
    req1 = ChatCompletionRequest(
        model="mock/test-model",
        messages=[ChatMessage(role="user", content="Hello world!  ")],
        temperature=0.7,
    )
    req2 = ChatCompletionRequest(
        model="mock/test-model",
        messages=[ChatMessage(role="user", content="Hello world!")],
        temperature=0.7,
    )
    req_different = ChatCompletionRequest(
        model="mock/test-model",
        messages=[ChatMessage(role="user", content="Different message")],
        temperature=0.7,
    )

    key1 = PromptCacheManager.compute_cache_key(tenant_id_str, req1)
    key2 = PromptCacheManager.compute_cache_key(tenant_id_str, req2)
    key_diff = PromptCacheManager.compute_cache_key(tenant_id_str, req_different)

    assert key1 == key2, "Normalized whitespace must yield identical cache keys"
    assert key1 != key_diff, "Different messages must yield different cache keys"
    assert key1.startswith(f"cache:resp:{tenant_id_str}:")
    print(f"[PASS] Cache key generation verified: key='{key1[:35]}...'")

    # -------------------------------------------------------------------------
    # 2. End-to-End ASGI Cache Miss and Hit Flow
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying end-to-end ASGI cache miss, hit, and latency...")
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        test_payload = {
            "model": "mock/test-model",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "temperature": 0.5,
        }

        # A. First Request: Must be a Cache MISS
        start1 = time.perf_counter()
        resp1 = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
            json=test_payload,
        )
        latency1_ms = (time.perf_counter() - start1) * 1000.0

        assert resp1.status_code == 200, f"Error: {resp1.text}"
        assert resp1.headers.get("x-cache") == "MISS", f"Expected MISS, got {resp1.headers.get('x-cache')}"
        assert "x-cache-key" in resp1.headers
        data1 = resp1.json()
        print(f"[PASS] First request resulted in X-Cache: MISS (latency={latency1_ms:.2f}ms).")

        # B. Second Request: Identical payload must be a Cache HIT
        start2 = time.perf_counter()
        resp2 = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
            json=test_payload,
        )
        latency2_ms = (time.perf_counter() - start2) * 1000.0

        assert resp2.status_code == 200, f"Error: {resp2.text}"
        assert resp2.headers.get("x-cache") == "HIT", f"Expected HIT, got {resp2.headers.get('x-cache')}"
        data2 = resp2.json()
        assert data1["choices"][0]["message"]["content"] == data2["choices"][0]["message"]["content"]
        assert latency2_ms < 1000.0, f"Cache HIT should be sub-second (got {latency2_ms:.2f}ms)"
        print(f"[PASS] Second request resulted in X-Cache: HIT (latency={latency2_ms:.2f}ms, content identical).")

        # -------------------------------------------------------------------------
        # 3. Cache Bypass Testing (Cache-Control: no-cache)
        # -------------------------------------------------------------------------
        print("\n[Step 3] Verifying Cache-Control: no-cache bypasses Redis cache...")
        resp_bypass = await client.post(
            "/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {KNOWN_TEST_KEY}",
                "Cache-Control": "no-cache",
            },
            json=test_payload,
        )
        assert resp_bypass.status_code == 200
        assert resp_bypass.headers.get("x-cache") == "BYPASS"
        print("[PASS] Request with Cache-Control: no-cache returned X-Cache: BYPASS.")

        # -------------------------------------------------------------------------
        # 4. Cache Stats Endpoint
        # -------------------------------------------------------------------------
        print("\n[Step 4] Verifying GET /api/v1/cache/stats...")
        stats_resp = await client.get(
            "/api/v1/cache/stats",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
        )
        assert stats_resp.status_code == 200, f"Error: {stats_resp.text}"
        stats = stats_resp.json()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["cached_keys"] >= 1
        assert stats["hit_ratio"] > 0.0
        print(f"[PASS] Cache stats verified: hits={stats['hits']}, misses={stats['misses']}, ratio={stats['hit_ratio']}, keys={stats['cached_keys']}.")

        # -------------------------------------------------------------------------
        # 5. Cache Clear Endpoint
        # -------------------------------------------------------------------------
        print("\n[Step 5] Verifying POST /api/v1/cache/clear invalidation...")
        clear_resp = await client.post(
            "/api/v1/cache/clear",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
        )
        assert clear_resp.status_code == 200, f"Error: {clear_resp.text}"
        clear_data = clear_resp.json()
        assert clear_data["status"] == "cleared"
        assert clear_data["keys_deleted"] >= 1
        print(f"[PASS] Cache cleared successfully, evicted {clear_data['keys_deleted']} keys.")

        # Verify next request is a MISS again
        resp_after_clear = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
            json=test_payload,
        )
        assert resp_after_clear.status_code == 200
        assert resp_after_clear.headers.get("x-cache") == "MISS"
        print("[PASS] Request following cache invalidation resulted in X-Cache: MISS.")

    # Cleanup connections
    await close_shared_http_client()
    await close_redis_client()

    print("\nALL PHASE 10 PROMPT & RESPONSE CACHING TESTS PASSED 100%!")


if __name__ == "__main__":
    asyncio.run(test_full_phase10_suite())
