"""
tests/test_phase12.py
=====================
Comprehensive test suite for Phase 12: Benchmark Harness & Baseline Eval:
  1. Benchmark Dataset Integrity (40 curated items, category & complexity distribution)
  2. Benchmark Evaluator Scoring (keyword matching, tier constraint enforcement)
  3. Comparative Execution Runner (40 items: All-Premium vs. Routed Gateway)
  4. Spend Cut & FinOps Metrics Verification (> 50% cost savings measured)
  5. Accuracy Retention & Latency Acceleration Verification (>= 90% accuracy maintained)
  6. End-to-End HTTP Endpoint Verification (POST /api/v1/benchmarks/run)
  7. Filtered Benchmark Subset Verification (category filtering & custom sample sizes)
"""
import asyncio
from decimal import Decimal
import os
import sys
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
import httpx

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")
django.setup()

from config.asgi import application
from gateway.adapters.base import close_shared_http_client
from gateway.benchmarks import (
    BENCHMARK_DATASET,
    BenchmarkEvaluator,
    BenchmarkItem,
    BenchmarkRunner,
    get_benchmark_dataset,
)
from gateway.utils.redis import close_redis_client
from tests.test_helpers import KNOWN_TEST_KEY, ensure_seed_data


@pytest.mark.django_db(transaction=True)
async def test_full_phase12_suite():
    print("\n--- Running Phase 12 Benchmark Harness & Baseline Eval Tests ---")
    tenant = await ensure_seed_data()

    # -------------------------------------------------------------------------
    # 1. Benchmark Dataset Integrity
    # -------------------------------------------------------------------------
    print("\n[Step 1] Verifying 40-item benchmark dataset structure...")
    assert len(BENCHMARK_DATASET) == 40, f"Expected 40 items, found {len(BENCHMARK_DATASET)}"

    categories = {item.category for item in BENCHMARK_DATASET}
    assert "chat" in categories
    assert "extraction" in categories
    assert "summarization" in categories
    assert "coding" in categories
    assert "reasoning" in categories

    complexities = {item.complexity for item in BENCHMARK_DATASET}
    assert "low" in complexities
    assert "medium" in complexities
    assert "high" in complexities

    # Verify filtering and sample sizing helper
    subset = get_benchmark_dataset(sample_size=10, categories=["chat", "extraction"])
    assert len(subset) == 10
    assert all(item.category in ["chat", "extraction"] for item in subset)
    print(f"[PASS] Dataset verified: 40 items across categories {categories} and complexities {complexities}.")

    # -------------------------------------------------------------------------
    # 2. Benchmark Evaluator Logic
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying BenchmarkEvaluator accuracy and tier penalties...")
    test_item = BenchmarkItem(
        id="test-item",
        category="chat",
        complexity="low",
        prompt="What is the capital of France?",
        expected_keywords=["paris"],
        required_tier="small",
    )
    # Correct response
    eval_pass = BenchmarkEvaluator.evaluate_response(test_item, "The capital of France is Paris.", "small")
    assert eval_pass.is_accurate is True
    assert eval_pass.score == 1.0

    # Incorrect response
    eval_fail = BenchmarkEvaluator.evaluate_response(test_item, "The capital of France is London.", "small")
    assert eval_fail.is_accurate is False
    assert eval_fail.score == 0.0

    # High complexity tier penalty test
    high_item = BenchmarkItem(
        id="test-high",
        category="coding",
        complexity="high",
        prompt="Write a distributed lock algorithm.",
        expected_keywords=["lock", "distributed"],
        required_tier="premium",
    )
    eval_under = BenchmarkEvaluator.evaluate_response(high_item, "A distributed lock needs a lock.", "small")
    assert eval_under.is_accurate is False
    assert "under-provisioned" in eval_under.reason
    print("[PASS] BenchmarkEvaluator correctly scored accuracy and enforced complexity tiers.")

    # -------------------------------------------------------------------------
    # 3. Execution Runner Across Full 40-Item Benchmark
    # -------------------------------------------------------------------------
    print("\n[Step 3] Executing 40-item comparative benchmark (All-Premium vs. Routed Gateway)...")
    results = await BenchmarkRunner.run_benchmark(
        sample_size=40,
        tenant=tenant,
        mode="simulated",
    )

    assert results["total_items"] == 40
    prem = results["all_premium"]
    routed = results["routed_gateway"]

    # Verify request counts
    assert prem["total_requests"] == 40
    assert routed["total_requests"] == 40

    # -------------------------------------------------------------------------
    # 4. Spend Cut & FinOps Metrics Verification
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying FinOps spend cut and cost savings multiplier...")
    print(f"  All-Premium Total Cost: ${prem['total_cost_usd']}")
    print(f"  Routed Gateway Total Cost: ${routed['total_cost_usd']}")
    print(f"  Spend Cut Percentage: {results['spend_cut_percentage']}%")
    print(f"  Cost Efficiency Multiplier: {results['cost_savings_multiplier']}x")

    assert routed["total_cost_usd"] < prem["total_cost_usd"], "Routed Gateway must cost less than All-Premium"
    assert results["spend_cut_percentage"] >= 50.0, (
        f"Expected at least 50% spend cut, got {results['spend_cut_percentage']}%"
    )
    assert results["cost_savings_multiplier"] >= 2.0, (
        f"Expected at least 2.0x savings, got {results['cost_savings_multiplier']}x"
    )
    print(f"[PASS] Spend cut validated: {results['spend_cut_percentage']}% savings ({results['cost_savings_multiplier']}x).")

    # -------------------------------------------------------------------------
    # 5. Accuracy & Latency Retention Verification
    # -------------------------------------------------------------------------
    print("\n[Step 5] Verifying accuracy retention and latency improvement...")
    print(f"  All-Premium Accuracy: {prem['accuracy_rate']}%")
    print(f"  Routed Gateway Accuracy: {routed['accuracy_rate']}%")
    print(f"  Accuracy Delta: {results['accuracy_delta']}%")
    print(f"  All-Premium Avg Latency: {prem['avg_latency_ms']}ms")
    print(f"  Routed Gateway Avg Latency: {routed['avg_latency_ms']}ms")
    print(f"  Latency Acceleration: {results['latency_reduction_percentage']}%")

    assert prem["accuracy_rate"] >= 90.0
    assert routed["accuracy_rate"] >= 90.0, "Routed Gateway must maintain at least 90% accuracy"
    assert abs(results["accuracy_delta"]) <= 10.0, "Accuracy delta must be within acceptable variance"
    assert results["latency_reduction_percentage"] > 0.0, "Routed Gateway must be faster than All-Premium"

    # Verify model distribution reflects intelligent multi-tier routing
    dist = routed["model_distribution"]
    print(f"  Routed Gateway Model Distribution: {dist}")
    assert len(dist) >= 2, "Routing must dispatch across multiple model tiers"
    print("[PASS] High accuracy maintained while significantly cutting latency.")

    # -------------------------------------------------------------------------
    # 6. End-to-End HTTP Endpoint Verification (POST /api/v1/benchmarks/run)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Verifying POST /api/v1/benchmarks/run endpoint...")
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Full 40 items run via HTTP
        resp_full = await client.post(
            "/api/v1/benchmarks/run",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
            json={"sample_size": 40},
        )
        assert resp_full.status_code == 200, f"Error: {resp_full.text}"
        data_full = resp_full.json()

        assert data_full["total_items"] == 40
        assert "all_premium" in data_full
        assert "routed_gateway" in data_full
        assert data_full["spend_cut_percentage"] >= 50.0
        assert data_full["cost_savings_multiplier"] >= 2.0
        assert len(data_full["sample_items"]) == 40
        assert "summary" in data_full
        print(f"[PASS] Full 40-item API benchmark returned status 200 with summary: {data_full['summary'][:80]}...")

        # Filtered category subset via HTTP
        resp_filtered = await client.post(
            "/api/v1/benchmarks/run",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
            json={
                "sample_size": 10,
                "categories": ["chat", "coding"],
            },
        )
        assert resp_filtered.status_code == 200
        data_filtered = resp_filtered.json()
        assert data_filtered["total_items"] == 10
        assert all(item["category"] in ["chat", "coding"] for item in data_filtered["sample_items"])
        print("[PASS] Category-filtered benchmark API returned status 200.")

    # Cleanup connections
    await close_redis_client()
    await close_shared_http_client()
    print("\n[SUCCESS] All Phase 12 Benchmark Harness tests passed cleanly!")
