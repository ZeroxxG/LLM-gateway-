"""
tests/test_phase8.py
====================
Comprehensive test suite for Phase 8:
  - Sub-2ms Intelligent Heuristic Decision Engine
  - Fast Task Classification (coding, reasoning, creative, extraction, summarization, chat)
  - Token Context Boundary Pruning
  - Latency SLA Filtering
  - FinOps Cost & Budget Threshold Constraints
  - Circuit Breaker-Aware Candidate Pruning
  - Diagnostic /routing/evaluate API Endpoint
  - End-to-end ASGI Chat Completions with Heuristic Routing & Diagnostic Headers
"""
import asyncio
from decimal import Decimal
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
from gateway.adapters.schemas import ChatMessage
from gateway.models import APIKey, Tenant
from gateway.routing.circuit_breaker import circuit_breaker
from gateway.routing.heuristics import HeuristicRouter, TaskClassifier, TaskType, TokenEstimator
from gateway.routing.router import DynamicRouter, RoutingStrategy
from gateway.utils.redis import close_redis_client, get_redis_client
from tests.test_helpers import ensure_seed_data, KNOWN_TEST_KEY


@pytest.mark.django_db(transaction=True)
async def test_full_phase8_suite():
    print("\n--- Running Phase 8 Intelligent Heuristic Routing Tests ---")
    await ensure_seed_data()
    # Warm up async Redis pool
    await get_redis_client()
    # Warm up router
    await HeuristicRouter.evaluate_routing(messages=[ChatMessage(role="user", content="warmup")])

    # -------------------------------------------------------------------------
    # 1. Fast Task Classification Tests (< 0.1ms)
    # -------------------------------------------------------------------------
    code_prompt = "```python\ndef calculate_fibonacci(n):\n    return n\n```\nHow to optimize this function?"
    reason_prompt = "Prove that the sum of the first n odd numbers is equal to n squared step-by-step."
    extract_prompt = "Extract customer name and email from this text and output JSON schema."
    summary_prompt = "Please provide a TL;DR and brief summary of the key takeaways from the earnings call."
    creative_prompt = "Write a creative story and poem about an astronaut lost in a quantum rift."
    chat_prompt = "Hello! How are you doing today?"

    assert TaskClassifier.classify_text(code_prompt) == TaskType.CODING
    assert TaskClassifier.classify_text(reason_prompt) == TaskType.REASONING
    assert TaskClassifier.classify_text(extract_prompt) == TaskType.EXTRACTION
    assert TaskClassifier.classify_text(summary_prompt) == TaskType.SUMMARIZATION
    assert TaskClassifier.classify_text(creative_prompt) == TaskType.CREATIVE
    assert TaskClassifier.classify_text(chat_prompt) == TaskType.CHAT
    print("[PASS] TaskClassifier accurately identified all 6 distinct task types.")

    # -------------------------------------------------------------------------
    # 2. Token Estimator Tests
    # -------------------------------------------------------------------------
    short_msgs = [ChatMessage(role="user", content="Hello world!")]
    assert TokenEstimator.estimate_messages(short_msgs) >= 1

    large_text = "word " * 5000  # ~25,000 characters
    large_msgs = [ChatMessage(role="user", content=large_text)]
    est_tokens = TokenEstimator.estimate_messages(large_msgs)
    assert 4000 <= est_tokens <= 7000
    print(f"[PASS] TokenEstimator accurately computed token approximation: {est_tokens} tokens.")

    # -------------------------------------------------------------------------
    # 3. Sub-2ms Decision Engine Performance & Scoring
    # -------------------------------------------------------------------------
    test_msgs = [ChatMessage(role="user", content="Write a python function to refactor SQL queries: SELECT * FROM users")]
    result = await HeuristicRouter.evaluate_routing(messages=test_msgs)

    # Strictly assert sub-2ms decision time
    assert result.decision_latency_ms < 2.0, f"Routing took {result.decision_latency_ms}ms (must be < 2.0ms)"
    assert result.detected_task_type == TaskType.CODING
    assert len(result.candidate_chain) >= 2
    print(f"[PASS] Sub-2ms decision engine executed in {result.decision_latency_ms}ms (< 2.0ms SLA). Detected: '{result.detected_task_type.value}'.")

    # -------------------------------------------------------------------------
    # 4. Token Context Boundary Pruning
    # -------------------------------------------------------------------------
    # Create prompt requiring > 40,000 tokens (mock/test-model has 32,768 context limit)
    huge_text = "token " * 45000  # ~270,000 chars -> ~67,500 tokens
    huge_msgs = [ChatMessage(role="user", content=huge_text)]
    context_res = await HeuristicRouter.evaluate_routing(messages=huge_msgs)

    # Verify mock/test-model was pruned due to context limit
    mock_eval = next((c for c in context_res.candidates if c.model_id == "mock/test-model"), None)
    assert mock_eval is not None
    assert mock_eval.eligible is False
    assert "Context window exceeded" in mock_eval.rejection_reason

    # Verify 128k+ models (Groq, Gemini Flash, GPT-4o-mini) remain eligible
    gemini_eval = next((c for c in context_res.candidates if c.model_id == "google/gemini-1.5-flash"), None)
    assert gemini_eval is not None
    assert gemini_eval.eligible is True
    print("[PASS] Token context boundary filter pruned models with small context windows.")

    # -------------------------------------------------------------------------
    # 5. Latency SLA Constraint Filtering
    # -------------------------------------------------------------------------
    # Strict SLA of 200ms -> Groq LPU (180ms) & Mock (15ms) qualify; slower models pruned
    sla_res = await HeuristicRouter.evaluate_routing(
        messages=test_msgs,
        latency_sla_ms=200.0,
    )
    eligible_models = [c.model_id for c in sla_res.candidates if c.eligible]
    assert "anthropic/claude-3-5-sonnet" not in eligible_models
    assert "openai/gpt-4o-mini" not in eligible_models
    claude_eval = next((c for c in sla_res.candidates if c.model_id == "anthropic/claude-3-5-sonnet"), None)
    assert claude_eval is not None
    assert "violates SLA" in claude_eval.rejection_reason
    print(f"[PASS] Latency SLA constraint (200ms) enforced. Eligible candidates: {eligible_models}.")

    # -------------------------------------------------------------------------
    # 6. Budget Ceiling Constraint Filtering
    # -------------------------------------------------------------------------
    # Query budget constraint: $0.000002 -> expensive models pruned
    budget_res = await HeuristicRouter.evaluate_routing(
        messages=test_msgs,
        max_cost_usd=Decimal("0.00000200"),
    )
    claude_budget_eval = next((c for c in budget_res.candidates if c.model_id == "anthropic/claude-3-5-sonnet"), None)
    assert claude_budget_eval is not None
    assert claude_budget_eval.eligible is False
    assert "exceeds query budget" in claude_budget_eval.rejection_reason
    print("[PASS] Query budget ceiling correctly filtered out models exceeding cost threshold.")

    # -------------------------------------------------------------------------
    # 7. DynamicRouter Strategy Resolution with Heuristics
    # -------------------------------------------------------------------------
    assert DynamicRouter.detect_strategy("heuristic") == RoutingStrategy.HEURISTIC
    assert DynamicRouter.detect_strategy("intelligent") == RoutingStrategy.HEURISTIC
    assert DynamicRouter.detect_strategy("smart") == RoutingStrategy.HEURISTIC

    heuristic_chain, detailed_res = await DynamicRouter.resolve_detailed(
        model_target="heuristic",
        messages=test_msgs,
    )
    assert len(heuristic_chain) >= 2
    assert detailed_res is not None
    assert detailed_res.detected_task_type == TaskType.CODING
    print(f"[PASS] DynamicRouter.resolve_detailed resolved heuristic chain: {heuristic_chain[:3]}.")

    # -------------------------------------------------------------------------
    # 8. End-to-End ASGI API Integration Tests
    # -------------------------------------------------------------------------
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # A. Diagnostic /routing/evaluate endpoint
        eval_resp = await client.post(
            "/api/v1/routing/evaluate",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
            json={
                "messages": [{"role": "user", "content": "Write a python function to parse JSON: def parse_json(s):"}],
                "latency_sla_ms": 1000.0,
            },
        )
        assert eval_resp.status_code == 200, f"Error: {eval_resp.text}"
        eval_data = eval_resp.json()
        assert eval_data["detected_task_type"] == "coding"
        assert eval_data["decision_latency_ms"] < 2.0
        assert len(eval_data["candidate_chain"]) >= 1
        assert len(eval_data["candidates"]) >= 4
        print(f"[PASS] POST /routing/evaluate verified: task={eval_data['detected_task_type']}, latency={eval_data['decision_latency_ms']}ms.")

        # B. Chat Completions with Heuristic Routing Strategy
        chat_resp = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
            json={
                "model": "heuristic",
                "messages": [{"role": "user", "content": "```python\ndef test(): pass\n``` How to unit test?"}],
            },
        )
        assert chat_resp.status_code == 200, f"Error: {chat_resp.text}"
        chat_data = chat_resp.json()
        assert chat_data["object"] == "chat.completion"

        # Verify heuristic routing response headers
        assert "x-routing-strategy" in chat_resp.headers
        assert "x-routing-task-type" in chat_resp.headers
        assert "x-routing-decision-latency-ms" in chat_resp.headers
        assert "x-routing-selected-model" in chat_resp.headers

        task_header = chat_resp.headers["x-routing-task-type"]
        latency_header = float(chat_resp.headers["x-routing-decision-latency-ms"])
        selected_model_header = chat_resp.headers["x-routing-selected-model"]

        assert task_header == "coding"
        assert latency_header < 2.0
        print(f"[PASS] Chat completions injected diagnostic headers: task='{task_header}', decision_latency={latency_header}ms, selected='{selected_model_header}'.")

        # C. Explicit Task Type Override via Completion Payload
        override_resp = await client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {KNOWN_TEST_KEY}"},
            json={
                "model": "heuristic",
                "task_type": "creative",
                "messages": [{"role": "user", "content": "Just a normal greeting"}],
            },
        )
        assert override_resp.status_code == 200
        assert override_resp.headers["x-routing-task-type"] == "creative"
        print("[PASS] Explicit task_type override was honored in routing headers.")

    # Cleanup connections
    await close_shared_http_client()
    await close_redis_client()

    print("\nALL PHASE 8 INTELLIGENT HEURISTIC ROUTING TESTS PASSED 100%!")


if __name__ == "__main__":
    asyncio.run(test_full_phase8_suite())
