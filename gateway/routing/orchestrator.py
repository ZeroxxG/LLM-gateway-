"""
gateway/routing/orchestrator.py
===============================
Dynamic Routing & Intelligent Fallback Orchestrator.

Combines the DynamicRouter, Redis Circuit Breakers, and ProviderFactory to:
  1. Resolve candidate models according to routing policy (direct, cost, latency, quality).
  2. Fast-skip providers whose circuits are currently OPEN.
  3. Attempt execution against candidates in sequence.
  4. Automatically fall back upon failure (timeout, 429, 5xx) while updating circuit state.
"""
import copy
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from gateway.adapters.exceptions import (
    ModelNotSupportedError,
    ProviderAPIError,
    ProviderAuthError,
    ProviderException,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from gateway.adapters.factory import ProviderFactory
from gateway.adapters.schemas import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse
from gateway.models import Tenant
from gateway.routing.circuit_breaker import CircuitState, circuit_breaker
from gateway.routing.router import DynamicRouter

logger = logging.getLogger(__name__)


class RoutingOrchestrator:
    """Orchestrates model resolution, circuit breaker checks, and fallback execution."""

    @classmethod
    async def route_and_execute(
        cls,
        request: ChatCompletionRequest,
        tenant: Optional[Tenant] = None,
    ) -> ChatCompletionResponse:
        """
        Executes completion with dynamic routing, circuit breaking, and automatic fallbacks.

        Raises:
            ProviderAPIError: If all candidate models in the fallback chain fail.
        """
        target = getattr(request, "strategy", None) or request.model
        candidate_chain, heuristic_decision = await DynamicRouter.resolve_detailed(
            model_target=target,
            messages=request.messages,
            task_type=getattr(request, "task_type", None),
            latency_sla_ms=getattr(request, "latency_sla_ms", None),
            max_cost_usd=getattr(request, "max_cost_usd", None),
            tenant=tenant,
        )
        primary_model = candidate_chain[0] if candidate_chain else request.model

        attempted_trace: List[Dict[str, Any]] = []

        for index, candidate_model in enumerate(candidate_chain):
            try:
                provider_name, _ = await ProviderFactory.resolve_provider(candidate_model)
            except ModelNotSupportedError:
                attempted_trace.append({
                    "model": candidate_model,
                    "status": "skipped",
                    "reason": "Model unsupported by factory",
                })
                continue

            # 1. Check Circuit Breaker State (Fast-Skip if OPEN)
            circuit_state = await circuit_breaker.get_state(provider_name)
            if circuit_state == CircuitState.OPEN:
                logger.warning(
                    "Skipping '%s' (%s) because circuit breaker is OPEN.",
                    candidate_model,
                    provider_name,
                )
                attempted_trace.append({
                    "model": candidate_model,
                    "provider": provider_name,
                    "status": "fast_skipped",
                    "reason": "Circuit breaker OPEN",
                })
                continue

            # 2. Attempt Execution
            logger.info("Attempting execution on candidate [%d/%d]: '%s' (%s)", index + 1, len(candidate_chain), candidate_model, provider_name)
            candidate_req = copy.deepcopy(request)
            candidate_req.model = candidate_model

            try:
                response = await ProviderFactory.execute_completion(
                    request=candidate_req,
                    tenant=tenant,
                )

                # Record success in circuit breaker
                await circuit_breaker.record_success(provider_name)

                # Log fallback if primary was substituted
                if candidate_model != request.model:
                    logger.warning(
                        "Fallback succeeded! Requested: '%s' -> Routed: '%s' (%s)",
                        request.model,
                        candidate_model,
                        provider_name,
                    )

                if heuristic_decision:
                    response.routing_decision = {
                        "selected_model": heuristic_decision.selected_model,
                        "detected_task_type": heuristic_decision.detected_task_type.value,
                        "estimated_tokens": heuristic_decision.estimated_tokens,
                        "decision_latency_ms": heuristic_decision.decision_latency_ms,
                        "strategy": heuristic_decision.strategy_used,
                        "candidate_chain": heuristic_decision.candidate_chain,
                    }

                return response

            except (ProviderRateLimitError, ProviderTimeoutError, ProviderAPIError, ProviderAuthError) as exc:
                logger.warning(
                    "Candidate '%s' (%s) failed: %s. Recording circuit failure.",
                    candidate_model,
                    provider_name,
                    exc,
                )
                await circuit_breaker.record_failure(provider_name, reason=str(exc))
                attempted_trace.append({
                    "model": candidate_model,
                    "provider": provider_name,
                    "status": "failed",
                    "error": str(exc),
                })
                # Fall back to next candidate in chain
                continue

            except Exception as exc:
                logger.exception("Unexpected error executing candidate '%s': %s", candidate_model, exc)
                await circuit_breaker.record_failure(provider_name, reason=str(exc))
                attempted_trace.append({
                    "model": candidate_model,
                    "provider": provider_name,
                    "status": "failed",
                    "error": str(exc),
                })
                continue

        # All candidates in the fallback chain were exhausted
        error_summary = "; ".join(f"{t['model']} ({t.get('provider', 'unknown')}): {t.get('reason') or t.get('error')}" for t in attempted_trace)
        logger.error("All fallback candidates exhausted for request '%s'. Trace: %s", request.model, attempted_trace)
        raise ProviderAPIError(
            f"All fallback candidates exhausted for '{request.model}'. Attempts: [{error_summary}]",
            provider="gateway",
            status_code=502,
        )

    @classmethod
    async def stream_and_execute(
        cls,
        request: ChatCompletionRequest,
        tenant: Optional[Tenant] = None,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Executes streaming completions with dynamic routing, circuit breaking,
        and fallback execution before the first chunk is emitted.
        """
        target = getattr(request, "strategy", None) or request.model
        candidate_chain, heuristic_decision = await DynamicRouter.resolve_detailed(
            model_target=target,
            messages=request.messages,
            task_type=getattr(request, "task_type", None),
            latency_sla_ms=getattr(request, "latency_sla_ms", None),
            max_cost_usd=getattr(request, "max_cost_usd", None),
            tenant=tenant,
        )

        attempted_trace: List[Dict[str, Any]] = []

        for index, candidate_model in enumerate(candidate_chain):
            try:
                provider_name, pricing = await ProviderFactory.resolve_provider(candidate_model)
            except ModelNotSupportedError:
                attempted_trace.append({
                    "model": candidate_model,
                    "status": "skipped",
                    "reason": "Model unsupported by factory",
                })
                continue

            circuit_state = await circuit_breaker.get_state(provider_name)
            if circuit_state == CircuitState.OPEN:
                attempted_trace.append({
                    "model": candidate_model,
                    "provider": provider_name,
                    "status": "fast_skipped",
                    "reason": "Circuit breaker OPEN",
                })
                continue

            candidate_req = copy.deepcopy(request)
            candidate_req.model = candidate_model

            generator = ProviderFactory.stream_completion(candidate_req)
            first_chunk = None

            try:
                first_chunk = await generator.__anext__()
            except StopAsyncIteration:
                pass
            except (ProviderRateLimitError, ProviderTimeoutError, ProviderAPIError, ProviderAuthError) as exc:
                logger.warning("Streaming candidate '%s' failed before first chunk: %s", candidate_model, exc)
                await circuit_breaker.record_failure(provider_name, reason=str(exc))
                attempted_trace.append({
                    "model": candidate_model,
                    "provider": provider_name,
                    "status": "failed",
                    "error": str(exc),
                })
                continue
            except Exception as exc:
                logger.exception("Unexpected error streaming from candidate '%s': %s", candidate_model, exc)
                await circuit_breaker.record_failure(provider_name, reason=str(exc))
                attempted_trace.append({
                    "model": candidate_model,
                    "provider": provider_name,
                    "status": "failed",
                    "error": str(exc),
                })
                continue

            # First chunk was successfully fetched
            await circuit_breaker.record_success(provider_name)
            accumulated_chunks = 0
            if first_chunk:
                yield first_chunk
                accumulated_chunks += 1

            try:
                async for chunk in generator:
                    accumulated_chunks += 1
                    yield chunk
            except Exception as exc:
                logger.warning("Mid-stream failure on '%s': %s", candidate_model, exc)
                await circuit_breaker.record_failure(provider_name, reason=str(exc))
                return

            # FinOps accounting upon complete stream delivery
            if pricing and tenant:
                from decimal import Decimal
                from django.db.models import F
                from gateway.models import Tenant as TenantModel
                from gateway.ratelimit.budget import FinOpsBudgetManager

                prompt_text = " ".join(msg.content for msg in request.messages)
                prompt_tokens = max(1, len(prompt_text.split()) * 2)
                completion_tokens = max(1, accumulated_chunks * 2)
                cost = pricing.calculate_cost(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

                if cost > Decimal("0.0000"):
                    await TenantModel.objects.filter(pk=tenant.pk).aupdate(
                        current_spend_usd=F("current_spend_usd") + cost
                    )
                    await FinOpsBudgetManager.record_spend_atomic(tenant.id, cost, tenant.monthly_budget_usd)

            return

        error_summary = "; ".join(f"{t['model']}: {t.get('error') or t.get('reason')}" for t in attempted_trace)
        raise ProviderAPIError(
            f"All fallback candidates exhausted for streaming '{request.model}'. Attempts: [{error_summary}]",
            provider="gateway",
            status_code=502,
        )

