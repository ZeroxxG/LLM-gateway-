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
from typing import Any, Dict, List, Optional

from gateway.adapters.exceptions import (
    ModelNotSupportedError,
    ProviderAPIError,
    ProviderAuthError,
    ProviderException,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from gateway.adapters.factory import ProviderFactory
from gateway.adapters.schemas import ChatCompletionRequest, ChatCompletionResponse
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
        candidate_chain = await DynamicRouter.resolve_candidate_chain(request.model)
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
