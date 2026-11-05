"""
gateway/routing/router.py
=========================
Dynamic Model Routing & Fallback Candidate Selection Engine.

Translates explicit model IDs or high-level routing strategies
(e.g., 'cost_optimized', 'fast', 'premium') into prioritized candidate chains.
"""
from decimal import Decimal
from enum import Enum
import logging
from typing import Any, List, Optional, Tuple

from gateway.models import ModelPricing, ModelTier, Tenant

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Supported dynamic routing optimization policies."""

    DIRECT = "direct"
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    HEURISTIC = "heuristic"
    AUTO = "auto"


# Static fallback mappings for explicit model requests
DEFAULT_FALLBACK_CHAINS: dict[str, List[str]] = {
    "groq/llama-3.1-8b": ["google/gemini-1.5-flash", "openai/gpt-4o-mini", "mock/test-model"],
    "openai/gpt-4o-mini": ["google/gemini-1.5-flash", "groq/llama-3.1-8b", "mock/test-model"],
    "google/gemini-1.5-flash": ["groq/llama-3.1-8b", "openai/gpt-4o-mini", "mock/test-model"],
    "anthropic/claude-3-5-sonnet": ["openai/gpt-4o", "google/gemini-1.5-pro", "mock/test-model"],
    "mock/test-model": ["mock/test-model"],
}


class DynamicRouter:
    """Resolves inbound target model into an ordered list of candidate models."""

    @classmethod
    def detect_strategy(cls, model_target: str) -> RoutingStrategy:
        """Detect whether the target is an explicit model or a routing strategy keyword."""
        target_lower = model_target.lower().strip()
        if target_lower in ("cost_optimized", "cheapest", "routing:cost", "cost"):
            return RoutingStrategy.COST_OPTIMIZED
        if target_lower in ("latency_optimized", "fast", "routing:fast", "speed"):
            return RoutingStrategy.LATENCY_OPTIMIZED
        if target_lower in ("quality_optimized", "premium", "routing:quality"):
            return RoutingStrategy.QUALITY_OPTIMIZED
        if target_lower in ("heuristic", "smart", "intelligent", "routing:heuristic", "routing:smart"):
            return RoutingStrategy.HEURISTIC
        if target_lower in ("auto", "default"):
            return RoutingStrategy.AUTO
        return RoutingStrategy.DIRECT

    @classmethod
    async def resolve_candidate_chain(
        cls,
        model_target: str,
        messages: Optional[List[Any]] = None,
        task_type: Optional[str] = None,
        latency_sla_ms: Optional[float] = None,
        max_cost_usd: Optional[Decimal] = None,
        tenant: Optional[Tenant] = None,
    ) -> List[str]:
        """
        Returns an ordered list of candidate model identifiers to attempt in sequence.

        Priority Ordering:
          Candidate 0: Primary target
          Candidate 1..N: Decreasing preference fallbacks
        """
        chain, _ = await cls.resolve_detailed(
            model_target=model_target,
            messages=messages,
            task_type=task_type,
            latency_sla_ms=latency_sla_ms,
            max_cost_usd=max_cost_usd,
            tenant=tenant,
        )
        return chain

    @classmethod
    async def resolve_detailed(
        cls,
        model_target: str,
        messages: Optional[List[Any]] = None,
        task_type: Optional[str] = None,
        latency_sla_ms: Optional[float] = None,
        max_cost_usd: Optional[Decimal] = None,
        tenant: Optional[Tenant] = None,
    ) -> Tuple[List[str], Optional[Any]]:
        """
        Resolves candidate chain and returns detailed heuristic decision metadata if applicable.
        """
        strategy = cls.detect_strategy(model_target)

        # 1. Heuristic & Intelligent Routing: Sub-2ms decision engine
        if strategy in (RoutingStrategy.HEURISTIC, RoutingStrategy.AUTO) and messages:
            from gateway.routing.heuristics import HeuristicRouter
            heuristic_res = await HeuristicRouter.evaluate_routing(
                messages=messages,
                task_type_override=task_type,
                latency_sla_ms=latency_sla_ms,
                max_cost_usd=max_cost_usd,
                tenant=tenant,
                requested_model=model_target,
            )
            return heuristic_res.candidate_chain, heuristic_res

        # 2. Strategy-based resolution: Query ModelPricing catalog
        if strategy in (RoutingStrategy.COST_OPTIMIZED, RoutingStrategy.AUTO):
            candidates = []
            async for pricing in ModelPricing.objects.filter(is_active=True).order_by("input_cost_per_million", "output_cost_per_million"):
                candidates.append(pricing.model_id)
            if "mock/test-model" not in candidates:
                candidates.append("mock/test-model")
            return candidates or ["groq/llama-3.1-8b", "google/gemini-1.5-flash", "mock/test-model"], None

        if strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            # Latency preference: Groq LPU first, then Gemini Flash, then others
            return [
                "groq/llama-3.1-8b",
                "google/gemini-1.5-flash",
                "openai/gpt-4o-mini",
                "mock/test-model",
            ], None

        if strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            # High-capability tier preference
            candidates = []
            async for pricing in ModelPricing.objects.filter(is_active=True, tier=ModelTier.PREMIUM):
                candidates.append(pricing.model_id)
            if not candidates:
                candidates = ["anthropic/claude-3-5-sonnet", "openai/gpt-4o-mini", "mock/test-model"]
            return candidates, None

        # 3. Direct model request: Return target + pre-configured fallback chain
        if model_target in DEFAULT_FALLBACK_CHAINS:
            chain = [model_target] + [m for m in DEFAULT_FALLBACK_CHAINS[model_target] if m != model_target]
            return chain, None

        # Default fallback chain if unknown explicit model
        return [model_target, "mock/test-model"], None
