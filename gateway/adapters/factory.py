"""
gateway/adapters/factory.py
===========================
Provider Factory & FinOps Cost Enrichment Engine.

Resolves the appropriate ProviderAdapter based on the canonical model_id
or ModelPricing catalog, executes the request, and calculates real-time
token expenses against the tenant's budget.
"""
import logging
from decimal import Decimal
from typing import AsyncGenerator, Dict, Optional

from django.db.models import F

from gateway.adapters.base import BaseProviderAdapter
from gateway.adapters.exceptions import ModelNotSupportedError
from gateway.adapters.gemini import GeminiAdapter
from gateway.adapters.groq import GroqAdapter
from gateway.adapters.mock import MockAdapter
from gateway.adapters.openai import OpenAIAdapter
from gateway.adapters.schemas import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse
from gateway.models import ModelPricing, Tenant

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory and lifecycle orchestrator for provider adapters."""

    _adapters: Dict[str, BaseProviderAdapter] = {
        "groq": GroqAdapter(),
        "openai": OpenAIAdapter(),
        "gemini": GeminiAdapter(),
        "google": GeminiAdapter(),
        "mock": MockAdapter(),
    }

    @classmethod
    def register_adapter(cls, provider_name: str, adapter: BaseProviderAdapter) -> None:
        """Register a custom adapter for testing or extension."""
        cls._adapters[provider_name.lower()] = adapter

    @classmethod
    async def resolve_provider(cls, model_id: str) -> tuple[str, Optional[ModelPricing]]:
        """
        Determines the provider name and retrieves the pricing configuration.
        First checks database catalog; falls back to model prefix conventions.
        """
        # 1. Check ModelPricing catalog in DB asynchronously
        pricing = await ModelPricing.objects.filter(model_id=model_id, is_active=True).afirst()
        if pricing:
            return pricing.provider.lower(), pricing

        # 2. Heuristic resolution from model prefix (e.g. "groq/...", "openai/...", "google/...", "mock/...")
        if "/" in model_id:
            prefix = model_id.split("/")[0].lower()
            if prefix in cls._adapters:
                return prefix, None

        if model_id.startswith("mock-") or model_id == "mock":
            return "mock", None

        raise ModelNotSupportedError(model_id)

    @classmethod
    def get_adapter(cls, provider: str) -> BaseProviderAdapter:
        """Retrieve the adapter instance for a given provider."""
        provider_key = provider.lower()
        adapter = cls._adapters.get(provider_key)
        if not adapter:
            raise ModelNotSupportedError(f"No adapter registered for provider '{provider}'")
        return adapter

    @classmethod
    async def execute_completion(
        cls,
        request: ChatCompletionRequest,
        tenant: Optional[Tenant] = None,
    ) -> ChatCompletionResponse:
        """
        Orchestrates full completion lifecycle:
          1. Resolves adapter and pricing model.
          2. Dispatches non-blocking async LLM request.
          3. Calculates precise FinOps token costs.
          4. Atomically updates tenant's current spend in PostgreSQL.
        """
        provider_name, pricing = await cls.resolve_provider(request.model)
        adapter = cls.get_adapter(provider_name)

        # Execute upstream LLM call asynchronously
        response = await adapter.complete(request)

        # Calculate FinOps token cost if pricing catalog exists
        if pricing:
            cost = pricing.calculate_cost(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
            response.cost_usd = cost

            # Atomically update Tenant spend if tenant is provided
            if tenant and cost > Decimal("0.0000"):
                await Tenant.objects.filter(pk=tenant.pk).aupdate(
                    current_spend_usd=F("current_spend_usd") + cost
                )
                from gateway.ratelimit.budget import FinOpsBudgetManager
                await FinOpsBudgetManager.record_spend_atomic(tenant.id, cost, tenant.monthly_budget_usd)

        return response

    @classmethod
    async def stream_completion(
        cls,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Resolves provider adapter and returns an asynchronous generator of chunks."""
        provider_name, _ = await cls.resolve_provider(request.model)
        adapter = cls.get_adapter(provider_name)
        async for chunk in adapter.stream_complete(request):
            yield chunk

