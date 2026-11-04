"""
gateway/api.py
==============
Django Ninja root API instance for the LLM Gateway.

Central Router:
  - Auto-generates OpenAPI 3.1 documentation at /api/v1/docs and /openapi.json.
  - Implements pure async def views to prevent thread blocking under ASGI.
  - Secures endpoints via cryptographically hashed API Keys using `APIKeyAuth`.
  - Dispatches chat completions via dynamic routing, circuit breaking, and fallbacks.
"""
import logging
from typing import List, Optional

from asgiref.sync import sync_to_async
from ninja import NinjaAPI
from ninja.errors import HttpError

from gateway.adapters.exceptions import (
    ModelNotSupportedError,
    ProviderAPIError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from gateway.adapters.schemas import ChatCompletionRequest, ChatCompletionResponse
from gateway.health.router import router as health_router
from gateway.models import ModelPricing, Tenant
from gateway.routing.circuit_breaker import circuit_breaker
from gateway.routing.orchestrator import RoutingOrchestrator
from gateway.schemas import AuthVerifyOut, CircuitStatusOut, ModelPricingOut, TenantOut
from gateway.security import api_key_auth, api_key_header_auth

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NinjaAPI instance configuration
# ---------------------------------------------------------------------------
api = NinjaAPI(
    title="Intelligent Multi-Model LLM Gateway",
    version="4.0.0",
    description=(
        "Enterprise-grade Intelligent Multi-Model LLM Gateway with Dynamic Routing, "
        "Fallbacks, and FinOps Controls.\n\n"
        "- **Phase 1**: Pure ASGI Scaffold, Docker Infrastructure & Concurrent Health Probes\n"
        "- **Phase 2**: Multi-Tenant Isolation, SHA-256 Hashed API Key Auth & Token Pricing Catalog\n"
        "- **Phase 3**: Unified LLM Provider Adapters (Groq, Gemini, OpenAI, Mock) & Real-time FinOps Tracking\n"
        "- **Phase 4**: Dynamic Model Routing, Intelligent Fallbacks & Distributed Redis Circuit Breakers"
    ),
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# Router Registration
# ---------------------------------------------------------------------------
# Public infrastructure probes (unauthenticated for Kubernetes / load balancers)
api.add_router("/health", health_router, tags=["Infrastructure"])


# ---------------------------------------------------------------------------
# Authentication & Tenancy Endpoints
# ---------------------------------------------------------------------------


@api.get(
    "/auth/verify",
    auth=[api_key_auth, api_key_header_auth],
    response=AuthVerifyOut,
    summary="Verify API Key & Retrieve Tenant Context",
    description=(
        "Authenticates the incoming request using `Authorization: Bearer <key>` "
        "or `X-API-Key: <key>`. Hashes the key with SHA-256, verifies existence in "
        "PostgreSQL, updates `last_used_at` asynchronously, and returns the tenant's profile."
    ),
    tags=["Authentication & Tenancy"],
)
async def verify_auth(request) -> AuthVerifyOut:
    """
    Async endpoint verifying credentials.
    Django Ninja sets request.auth to the verified Tenant model instance.
    """
    tenant: Tenant = request.auth
    return AuthVerifyOut(
        authenticated=True,
        tenant=TenantOut.model_validate(tenant),
    )


@api.get(
    "/tenant/me",
    auth=[api_key_auth, api_key_header_auth],
    response=TenantOut,
    summary="Get Current Tenant Details & FinOps Budget Status",
    description=(
        "Returns the authenticated tenant's profile, monthly budget limit, "
        "current spend, remaining budget, and rate limit quotas."
    ),
    tags=["Authentication & Tenancy"],
)
async def get_current_tenant(request) -> TenantOut:
    """
    Returns tenant information attached to request.auth by the security dependency.
    """
    tenant: Tenant = request.auth
    return TenantOut.model_validate(tenant)


# ---------------------------------------------------------------------------
# FinOps & Pricing Catalog
# ---------------------------------------------------------------------------


@api.get(
    "/models/pricing",
    response=List[ModelPricingOut],
    summary="List Active Model Pricing Catalog",
    description=(
        "Retrieves the active LLM pricing catalog across all supported providers "
        "(Groq, OpenAI, Anthropic, Gemini). Prices are listed in USD per 1M tokens."
    ),
    tags=["FinOps & Pricing"],
)
async def list_model_pricing(request) -> List[ModelPricingOut]:
    """
    Non-blocking async query fetching the active model pricing catalog.
    Uses async list comprehension with Django 5.x async queryset iteration (`async for`).
    """
    pricing_list = []
    async for pricing in ModelPricing.objects.filter(is_active=True).order_by("provider", "model_id"):
        pricing_list.append(ModelPricingOut.model_validate(pricing))
    return pricing_list


# ---------------------------------------------------------------------------
# Dynamic Routing & Circuit Breaker Telemetry
# ---------------------------------------------------------------------------


@api.get(
    "/routing/circuits",
    response=List[CircuitStatusOut],
    summary="Inspect Provider Circuit Breakers",
    description=(
        "Returns live distributed circuit breaker states (CLOSED, OPEN, HALF_OPEN), "
        "consecutive failure counts, and recovery timeouts for all upstream providers."
    ),
    tags=["Dynamic Routing & Resiliency"],
)
async def list_circuit_breakers(request) -> List[CircuitStatusOut]:
    """Inspects Redis-backed circuit breaker state across all providers."""
    providers = ["groq", "openai", "gemini", "mock"]
    circuits = await circuit_breaker.get_all_circuits(providers)
    return [CircuitStatusOut(**c) for c in circuits]


@api.post(
    "/routing/circuits/{provider}/reset",
    auth=[api_key_auth, api_key_header_auth],
    summary="Reset Provider Circuit Breaker",
    description="Manually resets a tripped circuit breaker back to CLOSED state.",
    tags=["Dynamic Routing & Resiliency"],
)
async def reset_circuit_breaker(request, provider: str) -> dict:
    """Resets circuit breaker failures and state in Redis."""
    await circuit_breaker.reset(provider)
    return {"status": "success", "provider": provider.lower(), "state": "CLOSED"}


# ---------------------------------------------------------------------------
# Unified Chat Completion Execution Engine (Dynamic Routing & Fallbacks)
# ---------------------------------------------------------------------------


@api.post(
    "/chat/completions",
    auth=[api_key_auth, api_key_header_auth],
    response=ChatCompletionResponse,
    summary="Unified Chat Completion with Dynamic Routing & Fallbacks",
    description=(
        "Executes chat completions with dynamic strategy routing (cost_optimized, fast, quality), "
        "automatic failover, and Redis-backed circuit breaking across upstream providers."
    ),
    tags=["Chat Completions"],
)
async def chat_completions(request, payload: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Async non-blocking chat completion endpoint with dynamic routing and fallbacks.
    1. Authenticates tenant via Bearer / X-API-Key.
    2. Validates FinOps monthly spending budget.
    3. Resolves candidate models and skips tripped (OPEN) circuit breakers.
    4. Automatically falls back upon provider failure.
    5. Calculates exact token costs and updates tenant spend in PostgreSQL.
    """
    tenant: Tenant = request.auth

    # FinOps Budget Enforcement
    if not tenant.has_budget_remaining():
        logger.warning("Tenant '%s' exceeded monthly budget cap of $%s", tenant.name, tenant.monthly_budget_usd)
        raise HttpError(402, "Tenant monthly spending budget exceeded. Operation blocked by FinOps policy.")

    try:
        response = await RoutingOrchestrator.route_and_execute(request=payload, tenant=tenant)
        return response

    except ModelNotSupportedError as exc:
        raise HttpError(400, str(exc))
    except ProviderAuthError as exc:
        raise HttpError(502, f"Upstream provider authentication failed: {exc.message}")
    except ProviderRateLimitError as exc:
        raise HttpError(429, f"Upstream provider rate limit reached: {exc.message}")
    except ProviderTimeoutError as exc:
        raise HttpError(504, f"Upstream provider request timed out: {exc.message}")
    except ProviderAPIError as exc:
        raise HttpError(502, f"Upstream provider returned an error: {exc.message}")
    except Exception as exc:
        logger.exception("Unexpected error during chat completion execution: %s", exc)
        raise HttpError(500, f"Internal gateway error: {str(exc)}")
