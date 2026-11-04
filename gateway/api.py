"""
gateway/api.py
==============
Django Ninja root API instance for the LLM Gateway.

Central Router:
  - Auto-generates OpenAPI 3.1 documentation at /api/v1/docs and /openapi.json.
  - Implements pure async def views to prevent thread blocking under ASGI.
  - Secures endpoints via cryptographically hashed API Keys using `APIKeyAuth`.
  - Dispatches chat completions to unified provider adapters (Groq, Gemini, OpenAI, Mock).
"""
import logging
from typing import List

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
from gateway.adapters.factory import ProviderFactory
from gateway.adapters.schemas import ChatCompletionRequest, ChatCompletionResponse
from gateway.health.router import router as health_router
from gateway.models import ModelPricing, Tenant
from gateway.schemas import AuthVerifyOut, ModelPricingOut, TenantOut
from gateway.security import api_key_auth, api_key_header_auth

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NinjaAPI instance configuration
# ---------------------------------------------------------------------------
api = NinjaAPI(
    title="Intelligent Multi-Model LLM Gateway",
    version="3.0.0",
    description=(
        "Enterprise-grade Intelligent Multi-Model LLM Gateway with Dynamic Routing, "
        "Fallbacks, and FinOps Controls.\n\n"
        "- **Phase 1**: Pure ASGI Scaffold, Docker Infrastructure & Concurrent Health Probes\n"
        "- **Phase 2**: Multi-Tenant Isolation, SHA-256 Hashed API Key Auth & Token Pricing Catalog\n"
        "- **Phase 3**: Unified LLM Provider Adapters (Groq, Gemini, OpenAI, Mock) & Real-time FinOps Tracking"
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
# Phase 3: Unified Chat Completion Execution Engine
# ---------------------------------------------------------------------------


@api.post(
    "/chat/completions",
    auth=[api_key_auth, api_key_header_auth],
    response=ChatCompletionResponse,
    summary="Unified Chat Completion Execution",
    description=(
        "Executes a chat completion across integrated providers (Groq, OpenAI, Gemini, Mock). "
        "Enforces tenant budget checks, normalizes request/response format to OpenAI specifications, "
        "and computes per-token FinOps expenses in real time."
    ),
    tags=["Chat Completions"],
)
async def chat_completions(request, payload: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Async non-blocking chat completion endpoint.
    1. Authenticates tenant via Bearer/X-API-Key.
    2. Enforces FinOps monthly spending budget.
    3. Resolves and dispatches to appropriate provider adapter.
    4. Calculates token cost and atomically logs spend to PostgreSQL.
    """
    tenant: Tenant = request.auth

    # FinOps Budget Enforcement
    if not tenant.has_budget_remaining():
        logger.warning("Tenant '%s' exceeded monthly budget cap of $%s", tenant.name, tenant.monthly_budget_usd)
        raise HttpError(402, "Tenant monthly spending budget exceeded. Operation blocked by FinOps policy.")

    try:
        response = await ProviderFactory.execute_completion(request=payload, tenant=tenant)
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
