"""
gateway/api.py
==============
Django Ninja root API instance for the LLM Gateway.

Central Router:
  - Auto-generates OpenAPI 3.1 documentation at /api/v1/docs and /openapi.json.
  - Implements pure async def views to prevent thread blocking under ASGI.
  - Secures endpoints via cryptographically hashed API Keys using `APIKeyAuth`.
  - Dispatches chat completions via dynamic routing, circuit breaking, and fallbacks.
  - Enforces distributed multi-tenant rate limiting (RPM/TPM) and FinOps budget quotas.
"""
import logging
from typing import List, Optional

from asgiref.sync import sync_to_async
from django.http import HttpResponse, JsonResponse
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
from gateway.ratelimit.budget import FinOpsBudgetManager
from gateway.ratelimit.limiter import rate_limiter
from gateway.routing.circuit_breaker import circuit_breaker
from gateway.routing.orchestrator import RoutingOrchestrator
from gateway.schemas import (
    AuthVerifyOut,
    BudgetTopUpIn,
    CircuitStatusOut,
    ModelPricingOut,
    RateLimitStatusOut,
    TenantOut,
)
from gateway.security import api_key_auth, api_key_header_auth

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NinjaAPI instance configuration
# ---------------------------------------------------------------------------
api = NinjaAPI(
    title="Intelligent Multi-Model LLM Gateway",
    version="5.0.0",
    description=(
        "Enterprise-grade Intelligent Multi-Model LLM Gateway with Dynamic Routing, "
        "Fallbacks, and FinOps Controls.\n\n"
        "- **Phase 1**: Pure ASGI Scaffold, Docker Infrastructure & Concurrent Health Probes\n"
        "- **Phase 2**: Multi-Tenant Isolation, SHA-256 Hashed API Key Auth & Token Pricing Catalog\n"
        "- **Phase 3**: Unified LLM Provider Adapters (Groq, Gemini, OpenAI, Mock) & Real-time FinOps Tracking\n"
        "- **Phase 4**: Dynamic Model Routing, Intelligent Fallbacks & Distributed Redis Circuit Breakers\n"
        "- **Phase 5**: Distributed Rate Limiting (RPM/TPM) & FinOps Soft/Hard Budget Controls"
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


@api.post(
    "/tenant/budget/top-up",
    auth=[api_key_auth, api_key_header_auth],
    response=TenantOut,
    summary="Top Up Tenant Spending Budget Cap",
    description="Adds funds to the tenant's monthly budget cap in USD.",
    tags=["Authentication & Tenancy"],
)
async def top_up_budget(request, payload: BudgetTopUpIn) -> TenantOut:
    """FinOps top-up endpoint allowing budget adjustments."""
    tenant: Tenant = request.auth
    updated_tenant = await FinOpsBudgetManager.top_up_budget(tenant.id, payload.additional_budget_usd)
    return TenantOut.model_validate(updated_tenant)


@api.get(
    "/tenant/ratelimit",
    auth=[api_key_auth, api_key_header_auth],
    response=RateLimitStatusOut,
    summary="Check Tenant Rate Limit Standing",
    description="Returns remaining RPM and TPM allowances for the active time window.",
    tags=["Authentication & Tenancy"],
)
async def get_ratelimit_status(request) -> RateLimitStatusOut:
    """Retrieves current rate limit counters from Redis."""
    tenant: Tenant = request.auth
    rpm_res = await rate_limiter.get_current_rpm(str(tenant.id), tenant.rpm_limit)
    return RateLimitStatusOut(
        rpm_limit=tenant.rpm_limit,
        rpm_remaining=rpm_res.remaining,
        tpm_limit=tenant.tpm_limit,
        tpm_remaining=tenant.tpm_limit,  # Baseline allocation
        window_reset_seconds=rpm_res.reset_seconds,
    )


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
# Unified Chat Completion Execution Engine (Rate Limiting + Budget + Routing)
# ---------------------------------------------------------------------------


@api.post(
    "/chat/completions",
    auth=[api_key_auth, api_key_header_auth],
    response=ChatCompletionResponse,
    summary="Unified Chat Completion Execution",
    description=(
        "Executes chat completions with distributed rate limiting (RPM/TPM), "
        "FinOps budget controls, dynamic routing strategies, and automatic failover."
    ),
    tags=["Chat Completions"],
)
async def chat_completions(
    request,
    payload: ChatCompletionRequest,
    response: HttpResponse,
) -> ChatCompletionResponse:
    """
    Async non-blocking chat completion endpoint with full enterprise controls:
      1. Authenticates tenant via Bearer / X-API-Key.
      2. Enforces FinOps hard budget limit (HTTP 402).
      3. Enforces Redis-backed Requests Per Minute (RPM) rate limit (HTTP 429).
      4. Dispatches request through Dynamic Routing & Fallback Orchestrator.
      5. Enforces Tokens Per Minute (TPM) quota post-flight.
      6. Injects RFC rate limit and soft budget warning headers into HTTP response.
    """
    tenant: Tenant = request.auth
    tenant_id_str = str(tenant.id)

    # 1. FinOps Pre-flight Budget Guard
    budget_eval = FinOpsBudgetManager.evaluate_budget(tenant)
    if not budget_eval.allowed:
        logger.warning(
            "Tenant '%s' blocked by FinOps hard budget lockout ($%s / $%s)",
            tenant.name,
            tenant.current_spend_usd,
            tenant.monthly_budget_usd,
        )
        raise HttpError(
            402,
            f"Tenant monthly spending budget exceeded (${tenant.current_spend_usd} / ${tenant.monthly_budget_usd}). "
            "Operation blocked by FinOps policy. Top up budget to resume access.",
        )

    # 2. Redis-backed RPM Rate Limiting Guard
    rpm_res = await rate_limiter.check_rpm(tenant_id_str, tenant.rpm_limit)
    if not rpm_res.allowed:
        logger.warning("Tenant '%s' exceeded RPM quota (%d requests/min)", tenant.name, tenant.rpm_limit)
        return JsonResponse(
            {
                "detail": (
                    f"Rate limit exceeded: Requests Per Minute (RPM) limit of {tenant.rpm_limit} reached. "
                    f"Please retry in {rpm_res.retry_after} seconds."
                )
            },
            status=429,
            headers={
                "Retry-After": str(rpm_res.retry_after),
                "X-RateLimit-Limit-RPM": str(rpm_res.limit),
                "X-RateLimit-Remaining-RPM": "0",
                "X-RateLimit-Reset": str(rpm_res.reset_seconds),
            },
        )

    # 3. Model Dispatch via Dynamic Routing Orchestrator
    try:
        completion_response = await RoutingOrchestrator.route_and_execute(
            request=payload,
            tenant=tenant,
        )

        # 4. Post-flight TPM Rate Limiting
        total_tokens = completion_response.usage.total_tokens
        tpm_res = await rate_limiter.check_and_consume_tpm(tenant_id_str, tenant.tpm_limit, total_tokens)

        # 5. Inject Standard RFC & FinOps Response Headers
        response["X-RateLimit-Limit-RPM"] = str(rpm_res.limit)
        response["X-RateLimit-Remaining-RPM"] = str(rpm_res.remaining)
        response["X-RateLimit-Limit-TPM"] = str(tpm_res.limit)
        response["X-RateLimit-Remaining-TPM"] = str(tpm_res.remaining)
        response["X-RateLimit-Reset"] = str(rpm_res.reset_seconds)

        if budget_eval.warning_header:
            response["X-Budget-Warning"] = budget_eval.warning_header

        return completion_response

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
