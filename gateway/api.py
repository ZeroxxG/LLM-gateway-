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
import asyncio
from datetime import timedelta
from decimal import Decimal
import json
import logging
import time
from typing import Any, List, Optional
import uuid

from asgiref.sync import sync_to_async
from celery.result import AsyncResult
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.utils import timezone
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
from gateway.benchmarks import BenchmarkRunner
from gateway.caching import PromptCacheManager
from gateway.health.router import router as health_router
from gateway.models import DailyUsageAggregate, ModelPricing, RequestAuditLog, Tenant
from gateway.ratelimit.budget import FinOpsBudgetManager
from gateway.ratelimit.limiter import rate_limiter
from gateway.routing.circuit_breaker import circuit_breaker
from gateway.routing.heuristics import HeuristicRouter
from gateway.routing.orchestrator import RoutingOrchestrator
from gateway.routing.router import DynamicRouter
from gateway.schemas import (
    AsyncTaskStatusOut,
    AsyncTaskTriggerOut,
    AuthVerifyOut,
    BenchmarkComparisonOut,
    BenchmarkRunIn,
    BudgetSyncOut,
    BudgetTopUpIn,
    CacheClearOut,
    CacheStatsOut,
    CandidateScoreOut,
    CircuitStatusOut,
    DailyUsageAggregateOut,
    LatencyPercentilesOut,
    ModelPricingOut,
    ObservabilitySummaryOut,
    RateLimitStatusOut,
    RealtimeBudgetStatusOut,
    RequestAuditLogOut,
    RoutingEvaluationIn,
    RoutingEvaluationOut,
    TenantOut,
    TenantUsageAnalyticsOut,
    SecurityScanIn,
    SecurityScanOut,
)
from gateway.guardrails import GuardrailViolationException, SecurityGuardrailManager
from gateway.observability import (
    CONTENT_TYPE_LATEST,
    LatencyTracker,
    gateway_tracer,
    get_prometheus_metrics_text,
    record_cache_metric,
    record_request_metrics,
    update_circuit_breaker_metric,
)
from gateway.security import api_key_auth, api_key_header_auth
from gateway.tasks import aggregate_daily_usage_task, dispatch_audit_log_async

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NinjaAPI instance configuration
# ---------------------------------------------------------------------------
api = NinjaAPI(
    title="Intelligent Multi-Model LLM Gateway",
    version="7.0.0",
    description=(
        "Enterprise-grade Intelligent Multi-Model LLM Gateway with Dynamic Routing, "
        "Fallbacks, and FinOps Controls.\n\n"
        "- **Phase 1**: Pure ASGI Scaffold, Docker Infrastructure & Concurrent Health Probes\n"
        "- **Phase 2**: Multi-Tenant Isolation, SHA-256 Hashed API Key Auth & Token Pricing Catalog\n"
        "- **Phase 3**: Unified LLM Provider Adapters (Groq, Gemini, OpenAI, Mock) & Real-time FinOps Tracking\n"
        "- **Phase 4**: Dynamic Model Routing, Intelligent Fallbacks & Distributed Redis Circuit Breakers\n"
        "- **Phase 5**: Distributed Rate Limiting (RPM/TPM) & FinOps Soft/Hard Budget Controls\n"
        "- **Phase 6**: Celery Background Workers, Asynchronous Usage Aggregation & Analytics\n"
        "- **Phase 7**: Real-Time Budget Enforcement – Atomic Redis spend tracking (INCRBYFLOAT) & hard-stop 402 rejection\n"
        "- **Phase 8**: Sub-2ms Heuristic Multi-Factor Routing Engine\n"
        "- **Phase 9**: Real-Time Token Streaming via Server-Sent Events (SSE)\n"
        "- **Phase 10**: Deterministic SHA-256 Prompt Response Caching with Invalidation Controls\n"
        "- **Phase 11**: Observability & Tracing – Prometheus collector registry and latency quantiles\n"
        "- **Phase 12**: Benchmark Harness & Baseline Evaluation (40-item comparative dataset)\n"
        "- **Phase 13**: Security, Guardrails & Content Moderation – Sub-2ms PII masking, injection defense & moderation"
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
    "/tenant/budget",
    auth=[api_key_auth, api_key_header_auth],
    response=RealtimeBudgetStatusOut,
    summary="Get Real-Time Redis Budget Standing",
    description="Returns real-time tenant budget cap, spend accumulated in Redis via INCRBYFLOAT, and lockout state.",
    tags=["Authentication & Tenancy"],
)
async def get_realtime_budget(request) -> RealtimeBudgetStatusOut:
    """Check live budget standing from Redis."""
    tenant: Tenant = request.auth
    data = await FinOpsBudgetManager.get_realtime_budget(tenant.id)
    return RealtimeBudgetStatusOut(
        tenant_id=tenant.id,
        monthly_budget_usd=data["monthly_budget_usd"],
        current_spend_usd=data["current_spend_usd"],
        remaining_budget_usd=data["remaining_budget_usd"],
        percentage_consumed=data["percentage_consumed"],
        is_locked_out=data["is_locked_out"],
    )


@api.post(
    "/tenant/budget/sync",
    auth=[api_key_auth, api_key_header_auth],
    response=BudgetSyncOut,
    summary="Sync Redis Spend to PostgreSQL Database",
    description="Flushes accumulated Redis spend back to the tenant record in PostgreSQL.",
    tags=["Authentication & Tenancy"],
)
async def sync_budget_to_db(request) -> BudgetSyncOut:
    """Sync Redis spend to PostgreSQL."""
    tenant: Tenant = request.auth
    synced_tenant = await FinOpsBudgetManager.sync_spend_to_db(tenant.id)
    return BudgetSyncOut(
        tenant_id=synced_tenant.id,
        synced_spend_usd=synced_tenant.current_spend_usd,
        status="synced",
    )


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
# Prompt Response Caching & Acceleration
# ---------------------------------------------------------------------------


@api.post(
    "/cache/clear",
    auth=[api_key_auth, api_key_header_auth],
    response=CacheClearOut,
    summary="Clear Tenant Prompt Response Cache",
    description="Flushes all cached prompt responses in Redis for the authenticated tenant.",
    tags=["Caching & Acceleration"],
)
async def clear_cache(request) -> CacheClearOut:
    """Invalidates all cached completion entries for the calling tenant."""
    tenant: Tenant = request.auth
    deleted = await PromptCacheManager.clear_tenant_cache(str(tenant.id))
    return CacheClearOut(
        tenant_id=tenant.id,
        status="cleared",
        keys_deleted=deleted,
    )


@api.get(
    "/cache/stats",
    auth=[api_key_auth, api_key_header_auth],
    response=CacheStatsOut,
    summary="Get Cache Performance Telemetry",
    description="Returns real-time cache hits, misses, hit ratio, and active cached keys in Redis for the authenticated tenant.",
    tags=["Caching & Acceleration"],
)
async def get_cache_stats(request) -> CacheStatsOut:
    """Returns hit/miss metrics and active cache key count from Redis."""
    tenant: Tenant = request.auth
    stats = await PromptCacheManager.get_stats(str(tenant.id))
    return CacheStatsOut(
        tenant_id=tenant.id,
        hits=stats["hits"],
        misses=stats["misses"],
        total_requests=stats["total_requests"],
        hit_ratio=stats["hit_ratio"],
        cached_keys=stats["cached_keys"],
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
    for c in circuits:
        update_circuit_breaker_metric(c["provider"], c["state"])
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
    update_circuit_breaker_metric(provider, "CLOSED")
    return {"status": "success", "provider": provider.lower(), "state": "CLOSED"}


@api.post(
    "/routing/evaluate",
    auth=[api_key_auth, api_key_header_auth],
    response=RoutingEvaluationOut,
    summary="Evaluate Heuristic Routing Decision",
    description=(
        "Simulates sub-2ms heuristic routing without executing LLM completions. "
        "Returns detected task type, context token estimate, decision latency, "
        "and candidate model scoring breakdowns."
    ),
    tags=["Dynamic Routing & Resiliency"],
)
async def evaluate_heuristic_routing(request, payload: RoutingEvaluationIn) -> RoutingEvaluationOut:
    """Evaluates heuristic routing in < 2ms."""
    tenant: Tenant = request.auth
    result = await HeuristicRouter.evaluate_routing(
        messages=payload.messages,
        task_type_override=payload.task_type,
        latency_sla_ms=payload.latency_sla_ms,
        max_cost_usd=payload.max_cost_usd,
        tenant=tenant,
    )
    return RoutingEvaluationOut(
        detected_task_type=result.detected_task_type.value,
        estimated_tokens=result.estimated_tokens,
        decision_latency_ms=result.decision_latency_ms,
        selected_model=result.selected_model,
        candidate_chain=result.candidate_chain,
        candidates=[
            CandidateScoreOut(
                model_id=c.model_id,
                provider=c.provider,
                score=c.score,
                capability_score=c.capability_score,
                latency_score=c.latency_score,
                cost_score=c.cost_score,
                estimated_cost_usd=c.estimated_cost_usd,
                typical_latency_ms=c.typical_latency_ms,
                eligible=c.eligible,
                rejection_reason=c.rejection_reason,
            )
            for c in result.candidates
        ],
        strategy_used=result.strategy_used,
    )


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
    trace_id = request.headers.get("x-trace-id") or gateway_tracer.generate_trace_id()
    start_time = time.time()

    # 1. FinOps Pre-flight Real-Time Budget Guard (Sub-millisecond in Redis)
    budget_eval = await FinOpsBudgetManager.evaluate_budget_realtime(tenant)
    if not budget_eval.allowed:
        logger.warning(
            "Tenant '%s' blocked by FinOps hard budget lockout ($%s / $%s)",
            tenant.name,
            budget_eval.current_spend_usd,
            budget_eval.monthly_budget_usd,
        )
        raise HttpError(
            402,
            f"Tenant monthly spending budget exceeded (${budget_eval.current_spend_usd} / ${budget_eval.monthly_budget_usd}). "
            "Operation blocked by FinOps real-time policy. Top up budget to resume access.",
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
                "X-Trace-Id": trace_id,
            },
        )

    # 3. Pre-flight Security Guardrails Scan (PII Masking, Prompt Injection, Content Moderation)
    try:
        guardrail_result = SecurityGuardrailManager.inspect_request(
            payload=payload,
            mask_pii=True,
            check_injection=True,
            check_moderation=True,
            block_on_violation=True,
        )
    except GuardrailViolationException as exc:
        logger.warning(
            "Tenant '%s' request blocked by security guardrails: %s (risk: %s)",
            tenant.name,
            exc.message,
            exc.injection_risk,
        )
        return JsonResponse(
            {
                "error": {
                    "type": "guardrail_violation",
                    "message": exc.message,
                    "violations": exc.violations,
                    "injection_risk": exc.injection_risk,
                }
            },
            status=exc.status_code,
            headers={
                "X-Security-Action": exc.action,
                "X-Security-Injection-Risk": exc.injection_risk,
                "X-Trace-Id": trace_id,
            },
        )

    security_action = guardrail_result.action
    security_pii = ",".join(guardrail_result.pii_detected_types)

    # 4. Resolve routing strategy
    strategy_name = getattr(payload, "strategy", None) or DynamicRouter.detect_strategy(payload.model).value

    # 5. Streaming Response Dispatch via Server-Sent Events (SSE)
    if getattr(payload, "stream", False):
        stream_start_time = time.time()

        async def sse_event_stream():
            stream_chunks = 0
            try:
                async for chunk in RoutingOrchestrator.stream_and_execute(request=payload, tenant=tenant):
                    stream_chunks += 1
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                duration = max(0.001, time.time() - stream_start_time)
                record_request_metrics(
                    tenant_id=tenant_id_str,
                    model=payload.model,
                    provider="streaming",
                    status_code=200,
                    routing_strategy=strategy_name,
                    duration_seconds=duration,
                    prompt_tokens=stream_chunks * 2,
                    completion_tokens=stream_chunks * 2,
                )
                await LatencyTracker.record_latency(tenant_id_str, duration * 1000.0, model=payload.model)
            except Exception as exc:
                logger.exception("Error during SSE streaming: %s", exc)
                err_json = json.dumps({"error": {"message": str(exc), "type": "stream_error"}})
                yield f"data: {err_json}\n\n"
                yield "data: [DONE]\n\n"

        stream_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-RateLimit-Limit-RPM": str(rpm_res.limit),
            "X-RateLimit-Remaining-RPM": str(rpm_res.remaining),
            "X-RateLimit-Reset": str(rpm_res.reset_seconds),
            "X-Budget-Limit": str(budget_eval.monthly_budget_usd),
            "X-Budget-Current-Spend": str(budget_eval.current_spend_usd),
            "X-Budget-Remaining": str(budget_eval.remaining_usd),
            "X-Routing-Strategy": strategy_name,
            "X-Trace-Id": trace_id,
            "X-Security-Action": security_action,
        }
        if security_pii:
            stream_headers["X-Security-PII-Detected"] = security_pii
        if budget_eval.warning_header:
            stream_headers["X-Budget-Warning"] = budget_eval.warning_header

        return StreamingHttpResponse(
            sse_event_stream(),
            content_type="text/event-stream",
            headers=stream_headers,
        )

    # 5. Check Prompt Response Cache in Redis
    cache_bypass = (
        request.headers.get("cache-control", "").lower() == "no-cache"
        or request.headers.get("x-cache-bypass", "").lower() == "true"
    )
    cache_key = PromptCacheManager.compute_cache_key(tenant_id_str, payload)

    if not cache_bypass:
        cached_response = await PromptCacheManager.get(cache_key)
        if cached_response is not None:
            await PromptCacheManager.record_hit(tenant_id_str)
            record_cache_metric(tenant_id_str, "hit")
            hit_duration = max(0.0005, time.time() - start_time)
            record_request_metrics(
                tenant_id=tenant_id_str,
                model=cached_response.model,
                provider=cached_response.provider,
                status_code=200,
                routing_strategy=strategy_name,
                duration_seconds=hit_duration,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
            )
            await LatencyTracker.record_latency(tenant_id_str, hit_duration * 1000.0, model=cached_response.model)
            response["X-Cache"] = "HIT"
            response["X-Cache-Key"] = cache_key
            response["X-Trace-Id"] = trace_id
            response["X-RateLimit-Limit-RPM"] = str(rpm_res.limit)
            response["X-RateLimit-Remaining-RPM"] = str(rpm_res.remaining)
            response["X-RateLimit-Reset"] = str(rpm_res.reset_seconds)
            response["X-Budget-Limit"] = str(budget_eval.monthly_budget_usd)
            response["X-Budget-Current-Spend"] = str(budget_eval.current_spend_usd)
            response["X-Budget-Remaining"] = str(budget_eval.remaining_usd)
            response["X-Routing-Strategy"] = strategy_name
            response["X-Routing-Selected-Model"] = cached_response.model
            response["X-Security-Action"] = security_action
            if security_pii:
                response["X-Security-PII-Detected"] = security_pii
            if budget_eval.warning_header:
                response["X-Budget-Warning"] = budget_eval.warning_header
            return cached_response
        else:
            await PromptCacheManager.record_miss(tenant_id_str)
            record_cache_metric(tenant_id_str, "miss")
            response["X-Cache"] = "MISS"
            response["X-Cache-Key"] = cache_key
    else:
        record_cache_metric(tenant_id_str, "bypass")
        response["X-Cache"] = "BYPASS"
        response["X-Cache-Key"] = cache_key

    # 6. Model Dispatch via Dynamic Routing Orchestrator
    try:
        completion_response = await RoutingOrchestrator.route_and_execute(
            request=payload,
            tenant=tenant,
        )
        latency_ms = (time.time() - start_time) * 1000.0

        # Store in Redis cache asynchronously if caching was enabled
        if not cache_bypass:
            await PromptCacheManager.set(cache_key, completion_response, ttl_seconds=3600)

        # Post-flight TPM Rate Limiting
        total_tokens = completion_response.usage.total_tokens
        tpm_res = await rate_limiter.check_and_consume_tpm(tenant_id_str, tenant.tpm_limit, total_tokens)

        # Prometheus Metrics & Observability Telemetry
        record_request_metrics(
            tenant_id=tenant_id_str,
            model=completion_response.model,
            provider=completion_response.provider,
            status_code=200,
            routing_strategy=strategy_name,
            duration_seconds=latency_ms / 1000.0,
            prompt_tokens=completion_response.usage.prompt_tokens,
            completion_tokens=completion_response.usage.completion_tokens,
            cost_usd=float(completion_response.cost_usd),
        )
        await LatencyTracker.record_latency(tenant_id_str, latency_ms, model=completion_response.model)
        await gateway_tracer.export_trace(
            trace_id=trace_id,
            tenant_id=tenant_id_str,
            model=completion_response.model,
            provider=completion_response.provider,
            prompt_tokens=completion_response.usage.prompt_tokens,
            completion_tokens=completion_response.usage.completion_tokens,
            latency_ms=latency_ms,
            cost_usd=float(completion_response.cost_usd),
            status="success",
        )

        # Asynchronous Audit Logging via Celery (Non-blocking)
        audit_payload = {
            "tenant_id": tenant_id_str,
            "request_id": completion_response.id,
            "model_id": completion_response.model,
            "provider": completion_response.provider,
            "prompt_tokens": completion_response.usage.prompt_tokens,
            "completion_tokens": completion_response.usage.completion_tokens,
            "total_tokens": completion_response.usage.total_tokens,
            "cost_usd": str(completion_response.cost_usd),
            "latency_ms": round(latency_ms, 2),
            "status_code": 200,
            "routing_strategy": strategy_name,
            "fallback_occurred": (completion_response.model != payload.model),
        }
        await dispatch_audit_log_async(audit_payload)

        # Inject Standard RFC, Trace, & FinOps Response Headers
        response["X-Trace-Id"] = trace_id
        response["X-RateLimit-Limit-RPM"] = str(rpm_res.limit)
        response["X-RateLimit-Remaining-RPM"] = str(rpm_res.remaining)
        response["X-RateLimit-Limit-TPM"] = str(tpm_res.limit)
        response["X-RateLimit-Remaining-TPM"] = str(tpm_res.remaining)
        response["X-RateLimit-Reset"] = str(rpm_res.reset_seconds)

        response["X-Budget-Limit"] = str(budget_eval.monthly_budget_usd)
        response["X-Budget-Current-Spend"] = str(budget_eval.current_spend_usd)
        response["X-Budget-Remaining"] = str(budget_eval.remaining_usd)

        if budget_eval.warning_header:
            response["X-Budget-Warning"] = budget_eval.warning_header

        # Inject Security Guardrail Response Headers
        response["X-Security-Action"] = security_action
        if security_pii:
            response["X-Security-PII-Detected"] = security_pii

        # Inject Heuristic Routing Diagnostic Headers
        if completion_response.routing_decision:
            rd = completion_response.routing_decision
            response["X-Routing-Strategy"] = str(rd.get("strategy", strategy_name))
            response["X-Routing-Task-Type"] = str(rd.get("detected_task_type", "general"))
            response["X-Routing-Decision-Latency-Ms"] = str(rd.get("decision_latency_ms", "0.0"))
            response["X-Routing-Selected-Model"] = str(rd.get("selected_model", completion_response.model))
        else:
            response["X-Routing-Strategy"] = strategy_name
            response["X-Routing-Selected-Model"] = completion_response.model

        return completion_response

    except ModelNotSupportedError as exc:
        record_request_metrics(
            tenant_id=tenant_id_str,
            model=payload.model,
            provider="unknown",
            status_code=400,
            routing_strategy=strategy_name,
            duration_seconds=max(0.001, time.time() - start_time),
        )
        await dispatch_audit_log_async({
            "tenant_id": tenant_id_str,
            "request_id": f"err_{uuid.uuid4().hex[:12]}",
            "model_id": payload.model,
            "provider": "unknown",
            "status_code": 400,
            "error_message": str(exc),
            "routing_strategy": strategy_name,
        })
        raise HttpError(400, str(exc))
    except ProviderAuthError as exc:
        record_request_metrics(
            tenant_id=tenant_id_str,
            model=payload.model,
            provider=exc.provider,
            status_code=502,
            routing_strategy=strategy_name,
            duration_seconds=max(0.001, time.time() - start_time),
        )
        await dispatch_audit_log_async({
            "tenant_id": tenant_id_str,
            "request_id": f"err_{uuid.uuid4().hex[:12]}",
            "model_id": payload.model,
            "provider": exc.provider,
            "status_code": 502,
            "error_message": exc.message,
            "routing_strategy": strategy_name,
        })
        raise HttpError(502, f"Upstream provider authentication failed: {exc.message}")
    except ProviderRateLimitError as exc:
        record_request_metrics(
            tenant_id=tenant_id_str,
            model=payload.model,
            provider=exc.provider,
            status_code=429,
            routing_strategy=strategy_name,
            duration_seconds=max(0.001, time.time() - start_time),
        )
        await dispatch_audit_log_async({
            "tenant_id": tenant_id_str,
            "request_id": f"err_{uuid.uuid4().hex[:12]}",
            "model_id": payload.model,
            "provider": exc.provider,
            "status_code": 429,
            "error_message": exc.message,
            "routing_strategy": strategy_name,
        })
        raise HttpError(429, f"Upstream provider rate limit reached: {exc.message}")
    except ProviderTimeoutError as exc:
        record_request_metrics(
            tenant_id=tenant_id_str,
            model=payload.model,
            provider=exc.provider,
            status_code=504,
            routing_strategy=strategy_name,
            duration_seconds=max(0.001, time.time() - start_time),
        )
        await dispatch_audit_log_async({
            "tenant_id": tenant_id_str,
            "request_id": f"err_{uuid.uuid4().hex[:12]}",
            "model_id": payload.model,
            "provider": exc.provider,
            "status_code": 504,
            "error_message": exc.message,
            "routing_strategy": strategy_name,
        })
        raise HttpError(504, f"Upstream provider request timed out: {exc.message}")
    except ProviderAPIError as exc:
        record_request_metrics(
            tenant_id=tenant_id_str,
            model=payload.model,
            provider=exc.provider,
            status_code=502,
            routing_strategy=strategy_name,
            duration_seconds=max(0.001, time.time() - start_time),
        )
        await dispatch_audit_log_async({
            "tenant_id": tenant_id_str,
            "request_id": f"err_{uuid.uuid4().hex[:12]}",
            "model_id": payload.model,
            "provider": exc.provider,
            "status_code": 502,
            "error_message": exc.message,
            "routing_strategy": strategy_name,
        })
        raise HttpError(502, f"Upstream provider returned an error: {exc.message}")
    except Exception as exc:
        record_request_metrics(
            tenant_id=tenant_id_str,
            model=payload.model,
            provider="unknown",
            status_code=500,
            routing_strategy=strategy_name,
            duration_seconds=max(0.001, time.time() - start_time),
        )
        logger.exception("Unexpected error during chat completion execution: %s", exc)
        raise HttpError(500, f"Internal gateway error: {str(exc)}")


# ---------------------------------------------------------------------------
# Asynchronous Analytics & Background Worker Telemetry
# ---------------------------------------------------------------------------


@api.get(
    "/analytics/usage",
    auth=[api_key_auth, api_key_header_auth],
    response=TenantUsageAnalyticsOut,
    summary="Get Tenant Usage Analytics & Aggregates",
    description="Returns aggregated token usage, request counts, and FinOps costs over a specified day window.",
    tags=["Analytics & Background Workers"],
)
async def get_tenant_analytics(request, days: int = 7) -> TenantUsageAnalyticsOut:
    """Return daily usage and token stats for this tenant."""
    tenant: Tenant = request.auth
    start_date = timezone.now().date() - timedelta(days=days)

    daily_aggregates = []
    total_requests = 0
    total_tokens = 0
    total_cost = Decimal("0.000000")

    async for agg in DailyUsageAggregate.objects.filter(
        tenant=tenant,
        date__gte=start_date,
    ).order_by("-date"):
        daily_aggregates.append(DailyUsageAggregateOut.model_validate(agg))
        total_requests += agg.total_requests
        total_tokens += agg.total_tokens
        total_cost += agg.total_cost_usd

    return TenantUsageAnalyticsOut(
        tenant_id=tenant.id,
        tenant_name=tenant.name,
        total_requests=total_requests,
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        daily_breakdown=daily_aggregates,
    )


@api.get(
    "/analytics/logs",
    auth=[api_key_auth, api_key_header_auth],
    response=List[RequestAuditLogOut],
    summary="List Granular Request Audit Logs",
    description="Retrieves paginated, immutable audit logs recorded asynchronously by Celery workers.",
    tags=["Analytics & Background Workers"],
)
async def list_audit_logs(
    request,
    limit: int = 20,
    offset: int = 0,
    status_code: Optional[int] = None,
) -> List[RequestAuditLogOut]:
    """Return recent request logs for this tenant."""
    tenant: Tenant = request.auth
    qs = RequestAuditLog.objects.filter(tenant=tenant)
    if status_code is not None:
        qs = qs.filter(status_code=status_code)

    logs = []
    async for log in qs.order_by("-created_at")[offset : offset + limit]:
        logs.append(RequestAuditLogOut.model_validate(log))
    return logs


@api.post(
    "/analytics/aggregate",
    auth=[api_key_auth, api_key_header_auth],
    response=AsyncTaskTriggerOut,
    summary="Trigger Daily Usage Aggregation Celery Task",
    description="Queues an asynchronous Celery background task to roll up recent audit logs into daily aggregates.",
    tags=["Analytics & Background Workers"],
)
async def trigger_aggregation(request, target_date: Optional[str] = None) -> AsyncTaskTriggerOut:
    """Queue a background task to roll up daily stats."""
    tenant: Tenant = request.auth
    task = await asyncio.to_thread(aggregate_daily_usage_task.delay, target_date_str=target_date)
    return AsyncTaskTriggerOut(
        task_id=task.id,
        status="PENDING",
        message=f"Daily aggregation task queued for target date '{target_date or 'today'}'.",
    )


@api.get(
    "/analytics/tasks/{task_id}",
    auth=[api_key_auth, api_key_header_auth],
    response=AsyncTaskStatusOut,
    summary="Check Background Task Execution Status",
    description="Queries Celery and Redis result backend to check completion and result payload of an async task.",
    tags=["Analytics & Background Workers"],
)
async def get_task_status(request, task_id: str) -> AsyncTaskStatusOut:
    """Check task status from Celery and Redis."""
    tenant: Tenant = request.auth

    def _inspect_task(tid: str):
        async_res = AsyncResult(tid)
        state = async_res.state
        ready = async_res.ready()
        successful = async_res.successful() if ready else False
        result = async_res.result if ready else None
        if isinstance(result, Exception):
            result = str(result)
        return state, ready, successful, result

    state, ready, successful, result = await asyncio.to_thread(_inspect_task, task_id)

    return AsyncTaskStatusOut(
        task_id=task_id,
        status=state,
        ready=ready,
        successful=successful,
        result=result,
    )


# ---------------------------------------------------------------------------
# Observability, Prometheus Metrics & Distributed Tracing
# ---------------------------------------------------------------------------


@api.get(
    "/metrics",
    auth=None,
    summary="Scrape Prometheus Metrics",
    description=(
        "Public Prometheus text scrape endpoint. Exposes real-time request counts, "
        "duration histograms (with p50/p95/p99 quantile buckets), token counters, "
        "FinOps costs, cache hit/miss counters, and upstream circuit breaker gauges."
    ),
    tags=["Observability & Tracing"],
)
def prometheus_metrics(request):
    """Returns real-time gateway metrics in Prometheus exposition format."""
    return HttpResponse(
        get_prometheus_metrics_text(),
        content_type=CONTENT_TYPE_LATEST,
    )


@api.get(
    "/observability/summary",
    auth=[api_key_auth, api_key_header_auth],
    response=ObservabilitySummaryOut,
    summary="Get Observability & Latency Percentiles Summary",
    description=(
        "Returns aggregated request metrics, token breakdown, total FinOps cost, and "
        "empirical p50, p90, p95, p99 latency quantiles calculated from sliding window observations."
    ),
    tags=["Observability & Tracing"],
)
async def get_observability_summary(request) -> ObservabilitySummaryOut:
    """Returns tenant observability summary with empirical latency percentiles."""
    tenant: Tenant = request.auth
    tenant_id_str = str(tenant.id)

    # 1. Aggregate metrics from audit logs
    total_requests = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_cost = Decimal("0.000000")

    async for log in RequestAuditLog.objects.filter(tenant=tenant):
        total_requests += 1
        prompt_tokens += log.prompt_tokens
        completion_tokens += log.completion_tokens
        total_cost += log.cost_usd

    # 2. Retrieve empirical latency percentiles
    percentiles_data = await LatencyTracker.get_percentiles(tenant_id_str)

    # Fallback to recent audit logs if sliding window buffer has no samples
    if percentiles_data["samples_count"] == 0 and total_requests > 0:
        log_latencies = []
        async for log in RequestAuditLog.objects.filter(tenant=tenant).order_by("-created_at")[:100]:
            log_latencies.append(log.latency_ms)
        if log_latencies:
            from gateway.observability.latency import calculate_percentile

            sorted_logs = sorted(log_latencies)
            percentiles_data = {
                "p50_ms": calculate_percentile(sorted_logs, 50.0),
                "p90_ms": calculate_percentile(sorted_logs, 90.0),
                "p95_ms": calculate_percentile(sorted_logs, 95.0),
                "p99_ms": calculate_percentile(sorted_logs, 99.0),
                "samples_count": len(sorted_logs),
            }

    return ObservabilitySummaryOut(
        tenant_id=tenant.id,
        total_requests=total_requests,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_cost_usd=total_cost,
        latency_percentiles=LatencyPercentilesOut(**percentiles_data),
    )


# ---------------------------------------------------------------------------
# Benchmark Harness & Baseline Evaluation (Phase 12)
# ---------------------------------------------------------------------------


@api.post(
    "/benchmarks/run",
    auth=[api_key_auth, api_key_header_auth],
    response=BenchmarkComparisonOut,
    summary="Execute Baseline vs. Routed Gateway Benchmark",
    description=(
        "Executes a curated 40-item benchmark suite comparing All-Premium baseline "
        "against Intelligent Routed Gateway. Measures total spend reduction, "
        "accuracy retention, and latency improvements across diverse task complexities."
    ),
    tags=["Benchmark Harness & Evaluation"],
)
async def run_benchmark(request, payload: Optional[BenchmarkRunIn] = None) -> BenchmarkComparisonOut:
    """Executes the benchmark harness comparing All-Premium vs. Routed Gateway."""
    tenant: Tenant = request.auth
    opts = payload or BenchmarkRunIn()
    result = await BenchmarkRunner.run_benchmark(
        sample_size=opts.sample_size,
        categories=opts.categories,
        tenant=tenant,
        mode=opts.mode,
        premium_model=opts.premium_model,
    )
    return BenchmarkComparisonOut(**result)


# ---------------------------------------------------------------------------
# Security Guardrails & Moderation Inspection (Phase 13)
# ---------------------------------------------------------------------------


@api.post(
    "/security/scan",
    auth=[api_key_auth, api_key_header_auth],
    response=SecurityScanOut,
    summary="Diagnostic Security & Guardrail Inspection",
    description=(
        "Scans arbitrary prompt text for sensitive PII entities (credit cards, SSNs, "
        "emails, phone numbers, secret keys), prompt injection/jailbreak attempts, "
        "and content moderation policy violations."
    ),
    tags=["Security & Guardrails"],
)
async def scan_security(request, payload: SecurityScanIn) -> SecurityScanOut:
    """Standalone security scan for pre-flight vetting."""
    res = SecurityGuardrailManager.scan_text(
        text=payload.text,
        mask_pii=payload.mask_pii,
        check_injection=payload.check_injection,
        check_moderation=payload.check_moderation,
    )
    return SecurityScanOut(
        is_safe=res["is_safe"],
        violations=res["violations"],
        detected_pii=res["detected_pii"],
        pii_count=res["pii_count"],
        masked_text=res["masked_text"],
        injection_risk=res["injection_risk"],
        injection_score=res["injection_score"],
        execution_time_ms=res["execution_time_ms"],
    )




