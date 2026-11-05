"""
gateway/schemas.py
==================
Pydantic v2 schemas for Tenant, Authentication, and Model Pricing.

Enforces strict type safety and auto-generates OpenAPI documentation.
"""
from datetime import date as DateType, datetime as DateTimeType
from decimal import Decimal
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from gateway.adapters.schemas import ChatMessage


class TenantOut(BaseModel):
    """Authenticated Tenant response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Unique Tenant ID (UUIDv4).")
    name: str = Field(..., description="Tenant organization name.")
    tier: str = Field(..., description="Tenant tier (free, standard, enterprise).")
    monthly_budget_usd: Decimal = Field(..., description="Monthly spending limit in USD.")
    current_spend_usd: Decimal = Field(..., description="Accumulated spending in current billing cycle.")
    remaining_budget_usd: Decimal = Field(..., description="Remaining spending budget for this cycle.")
    rpm_limit: int = Field(..., description="Requests per minute limit.")
    tpm_limit: int = Field(..., description="Tokens per minute limit.")
    is_active: bool = Field(..., description="Account active status.")


class ModelPricingOut(BaseModel):
    """Model Pricing Catalog item schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    model_id: str = Field(..., description="Canonical model identifier, e.g. 'groq/llama-3.1-8b'.")
    provider: str = Field(..., description="Provider name: groq, openai, anthropic, gemini.")
    input_cost_per_million: Decimal = Field(..., description="Price in USD per 1M prompt tokens.")
    output_cost_per_million: Decimal = Field(..., description="Price in USD per 1M completion tokens.")
    tier: str = Field(..., description="Model tier: small, medium, premium.")
    is_active: bool = Field(..., description="Availability status.")


class AuthVerifyOut(BaseModel):
    """API Key verification test response."""

    authenticated: bool = Field(True, description="Authentication confirmation.")
    tenant: TenantOut = Field(..., description="Authenticated tenant information.")


class CircuitStatusOut(BaseModel):
    """Diagnostic status for an upstream provider circuit breaker."""

    provider: str = Field(..., description="Provider name (groq, openai, gemini, etc.)")
    state: str = Field(..., description="Circuit state: CLOSED, OPEN, HALF_OPEN")
    consecutive_failures: int = Field(..., description="Current consecutive failure count.")
    failure_threshold: int = Field(..., description="Threshold that trips circuit to OPEN.")
    last_failure_timestamp: Optional[float] = Field(None, description="Unix timestamp of most recent failure.")
    recovery_timeout_seconds: float = Field(..., description="Cooldown duration before attempting HALF_OPEN probe.")


class BudgetTopUpIn(BaseModel):
    """Input payload to adjust or top up tenant monthly budget cap."""

    additional_budget_usd: Decimal = Field(..., gt=Decimal("0.00"), description="Additional budget in USD to add.")


class RealtimeBudgetStatusOut(BaseModel):
    """Real-time budget standing from Redis."""

    tenant_id: UUID = Field(..., description="Tenant unique identifier.")
    monthly_budget_usd: Decimal = Field(..., description="Monthly spending limit in USD.")
    current_spend_usd: Decimal = Field(..., description="Accumulated spend in Redis.")
    remaining_budget_usd: Decimal = Field(..., description="Remaining budget in USD.")
    percentage_consumed: float = Field(..., description="Percentage of monthly budget consumed.")
    is_locked_out: bool = Field(..., description="Whether tenant is locked out due to budget limit.")


class BudgetSyncOut(BaseModel):
    """Result of syncing Redis spend to database."""

    tenant_id: UUID = Field(..., description="Tenant unique identifier.")
    synced_spend_usd: Decimal = Field(..., description="Spend value synced to PostgreSQL.")
    status: str = Field("synced", description="Sync status.")


class RateLimitStatusOut(BaseModel):
    """Current rate limit status and remaining quota for tenant."""

    rpm_limit: int = Field(..., description="Requests per minute limit.")
    rpm_remaining: int = Field(..., description="Remaining requests allowed in current window.")
    tpm_limit: int = Field(..., description="Tokens per minute limit.")
    tpm_remaining: int = Field(..., description="Remaining tokens allowed in current window.")
    window_reset_seconds: int = Field(..., description="Seconds until window counter resets.")


class RequestAuditLogOut(BaseModel):
    """Granular execution audit log record schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID = Field(..., description="Audit log entry unique identifier.")
    request_id: str = Field(..., description="Traced request ID.")
    model_id: str = Field(..., description="Target model executed.")
    provider: str = Field(..., description="Target provider.")
    prompt_tokens: int = Field(..., description="Prompt tokens consumed.")
    completion_tokens: int = Field(..., description="Generated completion tokens.")
    total_tokens: int = Field(..., description="Total tokens.")
    cost_usd: Decimal = Field(..., description="Calculated FinOps cost.")
    latency_ms: float = Field(..., description="Execution latency in milliseconds.")
    status_code: int = Field(..., description="HTTP status code returned.")
    error_message: Optional[str] = Field(None, description="Error message if failed or retried.")
    routing_strategy: str = Field(..., description="Strategy used to route request.")
    fallback_occurred: bool = Field(..., description="Whether fallback routing took place.")
    created_at: DateTimeType = Field(..., description="Timestamp of request completion.")


class DailyUsageAggregateOut(BaseModel):
    """Aggregated usage metrics for a specific calendar day."""

    model_config = ConfigDict(from_attributes=True)

    date: DateType = Field(..., description="Calendar date of aggregate.")
    total_requests: int = Field(..., description="Total requests processed on this day.")
    successful_requests: int = Field(..., description="Successful (200 OK) requests.")
    failed_requests: int = Field(..., description="Failed requests.")
    total_prompt_tokens: int = Field(..., description="Sum of prompt tokens.")
    total_completion_tokens: int = Field(..., description="Sum of completion tokens.")
    total_tokens: int = Field(..., description="Total tokens consumed.")
    total_cost_usd: Decimal = Field(..., description="Total FinOps cost in USD.")


class TenantUsageAnalyticsOut(BaseModel):
    """Comprehensive analytics summary for the authenticated tenant."""

    tenant_id: UUID = Field(..., description="Tenant ID.")
    tenant_name: str = Field(..., description="Tenant name.")
    total_requests: int = Field(..., description="Total requests across the selected period.")
    total_tokens: int = Field(..., description="Total tokens consumed across the period.")
    total_cost_usd: Decimal = Field(..., description="Total cost incurred across the period.")
    daily_breakdown: List[DailyUsageAggregateOut] = Field(..., description="Day-by-day aggregate breakdown.")


class AsyncTaskTriggerOut(BaseModel):
    """Response returned when an async Celery background task is queued."""

    task_id: str = Field(..., description="Unique Celery Task UUID.")
    status: str = Field(..., description="Initial task status (e.g. PENDING).")
    message: str = Field(..., description="Human-readable description of scheduled work.")


class AsyncTaskStatusOut(BaseModel):
    """Status and result of an async Celery background task."""

    task_id: str = Field(..., description="Unique Celery Task UUID.")
    status: str = Field(..., description="Current Celery status (PENDING, STARTED, SUCCESS, FAILURE).")
    ready: bool = Field(..., description="Whether task execution has finished.")
    successful: bool = Field(..., description="Whether task execution succeeded without unhandled errors.")
    result: Optional[Any] = Field(None, description="Task return value or error detail.")


class CandidateScoreOut(BaseModel):
    """Scored evaluation result for a single model candidate."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(..., description="Canonical model identifier.")
    provider: str = Field(..., description="Upstream provider.")
    score: float = Field(..., description="Composite heuristic score (0 to 100).")
    capability_score: float = Field(..., description="Capability match score for task type.")
    latency_score: float = Field(..., description="Latency score.")
    cost_score: float = Field(..., description="Cost efficiency score.")
    estimated_cost_usd: Decimal = Field(..., description="Estimated cost in USD.")
    typical_latency_ms: float = Field(..., description="Benchmark typical latency in ms.")
    eligible: bool = Field(..., description="Whether candidate satisfies all constraints.")
    rejection_reason: Optional[str] = Field(None, description="Reason candidate was rejected or pruned.")


class RoutingEvaluationIn(BaseModel):
    """Request payload to test or dry-run heuristic routing."""

    messages: List[ChatMessage] = Field(..., min_length=1, description="Sample conversation messages.")
    task_type: Optional[str] = Field(None, description="Explicit task type override.")
    latency_sla_ms: Optional[float] = Field(None, gt=0.0, description="Latency SLA threshold in ms.")
    max_cost_usd: Optional[Decimal] = Field(None, gt=Decimal("0.00"), description="Budget limit in USD.")
    strategy: Optional[str] = Field("heuristic", description="Routing strategy name.")


class RoutingEvaluationOut(BaseModel):
    """Diagnostic response from the heuristic routing engine."""

    detected_task_type: str = Field(..., description="Detected or specified task type.")
    estimated_tokens: int = Field(..., description="Estimated prompt token count.")
    decision_latency_ms: float = Field(..., description="Decision engine execution time in milliseconds.")
    selected_model: str = Field(..., description="Top recommended model.")
    candidate_chain: List[str] = Field(..., description="Ranked list of eligible models.")
    candidates: List[CandidateScoreOut] = Field(..., description="Detailed candidate evaluations.")
    strategy_used: str = Field("heuristic", description="Routing policy used.")


