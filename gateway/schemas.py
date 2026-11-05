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

