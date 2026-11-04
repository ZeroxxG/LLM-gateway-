"""
gateway/schemas.py
==================
Pydantic v2 schemas for Tenant, Authentication, and Model Pricing.

Enforces strict type safety and auto-generates OpenAPI documentation.
"""
from decimal import Decimal
from typing import List, Optional
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
