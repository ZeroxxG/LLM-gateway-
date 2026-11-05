"""
gateway/models.py
=================
Multi-Tenant Data Models, API Key Store, and Model Pricing Catalog.

Entities:
  - Tenant: Multi-tenant boundary representing an organization or customer.
    Tracks budget, usage, and rate limits.
  - APIKey: Cryptographically hashed access tokens tied to Tenants.
    The raw secret token is never stored in plaintext (SHA-256 hashed).
  - ModelPricing: Dynamic per-million-token cost catalog used by the FinOps
    engine to calculate input/output expenses in real-time.
"""
import uuid
from decimal import Decimal
from typing import Optional

from django.db import models
from django.utils import timezone


class TenantTier(models.TextChoices):
    """Tier levels dictating rate limits, priority, and feature flags."""

    FREE = "free", "Free"
    STANDARD = "standard", "Standard"
    ENTERPRISE = "enterprise", "Enterprise"


class ModelTier(models.TextChoices):
    """Model categorization for intelligent cost-performance routing."""

    SMALL = "small", "Small (Fast / Low Cost)"
    MEDIUM = "medium", "Medium (Balanced)"
    PREMIUM = "premium", "Premium (High Intelligence / High Cost)"


class Tenant(models.Model):
    """
    Tenant entity representing an isolated billing and administrative domain.

    Every inbound LLM request is authenticated against an API Key belonging to a
    Tenant. The Tenant's budget and rate limits are enforced prior to routing.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Globally unique tenant identifier (UUIDv4).",
    )
    name = models.CharField(
        max_length=255,
        db_index=True,
        help_text="Human-readable company or project name.",
    )
    tier = models.CharField(
        max_length=50,
        choices=TenantTier.choices,
        default=TenantTier.FREE,
        db_index=True,
        help_text="Tenant subscription tier.",
    )
    monthly_budget_usd = models.DecimalField(
        max_digits=12,
        decimal_places=6,
        default=Decimal("0.000000"),
        help_text="Configured monthly spending cap in USD.",
    )
    current_spend_usd = models.DecimalField(
        max_digits=12,
        decimal_places=6,
        default=Decimal("0.000000"),
        help_text="Accumulated spending for the current billing cycle in USD.",
    )
    rpm_limit = models.IntegerField(
        default=60,
        help_text="Requests Per Minute allowed across all keys for this tenant.",
    )
    tpm_limit = models.IntegerField(
        default=40000,
        help_text="Tokens Per Minute allowed across all keys for this tenant.",
    )
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        help_text="Soft-deletion/suspension flag. If False, all tenant keys are rejected.",
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        editable=False,
        help_text="Timestamp of tenant creation.",
    )

    class Meta:
        db_table = "gateway_tenants"
        ordering = ["-created_at"]
        verbose_name = "Tenant"
        verbose_name_plural = "Tenants"

    def __str__(self) -> str:
        return f"{self.name} ({self.tier}) - {self.id}"

    @property
    def remaining_budget_usd(self) -> Decimal:
        """Calculate the remaining budget for the billing cycle."""
        return max(Decimal("0.000000"), self.monthly_budget_usd - self.current_spend_usd)

    def has_budget_remaining(self, estimated_cost: Decimal = Decimal("0.000000")) -> bool:
        """
        Check if the tenant has sufficient budget for an operation.
        If monthly_budget_usd is 0, it indicates an unlimited or unset budget.
        """
        if self.monthly_budget_usd <= Decimal("0.000000"):
            return True
        return (self.current_spend_usd + estimated_cost) < self.monthly_budget_usd


class APIKey(models.Model):
    """
    Hashed API Key credentials.

    Security Guarantee:
      The plaintext raw key (`gw_live_<32_hex_chars>`) is presented to the user
      EXACTLY ONCE upon creation. Only the SHA-256 hash is persisted in `hashed_key`.
      The `key_prefix` (first 8 chars, e.g., 'gw_live_') is stored in plaintext
      solely for administrative identification and logging.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Globally unique API key ID.",
    )
    tenant = models.ForeignKey(
        Tenant,
        on_delete=models.CASCADE,
        related_name="api_keys",
        help_text="Tenant that owns this API key.",
    )
    key_prefix = models.CharField(
        max_length=8,
        help_text="Visible key prefix (e.g., 'gw_live_') for identification without revealing secrets.",
    )
    hashed_key = models.CharField(
        max_length=128,
        unique=True,
        db_index=True,
        help_text="SHA-256 hex digest of the raw secret token.",
    )
    name = models.CharField(
        max_length=100,
        default="Default Key",
        help_text="Friendly label for this key (e.g. 'Production Backend', 'Staging Worker').",
    )
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        help_text="Revocation flag. If False, requests using this key will return 401 Unauthorized.",
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        editable=False,
        help_text="Timestamp of key generation.",
    )
    last_used_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp when this key was last successfully authenticated.",
    )

    class Meta:
        db_table = "gateway_api_keys"
        ordering = ["-created_at"]
        verbose_name = "API Key"
        verbose_name_plural = "API Keys"

    def __str__(self) -> str:
        return f"{self.name} [{self.key_prefix}...] ({self.tenant.name})"


class ModelPricing(models.Model):
    """
    Catalog of supported upstream LLM models and per-token pricing.

    FinOps Calculation:
      total_cost = (prompt_tokens * input_cost_per_million / 1_000_000) +
                   (completion_tokens * output_cost_per_million / 1_000_000)
    """

    model_id = models.CharField(
        max_length=100,
        unique=True,
        db_index=True,
        help_text="Canonical model identifier, e.g. 'groq/llama-3.1-8b', 'openai/gpt-4o-mini'.",
    )
    provider = models.CharField(
        max_length=50,
        db_index=True,
        help_text="Underlying provider: 'groq', 'openai', 'anthropic', 'gemini'.",
    )
    input_cost_per_million = models.DecimalField(
        max_digits=8,
        decimal_places=4,
        help_text="Cost in USD per 1,000,000 input (prompt) tokens.",
    )
    output_cost_per_million = models.DecimalField(
        max_digits=8,
        decimal_places=4,
        help_text="Cost in USD per 1,000,000 output (completion) tokens.",
    )
    tier = models.CharField(
        max_length=50,
        choices=ModelTier.choices,
        default=ModelTier.SMALL,
        db_index=True,
        help_text="Model performance/cost classification.",
    )
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        help_text="Whether this model is currently enabled for routing.",
    )

    class Meta:
        db_table = "gateway_model_pricing"
        ordering = ["provider", "model_id"]
        verbose_name = "Model Pricing"
        verbose_name_plural = "Model Pricing Catalog"

    def __str__(self) -> str:
        return f"{self.model_id} (${self.input_cost_per_million}/${self.output_cost_per_million} per M)"

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> Decimal:
        """
        Calculate total cost in USD for a given token usage breakdown.
        Accurate to 8 decimal places for precise FinOps accounting.
        """
        one_million = Decimal("1000000")
        prompt_cost = (Decimal(str(prompt_tokens)) * self.input_cost_per_million) / one_million
        completion_cost = (Decimal(str(completion_tokens)) * self.output_cost_per_million) / one_million
        return (prompt_cost + completion_cost).quantize(Decimal("0.00000001"))


class RequestAuditLog(models.Model):
    """
    Immutable audit log entry for every inbound LLM completion request.
    Recorded asynchronously by Celery workers to keep ASGI execution latency minimal.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Globally unique audit log entry ID.",
    )
    tenant = models.ForeignKey(
        Tenant,
        on_delete=models.CASCADE,
        related_name="audit_logs",
        db_index=True,
        help_text="Tenant that executed this request.",
    )
    request_id = models.CharField(
        max_length=64,
        db_index=True,
        help_text="Unique request identifier generated for tracing.",
    )
    model_id = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Model that executed the completion (e.g. 'groq/llama-3.1-8b').",
    )
    provider = models.CharField(
        max_length=50,
        db_index=True,
        help_text="Underlying provider (e.g. 'groq', 'openai', 'gemini', 'mock').",
    )
    prompt_tokens = models.PositiveIntegerField(
        default=0,
        help_text="Count of prompt tokens.",
    )
    completion_tokens = models.PositiveIntegerField(
        default=0,
        help_text="Count of generated completion tokens.",
    )
    total_tokens = models.PositiveIntegerField(
        default=0,
        help_text="Total tokens consumed.",
    )
    cost_usd = models.DecimalField(
        max_digits=12,
        decimal_places=8,
        default=Decimal("0.00000000"),
        help_text="Exact calculated dollar cost for this request.",
    )
    latency_ms = models.FloatField(
        default=0.0,
        help_text="Round-trip provider execution latency in milliseconds.",
    )
    status_code = models.IntegerField(
        default=200,
        db_index=True,
        help_text="HTTP status returned to client (e.g. 200, 429, 502).",
    )
    error_message = models.TextField(
        null=True,
        blank=True,
        help_text="Error message if request failed or experienced failover.",
    )
    routing_strategy = models.CharField(
        max_length=50,
        default="direct",
        help_text="Dynamic routing strategy applied (e.g. 'cost_optimized', 'direct').",
    )
    fallback_occurred = models.BooleanField(
        default=False,
        help_text="Whether fallback routing took place due to primary failure.",
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="Timestamp of request completion.",
    )

    class Meta:
        db_table = "gateway_request_audit_logs"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["tenant", "-created_at"]),
            models.Index(fields=["tenant", "status_code"]),
        ]
        verbose_name = "Request Audit Log"
        verbose_name_plural = "Request Audit Logs"

    def __str__(self) -> str:
        return f"[{self.status_code}] {self.model_id} ({self.total_tokens} tokens) - {self.tenant.name}"


class DailyUsageAggregate(models.Model):
    """
    Rollup aggregation of usage metrics per tenant per UTC day.
    Generated by periodic Celery background tasks for fast dashboard analytics.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Globally unique aggregate entry ID.",
    )
    tenant = models.ForeignKey(
        Tenant,
        on_delete=models.CASCADE,
        related_name="daily_aggregates",
        db_index=True,
        help_text="Tenant associated with this daily rollup.",
    )
    date = models.DateField(
        db_index=True,
        help_text="UTC calendar date for this aggregate bucket.",
    )
    total_requests = models.PositiveIntegerField(
        default=0,
        help_text="Total number of requests initiated on this day.",
    )
    successful_requests = models.PositiveIntegerField(
        default=0,
        help_text="Count of successful (HTTP 200) requests.",
    )
    failed_requests = models.PositiveIntegerField(
        default=0,
        help_text="Count of failed (HTTP 4xx/5xx) requests.",
    )
    total_prompt_tokens = models.BigIntegerField(
        default=0,
        help_text="Cumulative prompt tokens consumed.",
    )
    total_completion_tokens = models.BigIntegerField(
        default=0,
        help_text="Cumulative completion tokens consumed.",
    )
    total_tokens = models.BigIntegerField(
        default=0,
        help_text="Cumulative total tokens consumed.",
    )
    total_cost_usd = models.DecimalField(
        max_digits=12,
        decimal_places=6,
        default=Decimal("0.000000"),
        help_text="Total FinOps spend in USD accumulated on this day.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "gateway_daily_usage_aggregates"
        unique_together = ("tenant", "date")
        ordering = ["-date"]
        verbose_name = "Daily Usage Aggregate"
        verbose_name_plural = "Daily Usage Aggregates"

    def __str__(self) -> str:
        return f"{self.tenant.name} - {self.date} ({self.total_requests} reqs, ${self.total_cost_usd})"

