"""
gateway/ratelimit/budget.py
===========================
FinOps Hard & Soft Budget Quota Management.

Enforces spending ceilings with soft warning thresholds (80%, 90%)
and hard request rejections (100%) to prevent unexpected cloud billing spikes.
"""
from dataclasses import dataclass
from decimal import Decimal
import logging
from typing import Optional
from uuid import UUID

from django.db.models import F

from gateway.models import Tenant

logger = logging.getLogger(__name__)


@dataclass
class BudgetEvaluation:
    """Evaluation result of a tenant's FinOps budget standing."""

    allowed: bool
    monthly_budget_usd: Decimal
    current_spend_usd: Decimal
    remaining_usd: Decimal
    percentage_consumed: float
    warning_header: Optional[str] = None


class FinOpsBudgetManager:
    """Manages soft and hard budget limits for tenants."""

    SOFT_ALERT_THRESHOLD = 80.0  # Alert when >= 80% consumed
    CRITICAL_ALERT_THRESHOLD = 90.0  # Critical alert when >= 90% consumed

    @classmethod
    def evaluate_budget(cls, tenant: Tenant, estimated_cost: Decimal = Decimal("0.000000")) -> BudgetEvaluation:
        """
        Evaluates whether the tenant has budget remaining to proceed with a request.
        Calculates percentage consumed and generates soft threshold alert headers.
        """
        # Unlimited budget if configured as 0.00
        if tenant.monthly_budget_usd <= Decimal("0.000000"):
            return BudgetEvaluation(
                allowed=True,
                monthly_budget_usd=tenant.monthly_budget_usd,
                current_spend_usd=tenant.current_spend_usd,
                remaining_usd=Decimal("999999.000000"),
                percentage_consumed=0.0,
                warning_header=None,
            )

        new_total = tenant.current_spend_usd + estimated_cost
        remaining = max(Decimal("0.000000"), tenant.monthly_budget_usd - tenant.current_spend_usd)

        # Hard Lockout: Current spend has reached or exceeded cap
        if tenant.current_spend_usd >= tenant.monthly_budget_usd:
            logger.warning(
                "Tenant '%s' hard budget lockout triggered: spend $%s >= cap $%s",
                tenant.name,
                tenant.current_spend_usd,
                tenant.monthly_budget_usd,
            )
            return BudgetEvaluation(
                allowed=False,
                monthly_budget_usd=tenant.monthly_budget_usd,
                current_spend_usd=tenant.current_spend_usd,
                remaining_usd=remaining,
                percentage_consumed=100.0,
                warning_header=f"CRITICAL: Budget exhausted (${tenant.current_spend_usd} / ${tenant.monthly_budget_usd}).",
            )

        percentage = float((tenant.current_spend_usd / tenant.monthly_budget_usd) * 100)
        warning_header = None

        if percentage >= cls.CRITICAL_ALERT_THRESHOLD:
            warning_header = (
                f"CRITICAL: Tenant has consumed {percentage:.1f}% of monthly budget "
                f"(${tenant.current_spend_usd} / ${tenant.monthly_budget_usd})"
            )
        elif percentage >= cls.SOFT_ALERT_THRESHOLD:
            warning_header = (
                f"WARNING: Tenant has consumed {percentage:.1f}% of monthly budget "
                f"(${tenant.current_spend_usd} / ${tenant.monthly_budget_usd})"
            )

        return BudgetEvaluation(
            allowed=True,
            monthly_budget_usd=tenant.monthly_budget_usd,
            current_spend_usd=tenant.current_spend_usd,
            remaining_usd=remaining,
            percentage_consumed=round(percentage, 2),
            warning_header=warning_header,
        )

    @classmethod
    async def top_up_budget(cls, tenant_id: UUID, additional_usd: Decimal) -> Tenant:
        """Adds additional USD funds to a tenant's monthly budget cap."""
        await Tenant.objects.filter(pk=tenant_id).aupdate(
            monthly_budget_usd=F("monthly_budget_usd") + additional_usd
        )
        return await Tenant.objects.aget(pk=tenant_id)

    @classmethod
    async def reset_spend(cls, tenant_id: UUID) -> Tenant:
        """Resets the tenant's current spend to zero (e.g. at start of billing cycle)."""
        await Tenant.objects.filter(pk=tenant_id).aupdate(current_spend_usd=Decimal("0.000000"))
        return await Tenant.objects.aget(pk=tenant_id)
