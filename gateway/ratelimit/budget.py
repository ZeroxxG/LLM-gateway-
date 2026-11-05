"""
gateway/ratelimit/budget.py
===========================
Real-time budget tracking in Redis with INCRBYFLOAT and 402 hard-stop rejection.
"""
from dataclasses import dataclass
from decimal import Decimal
import logging
from typing import Optional, Tuple
from uuid import UUID

from django.db.models import F

from gateway.models import Tenant
from gateway.utils.redis import get_redis_client

logger = logging.getLogger(__name__)

# Atomic Redis Lua script to check budget and initialize cache if needed
CHECK_BUDGET_LUA = """
local spend_key = KEYS[1]
local budget_key = KEYS[2]
local lockout_key = KEYS[3]

local default_budget = tonumber(ARGV[1])
local default_spend = tonumber(ARGV[2])

local spend_val = redis.call('GET', spend_key)
if not spend_val then
    redis.call('SET', spend_key, default_spend)
    spend_val = default_spend
else
    spend_val = tonumber(spend_val)
end

local budget_val = redis.call('GET', budget_key)
if not budget_val then
    redis.call('SET', budget_key, default_budget)
    budget_val = default_budget
else
    budget_val = tonumber(budget_val)
end

if budget_val > 0 and spend_val >= budget_val then
    redis.call('SET', lockout_key, '1')
    return {0, tostring(spend_val), tostring(budget_val)}
end

return {1, tostring(spend_val), tostring(budget_val)}
"""

# Atomic Redis Lua script to increment spend using INCRBYFLOAT and enforce lockout
RECORD_SPEND_LUA = """
local spend_key = KEYS[1]
local budget_key = KEYS[2]
local lockout_key = KEYS[3]

local cost = tonumber(ARGV[1])
local default_budget = tonumber(ARGV[2])

local new_spend = redis.call('INCRBYFLOAT', spend_key, cost)
local new_spend_num = tonumber(new_spend)

local budget_val = redis.call('GET', budget_key)
local budget_num = budget_val and tonumber(budget_val) or default_budget

local is_exhausted = 0
if budget_num > 0 and new_spend_num >= budget_num then
    redis.call('SET', lockout_key, '1')
    is_exhausted = 1
else
    redis.call('DEL', lockout_key)
end

return {tostring(new_spend), tostring(budget_num), is_exhausted}
"""

# Atomic Redis Lua script to update budget cap on top-up and clear lockout
TOP_UP_LUA = """
local budget_key = KEYS[1]
local spend_key = KEYS[2]
local lockout_key = KEYS[3]

local additional_budget = tonumber(ARGV[1])
local default_budget = tonumber(ARGV[2])

local cur_budget = redis.call('GET', budget_key)
local cur_budget_num = cur_budget and tonumber(cur_budget) or default_budget
local new_budget = cur_budget_num + additional_budget
redis.call('SET', budget_key, new_budget)

local cur_spend = redis.call('GET', spend_key)
local cur_spend_num = cur_spend and tonumber(cur_spend) or 0

if new_budget > cur_spend_num then
    redis.call('DEL', lockout_key)
end

return {tostring(new_budget), tostring(cur_spend_num)}
"""


@dataclass
class BudgetEvaluation:
    """Result of checking tenant budget standing."""

    allowed: bool
    monthly_budget_usd: Decimal
    current_spend_usd: Decimal
    remaining_usd: Decimal
    percentage_consumed: float
    warning_header: Optional[str] = None


class FinOpsBudgetManager:
    """Manages real-time tenant budgets using Redis and Postgres."""

    SOFT_ALERT_THRESHOLD = 80.0  # soft alert at 80%
    CRITICAL_ALERT_THRESHOLD = 90.0  # critical alert at 90%

    @classmethod
    def _spend_key(cls, tenant_id: str) -> str:
        return f"finops:spend:{tenant_id}"

    @classmethod
    def _budget_key(cls, tenant_id: str) -> str:
        return f"finops:budget:{tenant_id}"

    @classmethod
    def _lockout_key(cls, tenant_id: str) -> str:
        return f"finops:lockout:{tenant_id}"

    @classmethod
    def evaluate_budget(cls, tenant: Tenant, estimated_cost: Decimal = Decimal("0.000000")) -> BudgetEvaluation:
        """Synchronous budget evaluation from tenant instance (used as fallback)."""
        if tenant.monthly_budget_usd <= Decimal("0.000000"):
            return BudgetEvaluation(
                allowed=True,
                monthly_budget_usd=tenant.monthly_budget_usd,
                current_spend_usd=tenant.current_spend_usd,
                remaining_usd=Decimal("999999.000000"),
                percentage_consumed=0.0,
                warning_header=None,
            )

        remaining = max(Decimal("0.000000"), tenant.monthly_budget_usd - tenant.current_spend_usd)

        if tenant.current_spend_usd >= tenant.monthly_budget_usd:
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
    async def evaluate_budget_realtime(
        cls, tenant: Tenant, estimated_cost: Decimal = Decimal("0.000000")
    ) -> BudgetEvaluation:
        """Check budget in Redis in real-time without database calls."""
        if tenant.monthly_budget_usd <= Decimal("0.000000"):
            return BudgetEvaluation(
                allowed=True,
                monthly_budget_usd=tenant.monthly_budget_usd,
                current_spend_usd=tenant.current_spend_usd,
                remaining_usd=Decimal("999999.000000"),
                percentage_consumed=0.0,
                warning_header=None,
            )

        redis = await get_redis_client()
        tenant_id_str = str(tenant.id)

        spend_key = cls._spend_key(tenant_id_str)
        budget_key = cls._budget_key(tenant_id_str)
        lockout_key = cls._lockout_key(tenant_id_str)

        res = await redis.eval(
            CHECK_BUDGET_LUA,
            3,
            spend_key,
            budget_key,
            lockout_key,
            str(tenant.monthly_budget_usd),
            str(tenant.current_spend_usd),
        )

        allowed_flag = int(res[0])
        spend_val = Decimal(res[1].decode("utf-8") if isinstance(res[1], bytes) else str(res[1]))
        budget_val = Decimal(res[2].decode("utf-8") if isinstance(res[2], bytes) else str(res[2]))

        remaining = max(Decimal("0.000000"), budget_val - spend_val)

        if allowed_flag == 0 or spend_val >= budget_val:
            logger.warning(
                "Tenant '%s' hard budget lockout: spend $%s >= cap $%s",
                tenant.name,
                spend_val,
                budget_val,
            )
            return BudgetEvaluation(
                allowed=False,
                monthly_budget_usd=budget_val,
                current_spend_usd=spend_val,
                remaining_usd=remaining,
                percentage_consumed=100.0,
                warning_header=f"CRITICAL: Budget exhausted (${spend_val} / ${budget_val}).",
            )

        percentage = float((spend_val / budget_val) * 100) if budget_val > 0 else 0.0
        warning_header = None
        if percentage >= cls.CRITICAL_ALERT_THRESHOLD:
            warning_header = (
                f"CRITICAL: Tenant has consumed {percentage:.1f}% of monthly budget "
                f"(${spend_val} / ${budget_val})"
            )
        elif percentage >= cls.SOFT_ALERT_THRESHOLD:
            warning_header = (
                f"WARNING: Tenant has consumed {percentage:.1f}% of monthly budget "
                f"(${spend_val} / ${budget_val})"
            )

        return BudgetEvaluation(
            allowed=True,
            monthly_budget_usd=budget_val,
            current_spend_usd=spend_val,
            remaining_usd=remaining,
            percentage_consumed=round(percentage, 2),
            warning_header=warning_header,
        )

    @classmethod
    async def record_spend_atomic(
        cls,
        tenant_id: UUID,
        cost: Decimal,
        default_budget: Decimal = Decimal("0.000000"),
    ) -> Tuple[Decimal, Decimal, bool]:
        """Atomically increment spend using Redis INCRBYFLOAT and check lockout."""
        if cost <= Decimal("0.00000000"):
            return Decimal("0.000000"), default_budget, False

        redis = await get_redis_client()
        tenant_id_str = str(tenant_id)

        spend_key = cls._spend_key(tenant_id_str)
        budget_key = cls._budget_key(tenant_id_str)
        lockout_key = cls._lockout_key(tenant_id_str)

        res = await redis.eval(
            RECORD_SPEND_LUA,
            3,
            spend_key,
            budget_key,
            lockout_key,
            str(cost),
            str(default_budget),
        )

        new_spend = Decimal(res[0].decode("utf-8") if isinstance(res[0], bytes) else str(res[0]))
        budget_num = Decimal(res[1].decode("utf-8") if isinstance(res[1], bytes) else str(res[1]))
        is_exhausted = (int(res[2]) == 1)

        return new_spend, budget_num, is_exhausted

    @classmethod
    async def get_realtime_budget(cls, tenant_id: UUID) -> dict:
        """Read current spend and budget from Redis."""
        redis = await get_redis_client()
        tenant_id_str = str(tenant_id)

        spend_raw = await redis.get(cls._spend_key(tenant_id_str))
        budget_raw = await redis.get(cls._budget_key(tenant_id_str))
        lockout_raw = await redis.get(cls._lockout_key(tenant_id_str))

        spend = Decimal(spend_raw.decode("utf-8") if isinstance(spend_raw, bytes) else (spend_raw or "0.000000"))
        budget = Decimal(budget_raw.decode("utf-8") if isinstance(budget_raw, bytes) else (budget_raw or "0.000000"))
        is_locked = (lockout_raw is not None)

        remaining = max(Decimal("0.000000"), budget - spend)
        pct = float((spend / budget) * 100) if budget > 0 else 0.0

        return {
            "monthly_budget_usd": budget,
            "current_spend_usd": spend,
            "remaining_budget_usd": remaining,
            "percentage_consumed": round(pct, 2),
            "is_locked_out": is_locked or (budget > 0 and spend >= budget),
        }

    @classmethod
    async def sync_spend_to_db(cls, tenant_id: UUID) -> Tenant:
        """Sync Redis spend back to PostgreSQL database."""
        redis = await get_redis_client()
        tenant_id_str = str(tenant_id)
        spend_raw = await redis.get(cls._spend_key(tenant_id_str))

        if spend_raw is not None:
            spend = Decimal(spend_raw.decode("utf-8") if isinstance(spend_raw, bytes) else str(spend_raw))
            await Tenant.objects.filter(pk=tenant_id).aupdate(current_spend_usd=spend)

        return await Tenant.objects.aget(pk=tenant_id)

    @classmethod
    async def top_up_budget(cls, tenant_id: UUID, additional_usd: Decimal) -> Tenant:
        """Add budget to tenant in both PostgreSQL and Redis, clearing lockout."""
        # Update DB
        await Tenant.objects.filter(pk=tenant_id).aupdate(
            monthly_budget_usd=F("monthly_budget_usd") + additional_usd
        )
        tenant = await Tenant.objects.aget(pk=tenant_id)

        # Update Redis
        redis = await get_redis_client()
        tenant_id_str = str(tenant_id)
        await redis.eval(
            TOP_UP_LUA,
            3,
            cls._budget_key(tenant_id_str),
            cls._spend_key(tenant_id_str),
            cls._lockout_key(tenant_id_str),
            str(additional_usd),
            str(tenant.monthly_budget_usd),
        )

        return tenant

    @classmethod
    async def reset_spend(cls, tenant_id: UUID) -> Tenant:
        """Reset spend to zero in both PostgreSQL and Redis."""
        # Reset DB
        await Tenant.objects.filter(pk=tenant_id).aupdate(current_spend_usd=Decimal("0.000000"))
        tenant = await Tenant.objects.aget(pk=tenant_id)

        # Reset Redis
        redis = await get_redis_client()
        tenant_id_str = str(tenant_id)
        await redis.set(cls._spend_key(tenant_id_str), "0.000000")
        await redis.delete(cls._lockout_key(tenant_id_str))

        return tenant
