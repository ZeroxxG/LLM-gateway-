"""
gateway/ratelimit/limiter.py
============================
Distributed Multi-Dimensional Rate Limiting Engine.

Enforces RPM (Requests Per Minute) and TPM (Tokens Per Minute) per Tenant using
atomic Redis Lua scripts. Guarantees zero race conditions under concurrent requests.
"""
from dataclasses import dataclass
import logging
import time
from typing import Optional

from gateway.utils.redis import get_redis_client

logger = logging.getLogger(__name__)

# Atomic Redis Lua script for token/request consumption
# KEYS[1]: bucket key (e.g. ratelimit:tenant_id:rpm:12345)
# ARGV[1]: maximum limit allowed
# ARGV[2]: amount to consume (1 for request, N for tokens)
# ARGV[3]: window TTL in seconds (60s)
# Returns: [allowed (1 or 0), current_usage, ttl_remaining]
RATE_LIMIT_LUA_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local amount = tonumber(ARGV[2])
local window_ttl = tonumber(ARGV[3])

local current = redis.call('GET', key)
local current_num = current and tonumber(current) or 0

if (current_num + amount) > limit then
    local ttl = redis.call('TTL', key)
    if ttl < 0 then
        ttl = window_ttl
        redis.call('EXPIRE', key, window_ttl)
    end
    return {0, current_num, ttl}
else
    local new_val = redis.call('INCRBY', key, amount)
    if new_val == amount then
        redis.call('EXPIRE', key, window_ttl)
    end
    local ttl = redis.call('TTL', key)
    return {1, new_val, ttl}
end
"""


@dataclass
class RateLimitResult:
    """Standardized result of a rate limit evaluation."""

    allowed: bool
    limit: int
    current: int
    remaining: int
    reset_seconds: int
    dimension: str  # "RPM" or "TPM"

    @property
    def retry_after(self) -> int:
        return max(1, self.reset_seconds)


class DistributedRateLimiter:
    """
    Redis-backed sliding window rate limiter enforcing tenant RPM and TPM quotas.
    """

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds

    def _get_window_bucket(self) -> int:
        """Calculate discrete window bucket ID based on current epoch time."""
        return int(time.time() // self.window_seconds)

    def _rpm_key(self, tenant_id: str, bucket: int) -> str:
        return f"ratelimit:{tenant_id}:rpm:{bucket}"

    def _tpm_key(self, tenant_id: str, bucket: int) -> str:
        return f"ratelimit:{tenant_id}:tpm:{bucket}"

    async def check_rpm(self, tenant_id: str, rpm_limit: int) -> RateLimitResult:
        """
        Atomically increments and verifies the tenant's Requests Per Minute (RPM).

        Returns:
            RateLimitResult containing limit, remaining, and TTL.
        """
        if rpm_limit <= 0:
            return RateLimitResult(allowed=True, limit=rpm_limit, current=0, remaining=999999, reset_seconds=0, dimension="RPM")

        redis = await get_redis_client()
        bucket = self._get_window_bucket()
        key = self._rpm_key(tenant_id, bucket)

        res = await redis.eval(
            RATE_LIMIT_LUA_SCRIPT,
            1,
            key,
            rpm_limit,
            1,  # 1 request consumed
            self.window_seconds,
        )

        allowed_flag, current_usage, ttl = int(res[0]), int(res[1]), int(res[2])
        remaining = max(0, rpm_limit - current_usage)

        return RateLimitResult(
            allowed=(allowed_flag == 1),
            limit=rpm_limit,
            current=current_usage,
            remaining=remaining,
            reset_seconds=max(1, ttl),
            dimension="RPM",
        )

    async def get_current_rpm(self, tenant_id: str, rpm_limit: int) -> RateLimitResult:
        """Read-only check of current RPM usage without consuming counter."""
        if rpm_limit <= 0:
            return RateLimitResult(allowed=True, limit=rpm_limit, current=0, remaining=999999, reset_seconds=0, dimension="RPM")

        redis = await get_redis_client()
        bucket = self._get_window_bucket()
        key = self._rpm_key(tenant_id, bucket)

        current_val = await redis.get(key)
        ttl = await redis.ttl(key)
        current_num = int(current_val) if current_val else 0
        remaining = max(0, rpm_limit - current_num)

        return RateLimitResult(
            allowed=(current_num < rpm_limit),
            limit=rpm_limit,
            current=current_num,
            remaining=remaining,
            reset_seconds=max(1, ttl) if ttl > 0 else self.window_seconds,
            dimension="RPM",
        )

    async def check_and_consume_tpm(self, tenant_id: str, tpm_limit: int, tokens: int) -> RateLimitResult:
        """
        Atomically records token consumption against the Tokens Per Minute (TPM) quota.

        Returns:
            RateLimitResult for the TPM evaluation.
        """
        if tpm_limit <= 0 or tokens <= 0:
            return RateLimitResult(allowed=True, limit=tpm_limit, current=0, remaining=999999, reset_seconds=0, dimension="TPM")

        redis = await get_redis_client()
        bucket = self._get_window_bucket()
        key = self._tpm_key(tenant_id, bucket)

        res = await redis.eval(
            RATE_LIMIT_LUA_SCRIPT,
            1,
            key,
            tpm_limit,
            tokens,
            self.window_seconds,
        )

        allowed_flag, current_usage, ttl = int(res[0]), int(res[1]), int(res[2])
        remaining = max(0, tpm_limit - current_usage)

        return RateLimitResult(
            allowed=(allowed_flag == 1),
            limit=tpm_limit,
            current=current_usage,
            remaining=remaining,
            reset_seconds=max(1, ttl),
            dimension="TPM",
        )

    async def reset_tenant_limits(self, tenant_id: str) -> None:
        """Clear all rate limit keys for a given tenant (useful in tests/admin reset)."""
        redis = await get_redis_client()
        bucket = self._get_window_bucket()
        await redis.delete(
            self._rpm_key(tenant_id, bucket),
            self._tpm_key(tenant_id, bucket),
        )


# Global rate limiter instance
rate_limiter = DistributedRateLimiter()
