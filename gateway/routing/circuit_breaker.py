"""
gateway/routing/circuit_breaker.py
==================================
Distributed Redis-Backed Circuit Breaker for Upstream LLM Providers.

Protects the gateway from cascading upstream outages, timeouts, and rate limits.

States:
  - CLOSED: Normal operation. Requests flow to the provider.
  - OPEN: Tripped after consecutive failure threshold. Fast-fails immediately.
  - HALF_OPEN: Trial state after recovery timeout. Permits probe request.
"""
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional

from gateway.utils.redis import get_redis_client

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Canonical circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class RedisCircuitBreaker:
    """
    Distributed async circuit breaker backed by Redis 7.x.

    Prevents connection pool exhaustion by fast-failing requests to providers
    that are currently experiencing outages or severe throttling.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        failure_window: int = 60,
    ):
        """
        Args:
            failure_threshold: Number of failures before tripping circuit from CLOSED to OPEN.
            recovery_timeout: Seconds to remain OPEN before transitioning to HALF_OPEN.
            failure_window: TTL in seconds for failure counters.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_window = failure_window

    def _state_key(self, provider: str) -> str:
        return f"cb:{provider.lower()}:state"

    def _failures_key(self, provider: str) -> str:
        return f"cb:{provider.lower()}:failures"

    def _last_failure_key(self, provider: str) -> str:
        return f"cb:{provider.lower()}:last_failure_ts"

    async def get_state(self, provider: str) -> CircuitState:
        """
        Retrieves the current circuit state for a provider.
        Automatically transitions OPEN -> HALF_OPEN if recovery_timeout has elapsed.
        """
        redis = await get_redis_client()
        provider_key = provider.lower()

        state_str = await redis.get(self._state_key(provider_key))
        if not state_str:
            return CircuitState.CLOSED

        current_state = CircuitState(state_str)

        # Check if OPEN circuit is ready for trial probe in HALF_OPEN
        if current_state == CircuitState.OPEN:
            last_ts_str = await redis.get(self._last_failure_key(provider_key))
            if last_ts_str:
                elapsed = time.time() - float(last_ts_str)
                if elapsed >= self.recovery_timeout:
                    logger.info(
                        "Circuit breaker for '%s' cooldown elapsed (%.1fs >= %.1fs). Transitioning OPEN -> HALF_OPEN.",
                        provider_key,
                        elapsed,
                        self.recovery_timeout,
                    )
                    await redis.set(self._state_key(provider_key), CircuitState.HALF_OPEN.value)
                    return CircuitState.HALF_OPEN

        return current_state

    async def can_execute(self, provider: str) -> bool:
        """
        Check if a request is permitted to call the provider.
        Returns False if circuit is OPEN (fast-skip).
        """
        state = await self.get_state(provider)
        return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    async def record_success(self, provider: str) -> None:
        """
        Records a successful request.
        Resets failure count and closes the circuit.
        """
        redis = await get_redis_client()
        provider_key = provider.lower()

        current_state = await self.get_state(provider_key)
        if current_state != CircuitState.CLOSED:
            logger.info("Circuit breaker for '%s' recovered successfully. Resetting to CLOSED.", provider_key)

        pipe = redis.pipeline()
        pipe.set(self._state_key(provider_key), CircuitState.CLOSED.value)
        pipe.delete(self._failures_key(provider_key))
        pipe.delete(self._last_failure_key(provider_key))
        await pipe.execute()

    async def record_failure(self, provider: str, reason: str = "") -> CircuitState:
        """
        Records an error (timeout, 5xx, or 429).
        Increments failures and trips circuit to OPEN if threshold reached.
        """
        redis = await get_redis_client()
        provider_key = provider.lower()
        now = time.time()

        current_state = await self.get_state(provider_key)

        # In HALF_OPEN, a single failure immediately trips back to OPEN
        if current_state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker probe failed for '%s' (%s). Re-tripping to OPEN.", provider_key, reason)
            pipe = redis.pipeline()
            pipe.set(self._state_key(provider_key), CircuitState.OPEN.value)
            pipe.set(self._last_failure_key(provider_key), str(now))
            await pipe.execute()
            return CircuitState.OPEN

        # Increment failure count with TTL window
        failures = await redis.incr(self._failures_key(provider_key))
        if failures == 1:
            await redis.expire(self._failures_key(provider_key), self.failure_window)

        await redis.set(self._last_failure_key(provider_key), str(now))

        if failures >= self.failure_threshold:
            logger.warning(
                "Circuit breaker tripped for '%s': %d consecutive failures reached. State -> OPEN (%s)",
                provider_key,
                failures,
                reason,
            )
            await redis.set(self._state_key(provider_key), CircuitState.OPEN.value)
            return CircuitState.OPEN

        return CircuitState.CLOSED

    async def reset(self, provider: str) -> None:
        """Manually reset the circuit breaker for a provider."""
        redis = await get_redis_client()
        provider_key = provider.lower()
        await redis.delete(
            self._state_key(provider_key),
            self._failures_key(provider_key),
            self._last_failure_key(provider_key),
        )

    async def get_circuit_status(self, provider: str) -> Dict[str, Any]:
        """Returns diagnostic metrics for a specific provider circuit."""
        redis = await get_redis_client()
        provider_key = provider.lower()

        state = await self.get_state(provider_key)
        failures_str = await redis.get(self._failures_key(provider_key))
        last_failure_str = await redis.get(self._last_failure_key(provider_key))

        failures = int(failures_str) if failures_str else 0
        last_failure_ts = float(last_failure_str) if last_failure_str else None

        return {
            "provider": provider_key,
            "state": state.value,
            "consecutive_failures": failures,
            "failure_threshold": self.failure_threshold,
            "last_failure_timestamp": last_failure_ts,
            "recovery_timeout_seconds": self.recovery_timeout,
        }

    async def get_all_circuits(self, providers: List[str]) -> List[Dict[str, Any]]:
        """Returns status for a collection of providers."""
        statuses = []
        for p in providers:
            status = await self.get_circuit_status(p)
            statuses.append(status)
        return statuses


# Global circuit breaker instance
circuit_breaker = RedisCircuitBreaker()
