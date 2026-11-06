"""
gateway/observability/latency.py
================================
Tracks request latency measurements and computes exact empirical quantiles
(p50, p90, p95, p99) using a rolling sliding-window buffer in Redis and local memory.
"""
import collections
import logging
import math
from typing import Dict, List, Optional

from gateway.utils.redis import get_redis_client

logger = logging.getLogger(__name__)

# In-memory sliding-window queues per tenant and global
_local_latency_buffer: Dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=1000))


def calculate_percentile(sorted_vals: List[float], p: float) -> float:
    """Compute percentile p (0..100) using linear interpolation."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(float(sorted_vals[0]), 2)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(float(sorted_vals[int(k)]), 2)
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return round(float(d0 + d1), 2)


class LatencyTracker:
    """Records latencies in Redis and calculates empirical quantiles."""

    MAX_SAMPLES = 1000
    BUFFER_TTL_SECONDS = 604800  # 7-day retention for sliding window history

    @classmethod
    async def record_latency(cls, tenant_id: str, latency_ms: float, model: str = "") -> None:
        """Record latency measurement in both local deque and Redis sliding window."""
        val = float(latency_ms)
        tid = str(tenant_id)

        # Store in local memory deque
        _local_latency_buffer[tid].append(val)
        _local_latency_buffer["global"].append(val)

        # Store in Redis list with LTRIM and refresh TTL
        try:
            client = await get_redis_client()
            tenant_key = f"llm:latency:{tid}"
            global_key = "llm:latency:global"
            pipe = client.pipeline()
            pipe.rpush(tenant_key, str(val))
            pipe.ltrim(tenant_key, -cls.MAX_SAMPLES, -1)
            pipe.expire(tenant_key, cls.BUFFER_TTL_SECONDS)
            pipe.rpush(global_key, str(val))
            pipe.ltrim(global_key, -cls.MAX_SAMPLES, -1)
            pipe.expire(global_key, cls.BUFFER_TTL_SECONDS)
            await pipe.execute()
        except Exception as exc:
            logger.warning("Failed to record latency sample in Redis: %s", exc)

    @classmethod
    async def get_percentiles(cls, tenant_id: Optional[str] = None) -> Dict[str, float]:
        """
        Compute p50, p90, p95, p99 percentiles from sliding window buffer.
        Falls back to in-memory deque if Redis returns no samples.
        """
        tid = str(tenant_id) if tenant_id else None
        key = f"llm:latency:{tid}" if tid else "llm:latency:global"
        samples: List[float] = []

        try:
            client = await get_redis_client()
            raw_items = await client.lrange(key, 0, -1)
            for item in raw_items:
                try:
                    samples.append(float(item))
                except (ValueError, TypeError):
                    pass
        except Exception as exc:
            logger.warning("Failed to read latency samples from Redis: %s", exc)

        # Fallback to local memory if Redis returned no samples
        if not samples:
            local_queue = _local_latency_buffer.get(tid or "global")
            if local_queue:
                samples = list(local_queue)

        if not samples:
            return {
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "samples_count": 0,
            }

        sorted_vals = sorted(samples)
        return {
            "p50_ms": calculate_percentile(sorted_vals, 50.0),
            "p90_ms": calculate_percentile(sorted_vals, 90.0),
            "p95_ms": calculate_percentile(sorted_vals, 95.0),
            "p99_ms": calculate_percentile(sorted_vals, 99.0),
            "samples_count": len(sorted_vals),
        }

    @classmethod
    async def clear(cls, tenant_id: Optional[str] = None) -> None:
        """Clear latency samples for testing or tenant reset."""
        tid = str(tenant_id) if tenant_id else None
        if tid:
            _local_latency_buffer.pop(tid, None)
        else:
            _local_latency_buffer.clear()

        try:
            client = await get_redis_client()
            if tid:
                await client.delete(f"llm:latency:{tid}")
            else:
                await client.delete("llm:latency:global")
        except Exception:
            pass
