"""
gateway/caching/cache.py
========================
Prompt & Response Caching Engine in Redis.

Provides deterministic SHA-256 exact-match caching for LLM completions to:
  1. Eliminate redundant LLM provider API invocations.
  2. Deliver sub-5ms responses for repeated queries.
  3. Reduce FinOps spend to $0.00 on cache hits.
  4. Support cache invalidation and operational hit/miss observability.
"""
import hashlib
import json
import logging
from typing import Any, Dict, Optional

from gateway.adapters.schemas import ChatCompletionRequest, ChatCompletionResponse
from gateway.utils.redis import get_redis_client

logger = logging.getLogger(__name__)


class PromptCacheManager:
    """Manages exact-match prompt completion caching in Redis."""

    DEFAULT_TTL_SECONDS: int = 3600  # 1 hour default TTL

    @classmethod
    def compute_cache_key(cls, tenant_id: str, request: ChatCompletionRequest) -> str:
        """
        Generates a deterministic SHA-256 cache key based on normalized request parameters.
        Includes messages sequence, model, temperature, top_p, and max_tokens.
        """
        normalized_messages = [
            {"role": msg.role, "content": msg.content.strip()}
            for msg in request.messages
        ]
        key_dict = {
            "model": request.model,
            "messages": normalized_messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        canonical_json = json.dumps(key_dict, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        return f"cache:resp:{tenant_id}:{digest}"

    @classmethod
    async def get(cls, cache_key: str) -> Optional[ChatCompletionResponse]:
        """
        Retrieves cached ChatCompletionResponse from Redis if available.
        Returns None on cache miss or deserialization error.
        """
        client = await get_redis_client()
        raw_val = await client.get(cache_key)
        if not raw_val:
            return None

        try:
            val_str = raw_val.decode("utf-8") if isinstance(raw_val, bytes) else str(raw_val)
            data = json.loads(val_str)
            return ChatCompletionResponse.model_validate(data)
        except Exception as exc:
            logger.warning("Failed to deserialize cached response for key '%s': %s", cache_key, exc)
            return None

    @classmethod
    async def set(
        cls,
        cache_key: str,
        response: ChatCompletionResponse,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Stores a serialized ChatCompletionResponse in Redis with expiration TTL."""
        client = await get_redis_client()
        ttl = ttl_seconds or cls.DEFAULT_TTL_SECONDS
        json_data = response.model_dump_json()
        await client.set(cache_key, json_data, ex=ttl)

    @classmethod
    async def record_hit(cls, tenant_id: str) -> None:
        """Increments cache hit counter for tenant telemetry."""
        client = await get_redis_client()
        await client.incr(f"cache:stats:{tenant_id}:hits")

    @classmethod
    async def record_miss(cls, tenant_id: str) -> None:
        """Increments cache miss counter for tenant telemetry."""
        client = await get_redis_client()
        await client.incr(f"cache:stats:{tenant_id}:misses")

    @classmethod
    async def get_stats(cls, tenant_id: str) -> Dict[str, Any]:
        """Calculates cache hit/miss ratio and counts active cached keys for a tenant."""
        client = await get_redis_client()
        hits_key = f"cache:stats:{tenant_id}:hits"
        misses_key = f"cache:stats:{tenant_id}:misses"

        raw_hits, raw_misses = await client.mget([hits_key, misses_key])
        hits = int(raw_hits or 0)
        misses = int(raw_misses or 0)
        total = hits + misses
        ratio = round((hits / total) if total > 0 else 0.0, 4)

        # Count active cached keys for this tenant using scan_iter
        pattern = f"cache:resp:{tenant_id}:*"
        count = 0
        async for _ in client.scan_iter(match=pattern, count=100):
            count += 1

        return {
            "tenant_id": tenant_id,
            "hits": hits,
            "misses": misses,
            "total_requests": total,
            "hit_ratio": ratio,
            "cached_keys": count,
        }

    @classmethod
    async def clear_tenant_cache(cls, tenant_id: str) -> int:
        """Deletes all active cache entries for a given tenant."""
        client = await get_redis_client()
        pattern = f"cache:resp:{tenant_id}:*"
        keys_to_delete = []
        async for key in client.scan_iter(match=pattern, count=100):
            keys_to_delete.append(key)

        if keys_to_delete:
            deleted_count = await client.delete(*keys_to_delete)
            return int(deleted_count)
        return 0
