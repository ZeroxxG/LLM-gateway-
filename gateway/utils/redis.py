"""
gateway/utils/redis.py
======================
Async Redis client — process-level singleton with lazy initialisation.

Design decisions:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Module-level singleton pattern                                     │
  │  ─────────────────────────────                                      │
  │  _redis_client is None on first import.                             │
  │  get_redis_client() creates it once on first call.                  │
  │  Every subsequent call in the same process returns the same object. │
  │                                                                     │
  │  Why singleton?                                                     │
  │  Creating a new connection (or pool) per request would exhaust OS   │
  │  file descriptors within seconds under load.                        │
  └─────────────────────────────────────────────────────────────────────┘

  max_connections=50:
    Each Uvicorn async worker is single-threaded but can multiplex many
    concurrent coroutines. 50 pooled connections covers heavy concurrency
    without overwhelming Redis's connection limit.

  decode_responses=True:
    All Redis values are returned as str, not bytes. This is consistent
    with Python string operations throughout the codebase and removes the
    need for explicit .decode("utf-8") calls everywhere.

  socket_connect_timeout / socket_timeout:
    Prevent the event loop from hanging indefinitely on a stalled Redis TCP
    connection. After the timeout, aioredis raises a ConnectionError which
    our health check and rate limiter can catch and handle gracefully.

  retry_on_timeout=True:
    Transparently retries the failed command once before raising. Handles
    transient network blips without surfacing errors to callers.

  health_check_interval=30:
    Idle connections in the pool are pinged every 30 seconds. This prevents
    the TCP connection from being silently dropped by a firewall or NAT
    device (a common production failure mode in cloud environments).

Usage:
    from gateway.utils.redis import get_redis_client

    client = await get_redis_client()
    await client.set("key", "value", ex=60)
    value = await client.get("key")   # Returns str, not bytes
"""
import logging
from typing import Optional

import redis.asyncio as aioredis
from django.conf import settings

logger = logging.getLogger(__name__)

# Process-level singleton — one connection pool per Uvicorn worker process
_redis_client: Optional[aioredis.Redis] = None


async def get_redis_client() -> aioredis.Redis:
    """
    Return the shared async Redis client, creating and configuring it on first call.

    Thread/coroutine safety:
      In asyncio's single-threaded event loop, cooperative multitasking means
      only one coroutine runs at a time. The `if _redis_client is None` check
      cannot race because no other coroutine can execute between the check and
      the assignment — there is no await between them.

    Returns:
        aioredis.Redis: Configured async Redis client backed by a connection pool.
    """
    global _redis_client

    if _redis_client is None:
        logger.debug("Initialising async Redis client → %s", settings.REDIS_URL)
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,      # Return str instead of bytes
            socket_connect_timeout=5,   # TCP connect timeout (seconds)
            socket_timeout=5,           # Command read/write timeout (seconds)
            retry_on_timeout=True,      # Retry once on TimeoutError
            max_connections=50,         # Connection pool cap per worker
            health_check_interval=30,   # Ping idle connections every 30 s
        )
        logger.info("Async Redis client initialised (pool max=%d)", 50)

    return _redis_client


async def close_redis_client() -> None:
    """
    Gracefully drain and close the Redis connection pool.

    Call this during ASGI lifespan shutdown to flush in-flight commands
    and release file descriptors cleanly before the process exits.
    """
    global _redis_client

    if _redis_client is not None:
        client = _redis_client
        _redis_client = None
        logger.info("Closing async Redis client pool")
        try:
            await client.aclose()
            await client.connection_pool.disconnect()
        except Exception:
            pass



