"""
gateway/security.py
===================
Cryptographic API Key Generation, Hashing, and Django Ninja Async Auth.

Security Architecture:
  1. Key Generation:
     - Uses Python's `secrets` module (cryptographically secure CSPRNG).
     - Format: `gw_live_<32_random_hex_chars>`.
     - 16 bytes = 128 bits of cryptographic entropy (impossible to brute force).

  2. One-Way Hashing:
     - The raw key is NEVER stored in PostgreSQL.
     - Only the SHA-256 digest (`hexdigest()`, 64 characters) is saved.
     - Even if the database is dumped or compromised, attacker cannot reconstruct
       valid API keys.
     - Timing attacks are prevented by standard constant-time lookups or hash matching.

  3. Async Non-Blocking Authentication:
     - `verify_api_key_async` queries the database using Django 5.x async ORM
       (`select_related("tenant").filter(...).afirst()`).
     - Never blocks the asyncio event loop thread.
     - Upon valid auth, `last_used_at` is updated via `aupdate()` asynchronously.
     - Attaches the verified `Tenant` instance directly to `request.auth`.
"""
import hashlib
import logging
import secrets
from typing import Optional, Tuple

from django.utils import timezone
from ninja.security import APIKeyHeader, HttpBearer

from gateway.models import APIKey, Tenant

logger = logging.getLogger(__name__)

DEFAULT_KEY_PREFIX = "gw_live_"


def hash_api_key(raw_key: str) -> str:
    """
    Compute deterministic SHA-256 hex digest of a raw API key token.

    Args:
        raw_key: The plaintext token provided by the client.

    Returns:
        64-character lowercase hex digest string.
    """
    cleaned_key = raw_key.strip()
    return hashlib.sha256(cleaned_key.encode("utf-8")).hexdigest()


def generate_api_key(prefix: str = DEFAULT_KEY_PREFIX) -> Tuple[str, str, str]:
    """
    Generate a cryptographically secure random API key pair for a tenant.

    Entropy:
      secrets.token_hex(16) yields 32 hexadecimal characters (128 bits of entropy).
      Example: "gw_live_a1b2c3d4e5f678901234567890abcdef"

    Returns:
        Tuple of:
          - raw_key: Plaintext token to display ONCE to the user.
          - key_prefix: The public prefix (e.g., "gw_live_") for identification.
          - hashed_key: SHA-256 hex digest to persist in the database.
    """
    random_entropy = secrets.token_hex(16)
    raw_key = f"{prefix}{random_entropy}"
    key_prefix = raw_key[: len(prefix)]
    hashed_key = hash_api_key(raw_key)
    return raw_key, key_prefix, hashed_key


async def verify_api_key_async(raw_key: str) -> Optional[Tenant]:
    """
    Authenticate an incoming raw API key against the database asynchronously.

    Execution:
      1. Hashes the incoming raw token using SHA-256.
      2. Queries PostgreSQL via Django 5.x async ORM:
         `APIKey.objects.select_related("tenant").filter(...).afirst()`
      3. Validates both key `is_active=True` and tenant `is_active=True`.
      4. Asynchronously updates `last_used_at` timestamp.
      5. Returns the associated `Tenant` model instance, or None if invalid.

    Thread Safety:
      Pure async non-blocking execution inside the asyncio event loop.
    """
    if not raw_key or not raw_key.strip():
        return None

    hashed = hash_api_key(raw_key)

    try:
        # Non-blocking single-query lookup joining Tenant
        key_record = await (
            APIKey.objects.select_related("tenant")
            .filter(
                hashed_key=hashed,
                is_active=True,
                tenant__is_active=True,
            )
            .afirst()
        )

        if key_record is None:
            logger.warning("API Key authentication failed: key not found or inactive")
            return None

        # Asynchronously touch last_used_at without blocking
        await APIKey.objects.filter(pk=key_record.pk).aupdate(last_used_at=timezone.now())

        return key_record.tenant

    except Exception as exc:
        logger.error("Unexpected error during API key verification: %s", exc, exc_info=True)
        return None


class APIKeyAuth(HttpBearer):
    """
    Django Ninja async security dependency parsing `Authorization: Bearer <token>`.

    When applied to a route or router:
      1. Extracts the token from the `Authorization: Bearer ...` header.
      2. Calls `verify_api_key_async(token)`.
      3. If valid, Django Ninja populates `request.auth` with the `Tenant` instance.
      4. If invalid or missing, Django Ninja terminates the request with HTTP 401.
    """

    openapi_name = "BearerAuth"
    openapi_description = "Supply your API key in the format: `Bearer gw_live_<hex32>`"

    async def authenticate(self, request, token: str) -> Optional[Tenant]:
        if not token:
            return None
        return await verify_api_key_async(token)


class APIKeyHeaderAuth(APIKeyHeader):
    """
    Alternative security dependency parsing `X-API-Key: <token>`.
    Permits clients that prefer custom headers over standard Authorization Bearer.
    """

    param_name = "X-API-Key"
    openapi_name = "ApiKeyHeaderAuth"
    openapi_description = "Supply your API key via `X-API-Key: gw_live_<hex32>` header"

    async def authenticate(self, request, key: Optional[str]) -> Optional[Tenant]:
        if not key:
            return None
        return await verify_api_key_async(key)


# Shared instance for easy router import
api_key_auth = APIKeyAuth()
api_key_header_auth = APIKeyHeaderAuth()
