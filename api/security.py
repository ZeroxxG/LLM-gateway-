"""
api/security.py
===============
Re-exports security utilities from gateway.security for namespace compatibility.
"""
from gateway.security import (
    APIKeyAuth,
    APIKeyHeaderAuth,
    api_key_auth,
    api_key_header_auth,
    generate_api_key,
    hash_api_key,
    verify_api_key_async,
)

__all__ = [
    "generate_api_key",
    "hash_api_key",
    "verify_api_key_async",
    "APIKeyAuth",
    "APIKeyHeaderAuth",
    "api_key_auth",
    "api_key_header_auth",
]
