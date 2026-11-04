"""
gateway/adapters/exceptions.py
==============================
Normalized Provider Exceptions.

Converts diverse upstream HTTP error codes and body payloads into
consistent, structured gateway exceptions for routing and fallback logic.
"""
from typing import Optional


class ProviderException(Exception):
    """Base exception for all LLM provider adapter failures."""

    def __init__(self, message: str, provider: str, status_code: Optional[int] = None, raw_response: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.raw_response = raw_response

    def __str__(self) -> str:
        return f"[{self.provider.upper()}] Status {self.status_code}: {self.message}"


class ProviderAuthError(ProviderException):
    """Raised when upstream API key is missing, invalid, or unauthorized (HTTP 401/403)."""
    pass


class ProviderRateLimitError(ProviderException):
    """Raised when upstream provider throttles the request (HTTP 429)."""
    pass


class ProviderTimeoutError(ProviderException):
    """Raised when an upstream request exceeds the configured network timeout."""
    pass


class ProviderAPIError(ProviderException):
    """Raised on upstream 5xx server errors or bad JSON responses."""
    pass


class ModelNotSupportedError(Exception):
    """Raised when a requested model is unrecognized or has no configured provider."""
    def __init__(self, model: str):
        super().__init__(f"Model '{model}' is not supported by any active provider adapter.")
        self.model = model
