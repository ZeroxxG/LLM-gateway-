"""
api/adapters.py
===============
Re-exports adapter classes for namespace compatibility.
"""
from gateway.adapters.base import BaseProviderAdapter, close_shared_http_client, get_shared_http_client
from gateway.adapters.exceptions import (
    ModelNotSupportedError,
    ProviderAPIError,
    ProviderAuthError,
    ProviderException,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from gateway.adapters.factory import ProviderFactory
from gateway.adapters.gemini import GeminiAdapter
from gateway.adapters.groq import GroqAdapter
from gateway.adapters.mock import MockAdapter
from gateway.adapters.openai import OpenAIAdapter
from gateway.adapters.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    TokenUsage,
)

__all__ = [
    "BaseProviderAdapter",
    "GroqAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    "MockAdapter",
    "ProviderFactory",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "TokenUsage",
    "ChatCompletionChoice",
    "ProviderException",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderAPIError",
    "ModelNotSupportedError",
    "get_shared_http_client",
    "close_shared_http_client",
]
