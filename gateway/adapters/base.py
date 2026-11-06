"""
gateway/adapters/base.py
========================
Abstract Base Provider Adapter with HTTP Connection Pooling.

All provider adapters (Groq, OpenAI, Gemini, Mock) inherit from BaseProviderAdapter.
Ensures:
  1. Strict async non-blocking HTTP network execution.
  2. Persistent HTTP connection pooling via singleton httpx.AsyncClient.
  3. Uniform latency profiling and error normalization.
"""
from abc import ABC, abstractmethod
import logging
from typing import AsyncGenerator, Optional
import httpx
from django.conf import settings

from gateway.adapters.schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

logger = logging.getLogger(__name__)

# Shared persistent async HTTP client across all adapters in worker process
_http_client: Optional[httpx.AsyncClient] = None


async def get_shared_http_client() -> httpx.AsyncClient:
    """
    Returns the process-wide httpx.AsyncClient singleton.
    Reuses connection pools and SSL sessions to eliminate TLS negotiation overhead.
    """
    global _http_client
    if _http_client is None or _http_client.is_closed:
        timeout_seconds = getattr(settings, "GATEWAY_REQUEST_TIMEOUT_SECONDS", 30)
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,
                read=float(timeout_seconds),
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=50,
                max_connections=200,
                keepalive_expiry=60.0,
            ),
            http2=False,
        )
    return _http_client


async def close_shared_http_client() -> None:
    """Gracefully drains and closes the shared HTTP connection pool."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


class BaseProviderAdapter(ABC):
    """Abstract interface defining the execution contract for all LLM providers."""

    provider_name: str = "base"

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Executes a chat completion request asynchronously against the upstream provider.

        Args:
            request: Standardized ChatCompletionRequest.

        Returns:
            ChatCompletionResponse normalized across providers.

        Raises:
            ProviderException: Normalized gateway exception for errors.
        """
        pass

    async def stream_complete(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Streams chat completion chunks asynchronously from upstream provider.
        Default implementation delegates to complete() and yields choices as chunks.
        """
        response = await self.complete(request)
        for choice in response.choices:
            yield ChatCompletionChunk(
                id=response.id,
                created=response.created,
                model=response.model,
                provider=self.provider_name,
                choices=[
                    ChatCompletionChunkChoice(
                        index=choice.index,
                        delta=ChatCompletionChunkDelta(
                            role=choice.message.role,
                            content=choice.message.content,
                        ),
                        finish_reason=choice.finish_reason,
                    )
                ],
            )


