"""
gateway/adapters/mock.py
========================
Mock Provider Adapter.

Deterministic, zero-cost provider adapter used for local integration tests,
CI/CD test suites, and simulating downstream failures.
"""
import asyncio
import logging
import time
from decimal import Decimal
from typing import AsyncGenerator
from uuid import uuid4

from gateway.adapters.base import BaseProviderAdapter
from gateway.adapters.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class MockAdapter(BaseProviderAdapter):
    """Mock adapter generating predictable synthetic completions."""

    provider_name: str = "mock"

    def __init__(self, simulated_latency_ms: float = 15.0, fail_on_stream: bool = False):
        self.simulated_latency_ms = simulated_latency_ms
        self.fail_on_stream = fail_on_stream

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        start_time = time.perf_counter()

        # Simulate non-blocking asynchronous network I/O
        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)

        # Approximate token estimation for mock response
        prompt_text = " ".join(msg.content for msg in request.messages)
        prompt_tokens = max(1, len(prompt_text.split()) * 2)

        last_user_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "Hello",
        )
        response_text = f"[Mock LLM Response] Successfully processed: '{last_user_msg}' using model '{request.model}'."
        completion_tokens = max(1, len(response_text.split()) * 2)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return ChatCompletionResponse(
            model=request.model,
            provider=self.provider_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_text,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=round(latency_ms, 2),
        )

    async def stream_complete(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Streams mock response chunks word-by-word."""
        if self.fail_on_stream:
            from gateway.adapters.exceptions import ProviderAPIError
            raise ProviderAPIError("Simulated mock streaming failure", provider=self.provider_name, status_code=500)

        chunk_id = f"chatcmpl-mock-{uuid4().hex[:10]}"
        created = int(time.time())

        # Initial chunk announcing assistant role
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=request.model,
            provider=self.provider_name,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
        )

        last_user_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "Hello",
        )
        response_text = f"[Mock Stream] Streamed response for: '{last_user_msg}'"
        words = response_text.split(" ")

        for i, word in enumerate(words):
            if self.simulated_latency_ms > 0:
                await asyncio.sleep(0.005)
            prefix = " " if i > 0 else ""
            yield ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=request.model,
                provider=self.provider_name,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=prefix + word),
                        finish_reason=None,
                    )
                ],
            )

        # Final chunk indicating stop
        yield ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=request.model,
            provider=self.provider_name,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content=""),
                    finish_reason="stop",
                )
            ],
        )

