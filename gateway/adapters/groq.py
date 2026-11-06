"""
gateway/adapters/groq.py
========================
Groq Cloud Llama-3 Provider Adapter.

Executes low-latency inference against Groq's LPUs via their OpenAI-compatible REST API.
Endpoint: https://api.groq.com/openai/v1/chat/completions
"""
import json
import logging
import time
from typing import AsyncGenerator, Dict
from decimal import Decimal
import httpx
from django.conf import settings

from gateway.adapters.base import BaseProviderAdapter, get_shared_http_client
from gateway.adapters.exceptions import (
    ProviderAPIError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
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

# Map gateway canonical model names to upstream Groq model IDs
GROQ_MODEL_MAP: Dict[str, str] = {
    "groq/llama-3.1-8b": "llama-3.1-8b-instant",
    "groq/llama-3.3-70b": "llama-3.3-70b-versatile",
    "groq/mixtral-8x7b": "mixtral-8x7b-32768",
}


class GroqAdapter(BaseProviderAdapter):
    """Adapter for Groq LPU inference."""

    provider_name: str = "groq"
    base_url: str = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or getattr(settings, "GROQ_API_KEY", "")

    def resolve_model(self, model: str) -> str:
        """Resolve gateway canonical model identifier to Groq model ID."""
        if model in GROQ_MODEL_MAP:
            return GROQ_MODEL_MAP[model]
        if model.startswith("groq/"):
            return model[len("groq/"):]
        return model

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.api_key:
            raise ProviderAuthError("Groq API key is not configured.", provider=self.provider_name, status_code=401)

        upstream_model = self.resolve_model(request.model)
        payload = {
            "model": upstream_model,
            "messages": [msg.model_dump() for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        client = await get_shared_http_client()
        start_time = time.perf_counter()

        try:
            response = await client.post(self.base_url, json=payload, headers=headers)
            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 401 or response.status_code == 403:
                raise ProviderAuthError(
                    f"Groq authentication rejected: {response.text}",
                    provider=self.provider_name,
                    status_code=response.status_code,
                    raw_response=response.text,
                )
            elif response.status_code == 429:
                raise ProviderRateLimitError(
                    f"Groq rate limit exceeded: {response.text}",
                    provider=self.provider_name,
                    status_code=429,
                    raw_response=response.text,
                )
            elif response.status_code >= 400:
                raise ProviderAPIError(
                    f"Groq API error ({response.status_code}): {response.text}",
                    provider=self.provider_name,
                    status_code=response.status_code,
                    raw_response=response.text,
                )

            data = response.json()
            raw_choice = data["choices"][0]
            raw_msg = raw_choice["message"]
            raw_usage = data.get("usage", {})

            return ChatCompletionResponse(
                id=data.get("id", ""),
                model=request.model,
                provider=self.provider_name,
                choices=[
                    ChatCompletionChoice(
                        index=raw_choice.get("index", 0),
                        message=ChatMessage(
                            role=raw_msg.get("role", "assistant"),
                            content=raw_msg.get("content", ""),
                        ),
                        finish_reason=raw_choice.get("finish_reason", "stop"),
                    )
                ],
                usage=TokenUsage(
                    prompt_tokens=raw_usage.get("prompt_tokens", 0),
                    completion_tokens=raw_usage.get("completion_tokens", 0),
                    total_tokens=raw_usage.get("total_tokens", 0),
                ),
                latency_ms=round(latency_ms, 2),
            )

        except httpx.TimeoutException as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            raise ProviderTimeoutError(
                f"Groq request timed out after {latency_ms:.1f}ms: {exc}",
                provider=self.provider_name,
            ) from exc
        except httpx.RequestError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            raise ProviderAPIError(
                f"Network transport error calling Groq: {exc}",
                provider=self.provider_name,
            ) from exc

    async def stream_complete(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Streams chat completion tokens from Groq via Server-Sent Events."""
        if not self.api_key:
            raise ProviderAuthError("Groq API key is not configured.", provider=self.provider_name, status_code=401)

        upstream_model = self.resolve_model(request.model)
        payload = {
            "model": upstream_model,
            "messages": [msg.model_dump() for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = await get_shared_http_client()

        try:
            async with client.stream("POST", self.base_url, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    raw_err = await resp.aread()
                    raise ProviderAPIError(
                        f"Groq stream rejected ({resp.status_code}): {raw_err.decode('utf-8', errors='replace')}",
                        provider=self.provider_name,
                        status_code=resp.status_code,
                    )

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        raw_data = line[len("data: "):].strip()
                        if raw_data == "[DONE]":
                            break
                        try:
                            item = json.loads(raw_data)
                            choices = []
                            for c in item.get("choices", []):
                                delta = c.get("delta", {})
                                choices.append(
                                    ChatCompletionChunkChoice(
                                        index=c.get("index", 0),
                                        delta=ChatCompletionChunkDelta(
                                            role=delta.get("role"),
                                            content=delta.get("content"),
                                        ),
                                        finish_reason=c.get("finish_reason"),
                                    )
                                )
                            yield ChatCompletionChunk(
                                id=item.get("id", ""),
                                model=request.model,
                                provider=self.provider_name,
                                choices=choices,
                            )
                        except json.JSONDecodeError:
                            continue
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Groq stream timed out: {exc}", provider=self.provider_name) from exc
        except httpx.RequestError as exc:
            raise ProviderAPIError(f"Network error on Groq stream: {exc}", provider=self.provider_name) from exc

