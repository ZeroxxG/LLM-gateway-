"""
gateway/adapters/openai.py
==========================
OpenAI Provider Adapter.

Executes asynchronous chat completions against the OpenAI API.
Endpoint: https://api.openai.com/v1/chat/completions
"""
import logging
import time
from typing import Dict
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
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    TokenUsage,
)

logger = logging.getLogger(__name__)

OPENAI_MODEL_MAP: Dict[str, str] = {
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/o1-mini": "o1-mini",
}


class OpenAIAdapter(BaseProviderAdapter):
    """Adapter for OpenAI API."""

    provider_name: str = "openai"
    base_url: str = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or getattr(settings, "OPENAI_API_KEY", "")

    def resolve_model(self, model: str) -> str:
        """Resolve canonical model identifier to OpenAI model name."""
        if model in OPENAI_MODEL_MAP:
            return OPENAI_MODEL_MAP[model]
        if model.startswith("openai/"):
            return model[len("openai/"):]
        return model

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.api_key:
            raise ProviderAuthError("OpenAI API key is not configured.", provider=self.provider_name, status_code=401)

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
                    f"OpenAI authentication failed: {response.text}",
                    provider=self.provider_name,
                    status_code=response.status_code,
                    raw_response=response.text,
                )
            elif response.status_code == 429:
                raise ProviderRateLimitError(
                    f"OpenAI rate limit / quota exceeded: {response.text}",
                    provider=self.provider_name,
                    status_code=429,
                    raw_response=response.text,
                )
            elif response.status_code >= 400:
                raise ProviderAPIError(
                    f"OpenAI API error ({response.status_code}): {response.text}",
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
                f"OpenAI request timed out after {latency_ms:.1f}ms: {exc}",
                provider=self.provider_name,
            ) from exc
        except httpx.RequestError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            raise ProviderAPIError(
                f"Network transport error calling OpenAI: {exc}",
                provider=self.provider_name,
            ) from exc
