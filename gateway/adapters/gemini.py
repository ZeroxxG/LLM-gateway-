"""
gateway/adapters/gemini.py
==========================
Google Gemini Provider Adapter.

Executes asynchronous completions against Google's Gemini v1beta REST API.
Translates between OpenAI-standard message structures and Gemini's content/parts schema.
Endpoint: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
"""
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List
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

GEMINI_MODEL_MAP: Dict[str, str] = {
    "google/gemini-1.5-flash": "gemini-1.5-flash",
    "google/gemini-1.5-pro": "gemini-1.5-pro",
    "google/gemini-2.0-flash": "gemini-2.0-flash",
    "gemini/gemini-1.5-flash": "gemini-1.5-flash",
}


class GeminiAdapter(BaseProviderAdapter):
    """Adapter for Google Gemini API."""

    provider_name: str = "gemini"
    base_endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or getattr(settings, "GEMINI_API_KEY", "")

    def resolve_model(self, model: str) -> str:
        """Resolve canonical gateway model identifier to Gemini model string."""
        if model in GEMINI_MODEL_MAP:
            return GEMINI_MODEL_MAP[model]
        if model.startswith("google/"):
            return model[len("google/"):]
        if model.startswith("gemini/"):
            return model[len("gemini/"):]
        return model

    def _convert_messages(self, messages: List[ChatMessage]) -> tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
        """
        Translates OpenAI messages format to Gemini systemInstruction and contents schema.

        OpenAI:
          role: "system", content: "..."
          role: "user", content: "..."
          role: "assistant", content: "..."

        Gemini:
          systemInstruction: {"parts": [{"text": "..."}]}
          contents: [
            {"role": "user", "parts": [{"text": "..."}]},
            {"role": "model", "parts": [{"text": "..."}]}
          ]
        """
        system_instruction = None
        contents: List[Dict[str, Any]] = []

        system_parts = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                gemini_role = "user" if msg.role == "user" else "model"
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": msg.content}],
                })

        if system_parts:
            system_instruction = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }

        return system_instruction, contents

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.api_key:
            raise ProviderAuthError("Gemini API key is not configured.", provider=self.provider_name, status_code=401)

        upstream_model = self.resolve_model(request.model)
        url = self.base_endpoint.format(model=upstream_model)

        system_instruction, contents = self._convert_messages(request.messages)

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
                "topP": request.top_p,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = system_instruction

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        client = await get_shared_http_client()
        start_time = time.perf_counter()

        try:
            response = await client.post(url, json=payload, headers=headers)
            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 400 and "API_KEY_INVALID" in response.text:
                raise ProviderAuthError(
                    f"Gemini API key is invalid: {response.text}",
                    provider=self.provider_name,
                    status_code=401,
                    raw_response=response.text,
                )
            elif response.status_code == 429:
                raise ProviderRateLimitError(
                    f"Gemini rate limit exceeded: {response.text}",
                    provider=self.provider_name,
                    status_code=429,
                    raw_response=response.text,
                )
            elif response.status_code >= 400:
                raise ProviderAPIError(
                    f"Gemini API error ({response.status_code}): {response.text}",
                    provider=self.provider_name,
                    status_code=response.status_code,
                    raw_response=response.text,
                )

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ProviderAPIError("Gemini returned empty candidates.", provider=self.provider_name, raw_response=response.text)

            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            content_text = "".join(part.get("text", "") for part in parts)
            finish_reason = candidate.get("finishReason", "stop").lower()

            usage_meta = data.get("usageMetadata", {})
            prompt_tokens = usage_meta.get("promptTokenCount", 0)
            completion_tokens = usage_meta.get("candidatesTokenCount", 0)
            total_tokens = usage_meta.get("totalTokenCount", prompt_tokens + completion_tokens)

            return ChatCompletionResponse(
                model=request.model,
                provider=self.provider_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=content_text,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
                latency_ms=round(latency_ms, 2),
            )

        except httpx.TimeoutException as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            raise ProviderTimeoutError(
                f"Gemini request timed out after {latency_ms:.1f}ms: {exc}",
                provider=self.provider_name,
            ) from exc
        except httpx.RequestError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            raise ProviderAPIError(
                f"Network transport error calling Gemini: {exc}",
                provider=self.provider_name,
            ) from exc

    async def stream_complete(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Streams chat completion tokens from Gemini via SSE."""
        if not self.api_key:
            raise ProviderAuthError("Gemini API key is not configured.", provider=self.provider_name, status_code=401)

        upstream_model = self.resolve_model(request.model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{upstream_model}:streamGenerateContent?alt=sse"

        system_instruction, contents = self._convert_messages(request.messages)
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
                "topP": request.top_p,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        client = await get_shared_http_client()

        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    raw_err = await resp.aread()
                    raise ProviderAPIError(
                        f"Gemini stream rejected ({resp.status_code}): {raw_err.decode('utf-8', errors='replace')}",
                        provider=self.provider_name,
                        status_code=resp.status_code,
                    )

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        raw_data = line[len("data: "):].strip()
                        try:
                            item = json.loads(raw_data)
                            candidates = item.get("candidates", [])
                            if not candidates:
                                continue
                            c = candidates[0]
                            parts = c.get("content", {}).get("parts", [])
                            text_piece = "".join(p.get("text", "") for p in parts)
                            finish_reason = c.get("finishReason")
                            if finish_reason:
                                finish_reason = finish_reason.lower()

                            yield ChatCompletionChunk(
                                id=f"chatcmpl-gemini-{int(time.time())}",
                                model=request.model,
                                provider=self.provider_name,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=ChatCompletionChunkDelta(content=text_piece),
                                        finish_reason=finish_reason,
                                    )
                                ],
                            )
                        except json.JSONDecodeError:
                            continue
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Gemini stream timed out: {exc}", provider=self.provider_name) from exc
        except httpx.RequestError as exc:
            raise ProviderAPIError(f"Network error on Gemini stream: {exc}", provider=self.provider_name) from exc

