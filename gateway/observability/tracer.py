"""
gateway/observability/tracer.py
===============================
Distributed tracing and telemetry integration.

Generates unique trace IDs, attaches trace contexts, optionally exports
traces and generations to Langfuse if configured, and emits structured logs.
"""
import logging
import uuid
from typing import Any, Dict, Optional

from django.conf import settings

logger = logging.getLogger(__name__)


class GatewayTracer:
    """Coordinates distributed tracing and optional Langfuse synchronization."""

    def __init__(self):
        self._langfuse_client = None
        self._init_langfuse()

    def _init_langfuse(self) -> None:
        """Initialize Langfuse client if API keys are configured in Django settings."""
        public_key = getattr(settings, "LANGFUSE_PUBLIC_KEY", "")
        secret_key = getattr(settings, "LANGFUSE_SECRET_KEY", "")
        host = getattr(settings, "LANGFUSE_HOST", "https://cloud.langfuse.com")

        if public_key and secret_key:
            try:
                from langfuse import Langfuse
                self._langfuse_client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info("Langfuse telemetry client initialized successfully.")
            except Exception as exc:
                logger.warning("Could not initialize Langfuse client: %s", exc)
                self._langfuse_client = None
        else:
            self._langfuse_client = None

    @property
    def is_langfuse_enabled(self) -> bool:
        """Whether Langfuse client is initialized and active."""
        return self._langfuse_client is not None

    def generate_trace_id(self) -> str:
        """Generate a random unique trace ID with 'tr_' prefix."""
        return f"tr_{uuid.uuid4().hex[:16]}"

    async def export_trace(
        self,
        trace_id: str,
        tenant_id: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost_usd: float,
        status: str = "success",
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record trace and generation.
        Exports to Langfuse if configured and emits structured log.
        """
        meta = metadata or {}
        trace_data = {
            "trace_id": trace_id,
            "tenant_id": str(tenant_id),
            "model": model,
            "provider": provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": str(cost_usd),
            "status": status,
            "error": error,
            "metadata": meta,
        }

        # Log trace info
        logger.info(
            "Trace %s: %s/%s completed in %.2fms ($%s) - status: %s",
            trace_id,
            provider,
            model,
            latency_ms,
            cost_usd,
            status,
            extra=trace_data,
        )

        # Export to Langfuse if enabled
        if self._langfuse_client:
            try:
                trace = self._langfuse_client.trace(
                    id=trace_id,
                    name="chat_completion",
                    user_id=str(tenant_id),
                    metadata=meta,
                )
                trace.generation(
                    name=f"{provider}/{model}",
                    model=model,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    level="ERROR" if status != "success" else "DEFAULT",
                    status_message=error,
                )
            except Exception as exc:
                logger.warning("Error pushing trace to Langfuse: %s", exc)


# Singleton tracer instance
gateway_tracer = GatewayTracer()
