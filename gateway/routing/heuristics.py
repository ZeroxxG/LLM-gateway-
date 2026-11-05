"""
gateway/routing/heuristics.py
=============================
Sub-2ms Intelligent Heuristic Decision Engine.

Routes requests based on:
  1. Task Type: coding, reasoning, creative, extraction, summarization, chat.
  2. Token Context: filters models whose context window cannot fit the prompt.
  3. Latency SLA: enforces maximum allowable latency constraints.
  4. FinOps Budget: filters models exceeding request or tenant budget limits.
"""
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from gateway.adapters.schemas import ChatCompletionRequest, ChatMessage
from gateway.models import Tenant
from gateway.routing.circuit_breaker import CircuitState, circuit_breaker

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Categorized task classifications."""

    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    CHAT = "chat"


@dataclass(frozen=True)
class ModelProfile:
    """Static capability and benchmark profile for an LLM model."""

    model_id: str
    provider: str
    context_window: int
    typical_latency_ms: float
    input_cost_per_million: Decimal
    output_cost_per_million: Decimal
    tier: str
    task_affinity: Dict[TaskType, float] = field(default_factory=dict)


# In-memory model capability catalog for sub-millisecond evaluation
MODEL_CATALOG: Dict[str, ModelProfile] = {
    "groq/llama-3.1-8b": ModelProfile(
        model_id="groq/llama-3.1-8b",
        provider="groq",
        context_window=128000,
        typical_latency_ms=180.0,
        input_cost_per_million=Decimal("0.0500"),
        output_cost_per_million=Decimal("0.0800"),
        tier="small",
        task_affinity={
            TaskType.CHAT: 0.95,
            TaskType.CODING: 0.80,
            TaskType.EXTRACTION: 0.82,
            TaskType.SUMMARIZATION: 0.85,
            TaskType.REASONING: 0.72,
            TaskType.CREATIVE: 0.75,
        },
    ),
    "google/gemini-1.5-flash": ModelProfile(
        model_id="google/gemini-1.5-flash",
        provider="gemini",
        context_window=1000000,  # 1M token context
        typical_latency_ms=350.0,
        input_cost_per_million=Decimal("0.0750"),
        output_cost_per_million=Decimal("0.3000"),
        tier="small",
        task_affinity={
            TaskType.SUMMARIZATION: 0.96,  # Best for long documents
            TaskType.EXTRACTION: 0.92,
            TaskType.CHAT: 0.90,
            TaskType.CODING: 0.84,
            TaskType.REASONING: 0.85,
            TaskType.CREATIVE: 0.82,
        },
    ),
    "openai/gpt-4o-mini": ModelProfile(
        model_id="openai/gpt-4o-mini",
        provider="openai",
        context_window=128000,
        typical_latency_ms=550.0,
        input_cost_per_million=Decimal("0.1500"),
        output_cost_per_million=Decimal("0.6000"),
        tier="medium",
        task_affinity={
            TaskType.EXTRACTION: 0.94,
            TaskType.REASONING: 0.88,
            TaskType.CODING: 0.86,
            TaskType.CHAT: 0.88,
            TaskType.SUMMARIZATION: 0.88,
            TaskType.CREATIVE: 0.85,
        },
    ),
    "anthropic/claude-3-5-sonnet": ModelProfile(
        model_id="anthropic/claude-3-5-sonnet",
        provider="anthropic",
        context_window=200000,
        typical_latency_ms=1400.0,
        input_cost_per_million=Decimal("3.0000"),
        output_cost_per_million=Decimal("15.0000"),
        tier="premium",
        task_affinity={
            TaskType.CODING: 0.98,  # Industry-leading for coding
            TaskType.REASONING: 0.97,
            TaskType.CREATIVE: 0.95,
            TaskType.SUMMARIZATION: 0.92,
            TaskType.EXTRACTION: 0.92,
            TaskType.CHAT: 0.85,
        },
    ),
    "mock/test-model": ModelProfile(
        model_id="mock/test-model",
        provider="mock",
        context_window=32768,
        typical_latency_ms=15.0,
        input_cost_per_million=Decimal("0.1000"),
        output_cost_per_million=Decimal("0.2000"),
        tier="small",
        task_affinity={
            TaskType.CHAT: 0.80,
            TaskType.CODING: 0.75,
            TaskType.REASONING: 0.70,
            TaskType.CREATIVE: 0.70,
            TaskType.EXTRACTION: 0.75,
            TaskType.SUMMARIZATION: 0.75,
        },
    ),
}


class TaskClassifier:
    """Fast compiled regex classifier to detect task type in microseconds."""

    _CODE_PATTERNS = re.compile(
        r"(```|def\s+\w+|class\s+\w+|import\s+\w+|from\s+\w+\s+import|function\s*\(|"
        r"SELECT\s+.+\s+FROM|curl\s+-|docker-compose|traceback|syntaxerror|exception|"
        r"refactor|debug|api\s+endpoint|html|css|javascript|typescript|python|golang)",
        re.IGNORECASE,
    )

    _REASONING_PATTERNS = re.compile(
        r"(prove\s+that|calculate|solve\s+for|step\s*-\s*by\s*-\s*step|mathematical|"
        r"logic\s+puzzle|equation|probability|derivation|theorem|deduce|hypothesis)",
        re.IGNORECASE,
    )

    _EXTRACTION_PATTERNS = re.compile(
        r"(extract|output\s+json|return\s+as\s+json|parse|schema|csv|entities|"
        r"structured\s+data|key-value|tabular\s+format)",
        re.IGNORECASE,
    )

    _SUMMARIZATION_PATTERNS = re.compile(
        r"(summarize|summary|tl;?dr|condense|brief\s+overview|key\s+takeaways|"
        r"abstract|synopsis|bullet\s+points)",
        re.IGNORECASE,
    )

    _CREATIVE_PATTERNS = re.compile(
        r"(write\s+a\s+story|poem|creative\s+writing|brainstorm|fiction|narrative|"
        r"compose\s+a\s+song|screenplay|dialogue)",
        re.IGNORECASE,
    )

    @classmethod
    def classify_text(cls, text: str) -> TaskType:
        """Categorize prompt text in < 0.1ms using precompiled patterns."""
        sample = text[:4000]  # Examine first 4000 chars for speed

        if cls._CODE_PATTERNS.search(sample):
            return TaskType.CODING
        if cls._REASONING_PATTERNS.search(sample):
            return TaskType.REASONING
        if cls._EXTRACTION_PATTERNS.search(sample):
            return TaskType.EXTRACTION
        if cls._SUMMARIZATION_PATTERNS.search(sample):
            return TaskType.SUMMARIZATION
        if cls._CREATIVE_PATTERNS.search(sample):
            return TaskType.CREATIVE

        return TaskType.CHAT

    @classmethod
    def classify_messages(cls, messages: List[ChatMessage]) -> TaskType:
        """Classify a list of conversation messages."""
        # Focus on user and system messages
        content_parts = [m.content for m in messages if m.role in ("user", "system")]
        full_text = " ".join(content_parts)
        return cls.classify_text(full_text)


class TokenEstimator:
    """Fast prompt token estimator running in microseconds."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Approximates tokens as len(text) / 4 (standard English/code average)."""
        if not text:
            return 1
        return max(1, len(text) // 4)

    @classmethod
    def estimate_messages(cls, messages: List[ChatMessage]) -> int:
        """Calculates total estimated tokens across all messages."""
        total_chars = sum(len(m.content) for m in messages)
        return cls.estimate_tokens(" " * total_chars)


@dataclass
class CandidateEvaluation:
    """Scored evaluation result for a single model candidate."""

    model_id: str
    provider: str
    score: float
    capability_score: float
    latency_score: float
    cost_score: float
    estimated_cost_usd: Decimal
    typical_latency_ms: float
    eligible: bool
    rejection_reason: Optional[str] = None


@dataclass
class HeuristicRoutingResult:
    """Complete output of the sub-2ms heuristic routing decision."""

    selected_model: str
    detected_task_type: TaskType
    estimated_tokens: int
    decision_latency_ms: float
    candidate_chain: List[str]
    candidates: List[CandidateEvaluation]
    strategy_used: str = "heuristic"


class HeuristicRouter:
    """
    Sub-2ms decision engine.
    Scores and filters candidate models by task type, token context, latency SLA, and budget.
    """

    @classmethod
    async def evaluate_routing(
        cls,
        messages: List[ChatMessage],
        task_type_override: Optional[str] = None,
        latency_sla_ms: Optional[float] = None,
        max_cost_usd: Optional[Decimal] = None,
        tenant: Optional[Tenant] = None,
        requested_model: Optional[str] = None,
    ) -> HeuristicRoutingResult:
        """
        Executes sub-2ms decision pipeline.
        """
        start_ns = time.perf_counter_ns()

        # 1. Determine Task Type
        if task_type_override:
            try:
                task_type = TaskType(task_type_override.lower().strip())
            except ValueError:
                task_type = TaskClassifier.classify_messages(messages)
        else:
            task_type = TaskClassifier.classify_messages(messages)

        # 2. Estimate Token Count
        estimated_prompt_tokens = TokenEstimator.estimate_messages(messages)
        estimated_total_tokens = estimated_prompt_tokens + 256  # default completion buffer

        # 3. Check Tenant Remaining Budget if provided
        tenant_remaining = tenant.remaining_budget_usd if tenant else Decimal("999999.00")

        evaluations: List[CandidateEvaluation] = []

        # Batch query circuit breaker states in a single round-trip
        providers = list({p.provider for p in MODEL_CATALOG.values()})
        circuit_states = await circuit_breaker.get_all_states(providers)

        # 4. Evaluate each model in the catalog
        for model_id, profile in MODEL_CATALOG.items():
            # A. Context Window Filter
            if estimated_total_tokens > profile.context_window:
                evaluations.append(
                    CandidateEvaluation(
                        model_id=model_id,
                        provider=profile.provider,
                        score=0.0,
                        capability_score=0.0,
                        latency_score=0.0,
                        cost_score=0.0,
                        estimated_cost_usd=Decimal("0.00"),
                        typical_latency_ms=profile.typical_latency_ms,
                        eligible=False,
                        rejection_reason=(
                            f"Context window exceeded ({estimated_total_tokens} tokens > "
                            f"{profile.context_window} limit)"
                        ),
                    )
                )
                continue

            # B. Circuit Breaker Filter
            circuit_state = circuit_states.get(profile.provider, CircuitState.CLOSED)
            if circuit_state == CircuitState.OPEN:
                evaluations.append(
                    CandidateEvaluation(
                        model_id=model_id,
                        provider=profile.provider,
                        score=0.0,
                        capability_score=0.0,
                        latency_score=0.0,
                        cost_score=0.0,
                        estimated_cost_usd=Decimal("0.00"),
                        typical_latency_ms=profile.typical_latency_ms,
                        eligible=False,
                        rejection_reason=f"Circuit breaker for provider '{profile.provider}' is OPEN",
                    )
                )
                continue

            # C. Estimated Cost Calculation
            one_million = Decimal("1000000")
            prompt_cost = (Decimal(str(estimated_prompt_tokens)) * profile.input_cost_per_million) / one_million
            output_cost = (Decimal("256") * profile.output_cost_per_million) / one_million
            est_cost = (prompt_cost + output_cost).quantize(Decimal("0.00000001"))

            # D. Budget Cap Filter
            if max_cost_usd is not None and est_cost > max_cost_usd:
                evaluations.append(
                    CandidateEvaluation(
                        model_id=model_id,
                        provider=profile.provider,
                        score=0.0,
                        capability_score=0.0,
                        latency_score=0.0,
                        cost_score=0.0,
                        estimated_cost_usd=est_cost,
                        typical_latency_ms=profile.typical_latency_ms,
                        eligible=False,
                        rejection_reason=f"Estimated cost ${est_cost} exceeds query budget ${max_cost_usd}",
                    )
                )
                continue

            if est_cost > tenant_remaining and tenant_remaining > Decimal("0.000000"):
                evaluations.append(
                    CandidateEvaluation(
                        model_id=model_id,
                        provider=profile.provider,
                        score=0.0,
                        capability_score=0.0,
                        latency_score=0.0,
                        cost_score=0.0,
                        estimated_cost_usd=est_cost,
                        typical_latency_ms=profile.typical_latency_ms,
                        eligible=False,
                        rejection_reason=f"Estimated cost ${est_cost} exceeds tenant remaining balance ${tenant_remaining}",
                    )
                )
                continue

            # E. Latency SLA Filter
            if latency_sla_ms is not None and profile.typical_latency_ms > latency_sla_ms:
                evaluations.append(
                    CandidateEvaluation(
                        model_id=model_id,
                        provider=profile.provider,
                        score=0.0,
                        capability_score=0.0,
                        latency_score=0.0,
                        cost_score=0.0,
                        estimated_cost_usd=est_cost,
                        typical_latency_ms=profile.typical_latency_ms,
                        eligible=False,
                        rejection_reason=f"Typical latency {profile.typical_latency_ms}ms violates SLA {latency_sla_ms}ms",
                    )
                )
                continue

            # F. Calculate Weighted Composite Score (0.0 to 100.0)
            # 1. Capability Score (0-100) based on task affinity
            cap_affinity = profile.task_affinity.get(task_type, 0.75)
            cap_score = cap_affinity * 100.0

            # 2. Latency Score (0-100): Lower latency gives higher score
            # 15ms -> ~100, 200ms -> ~85, 500ms -> ~70, 1500ms -> ~40
            lat_score = max(10.0, 100.0 - (profile.typical_latency_ms / 20.0))

            # 3. Cost Score (0-100): Lower cost gives higher score
            # $0.05/M -> ~95, $0.15/M -> ~85, $3.00/M -> ~30
            cost_factor = float(profile.input_cost_per_million + profile.output_cost_per_million)
            cost_score = max(10.0, 100.0 - (cost_factor * 15.0))

            # Dynamic Weights based on task type:
            # - Coding/Reasoning: Higher weight on capability (50% capability, 25% latency, 25% cost)
            # - Chat: Higher weight on latency (35% capability, 45% latency, 20% cost)
            # - General: Balanced (40% capability, 30% latency, 30% cost)
            if task_type in (TaskType.CODING, TaskType.REASONING):
                w_cap, w_lat, w_cost = 0.50, 0.25, 0.25
            elif task_type == TaskType.CHAT:
                w_cap, w_lat, w_cost = 0.35, 0.45, 0.20
            else:
                w_cap, w_lat, w_cost = 0.40, 0.30, 0.30

            composite_score = round((cap_score * w_cap) + (lat_score * w_lat) + (cost_score * w_cost), 2)

            evaluations.append(
                CandidateEvaluation(
                    model_id=model_id,
                    provider=profile.provider,
                    score=composite_score,
                    capability_score=round(cap_score, 2),
                    latency_score=round(lat_score, 2),
                    cost_score=round(cost_score, 2),
                    estimated_cost_usd=est_cost,
                    typical_latency_ms=profile.typical_latency_ms,
                    eligible=True,
                )
            )

        # 5. Sort Eligible Candidates by Score Descending
        eligible_candidates = [e for e in evaluations if e.eligible]
        eligible_candidates.sort(key=lambda c: c.score, reverse=True)

        if eligible_candidates:
            selected_model = eligible_candidates[0].model_id
            candidate_chain = [c.model_id for c in eligible_candidates]
        else:
            # Fallback to safe default
            selected_model = "mock/test-model"
            candidate_chain = ["mock/test-model"]

        decision_latency_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000.0, 3)

        return HeuristicRoutingResult(
            selected_model=selected_model,
            detected_task_type=task_type,
            estimated_tokens=estimated_prompt_tokens,
            decision_latency_ms=decision_latency_ms,
            candidate_chain=candidate_chain,
            candidates=evaluations,
            strategy_used="heuristic",
        )
