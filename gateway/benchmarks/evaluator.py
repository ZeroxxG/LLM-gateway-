"""
gateway/benchmarks/evaluator.py
===============================
Accuracy and quality evaluation engine for the benchmark harness.

Evaluates response text against benchmark ground truth criteria,
keyword presence, and complexity constraints.
"""
from dataclasses import dataclass
import re
from typing import List, Tuple

from gateway.benchmarks.dataset import BenchmarkItem


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of evaluating a single completion response."""

    is_accurate: bool
    score: float  # 0.0 to 1.0
    matched_keywords: List[str]
    missing_keywords: List[str]
    reason: str


class BenchmarkEvaluator:
    """Evaluates completion quality and accuracy against test criteria."""

    @classmethod
    def evaluate_response(
        cls,
        item: BenchmarkItem,
        response_text: str,
        model_tier: str,
    ) -> EvaluationResult:
        """
        Evaluate if response satisfies benchmark item requirements.

        Criteria:
          1. Matches expected key terms or concepts (case-insensitive).
          2. Considers model tier capability relative to task complexity.
        """
        normalized_resp = response_text.lower()
        matched = []
        missing = []

        for kw in item.expected_keywords:
            pattern = re.escape(kw.lower())
            if re.search(pattern, normalized_resp):
                matched.append(kw)
            else:
                missing.append(kw)

        # Keyword match ratio
        keyword_ratio = len(matched) / len(item.expected_keywords) if item.expected_keywords else 1.0

        # High complexity tasks require adequate model capability
        tier_penalty = False
        if item.complexity == "high" and model_tier.lower() == "small":
            tier_penalty = True

        # Determine accuracy threshold
        # For multi-keyword items, matching at least 50% or primary keyword is acceptable
        if keyword_ratio >= 0.5 and not tier_penalty:
            is_accurate = True
            score = round(keyword_ratio, 2)
            reason = f"Matched {len(matched)}/{len(item.expected_keywords)} criteria on {model_tier} tier."
        elif tier_penalty:
            is_accurate = False
            score = round(keyword_ratio * 0.4, 2)
            reason = f"Task complexity is 'high' but routed model tier '{model_tier}' was under-provisioned."
        else:
            is_accurate = False
            score = round(keyword_ratio, 2)
            reason = f"Missing required keywords: {missing}"

        return EvaluationResult(
            is_accurate=is_accurate,
            score=score,
            matched_keywords=matched,
            missing_keywords=missing,
            reason=reason,
        )
