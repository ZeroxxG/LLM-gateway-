"""
gateway/guardrails/manager.py
=============================
Unified Security Guardrail Orchestrator for the Intelligent LLM Gateway.

Executes sub-2ms pre-flight protection across three security domains:
  1. PII Detection & Anonymization (Credit cards, SSNs, emails, phone numbers, secret keys)
  2. Prompt Injection & Jailbreak Defense (System override, DAN, delimiter hijacking)
  3. Safety & Content Moderation (Dangerous weapons, malware, self-harm, violent extremism)
"""
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple

from gateway.adapters.schemas import ChatCompletionRequest, ChatMessage
from gateway.guardrails.injection import InjectionScanResult, PromptInjectionScanner
from gateway.guardrails.moderation import ContentModerator, ModerationResult
from gateway.guardrails.pii import PIIEntity, PIIScanner


class GuardrailViolationException(Exception):
    """Exception raised when an inbound request violates gateway security policies."""

    def __init__(
        self,
        message: str,
        violations: List[str],
        action: str = "BLOCKED",
        status_code: int = 422,
        injection_risk: str = "NONE",
    ):
        super().__init__(message)
        self.message = message
        self.violations = violations
        self.action = action
        self.status_code = status_code
        self.injection_risk = injection_risk


@dataclass
class GuardrailInspectionResult:
    """Outcome of pre-flight security guardrail inspection."""

    action: str  # "PASS", "MASKED", "BLOCKED"
    is_safe: bool
    violations: List[str] = field(default_factory=list)
    detected_pii: List[PIIEntity] = field(default_factory=list)
    injection_result: Optional[InjectionScanResult] = None
    moderation_result: Optional[ModerationResult] = None
    original_messages: List[ChatMessage] = field(default_factory=list)
    sanitized_messages: List[ChatMessage] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def pii_detected_types(self) -> List[str]:
        """Unique list of detected PII types."""
        return sorted(list({e.entity_type for e in self.detected_pii}))


class SecurityGuardrailManager:
    """
    Central coordinator orchestrating PII anonymization, prompt injection defenses,
    and moderation policies with minimal latency overhead.
    """

    @classmethod
    def scan_text(
        cls,
        text: str,
        mask_pii: bool = True,
        check_injection: bool = True,
        check_moderation: bool = True,
    ) -> Dict[str, Any]:
        """
        Standalone text scanner for diagnostic checks and security auditing.
        """
        start = time.perf_counter()
        violations: List[str] = []
        is_safe = True

        # 1. PII Scan & Mask
        masked_text = text
        detected_entities: List[PIIEntity] = []
        if mask_pii:
            masked_text, detected_entities = PIIScanner.mask(text, indexed=True)

        # 2. Injection Scan
        injection_res = None
        injection_risk = "NONE"
        if check_injection:
            injection_res = PromptInjectionScanner.scan(text)
            injection_risk = injection_res.risk_level
            if injection_res.is_injected:
                is_safe = False
                for p in injection_res.detected_patterns:
                    violations.append(f"Prompt injection risk ({injection_risk}): {p}")

        # 3. Moderation Scan
        moderation_res = None
        if check_moderation:
            moderation_res = ContentModerator.scan(text)
            if not moderation_res.is_safe:
                is_safe = False
                for cat in moderation_res.flagged_categories:
                    violations.append(f"Safety policy violation ({moderation_res.severity}): {cat}")

        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)

        return {
            "is_safe": is_safe,
            "violations": violations,
            "detected_pii": [e.entity_type for e in detected_entities],
            "pii_count": len(detected_entities),
            "masked_text": masked_text,
            "injection_risk": injection_risk,
            "injection_score": injection_res.risk_score if injection_res else 0.0,
            "execution_time_ms": latency_ms,
        }

    @classmethod
    def inspect_request(
        cls,
        payload: ChatCompletionRequest,
        mask_pii: bool = True,
        check_injection: bool = True,
        check_moderation: bool = True,
        block_on_violation: bool = True,
    ) -> GuardrailInspectionResult:
        """
        Execute pre-flight security scan and sanitize inbound conversation messages.

        If `block_on_violation` is True and an injection or critical policy violation
        is found, raises `GuardrailViolationException`.
        If PII is found and `mask_pii` is True, rewrites message contents in-place.
        """
        start = time.perf_counter()
        violations: List[str] = []
        all_pii: List[PIIEntity] = []
        is_safe = True
        highest_injection_risk = "NONE"

        sanitized_messages: List[ChatMessage] = []

        # Iterate over all messages in the conversation payload
        for msg in payload.messages:
            content = msg.content

            # 1. Moderation check
            if check_moderation:
                mod_res = ContentModerator.scan(content)
                if not mod_res.is_safe:
                    is_safe = False
                    for cat in mod_res.flagged_categories:
                        violations.append(f"Prohibited content: {cat}")

            # 2. Prompt injection check
            if check_injection:
                inj_res = PromptInjectionScanner.scan(content)
                if inj_res.is_injected:
                    is_safe = False
                    highest_injection_risk = inj_res.risk_level
                    for pat in inj_res.detected_patterns:
                        violations.append(f"Prompt injection pattern: {pat}")

            # 3. PII Masking
            if mask_pii:
                masked_content, entities = PIIScanner.mask(content, indexed=True)
                all_pii.extend(entities)
                sanitized_messages.append(ChatMessage(role=msg.role, content=masked_content))
            else:
                sanitized_messages.append(ChatMessage(role=msg.role, content=content))

        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)

        # If security violations occurred and blocking is enabled, raise exception
        if not is_safe and block_on_violation:
            error_message = (
                f"Request rejected by security guardrails: {len(violations)} policy violation(s) detected. "
                f"Highest risk: {highest_injection_risk}."
            )
            raise GuardrailViolationException(
                message=error_message,
                violations=violations,
                action="BLOCKED",
                status_code=422,
                injection_risk=highest_injection_risk,
            )

        # Determine action
        if not is_safe:
            action = "VIOLATION_LOGGED"
        elif all_pii:
            action = "MASKED"
        else:
            action = "PASS"

        # Apply sanitized messages back to payload
        if mask_pii and all_pii:
            payload.messages = sanitized_messages

        return GuardrailInspectionResult(
            action=action,
            is_safe=is_safe,
            violations=violations,
            detected_pii=all_pii,
            sanitized_messages=sanitized_messages,
            latency_ms=latency_ms,
        )
