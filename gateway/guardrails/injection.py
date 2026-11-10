"""
gateway/guardrails/injection.py
===============================
Heuristic and regex pattern scanner for prompt injection, jailbreaks, and system override attacks.

Covers:
  - Instruction override / system prompt hijack (e.g. 'ignore previous instructions')
  - Delimiter attacks (e.g. '--- END SYSTEM PROMPT ---', '<|im_start|>', '</system>')
  - Persona/roleplay bypasses (e.g. DAN, developer mode, unrestricted mode)
  - Exfiltration and leakage traps (e.g. markdown image leakage, repeat prompt verbatim)
  - Encoded payload inspection (Base64 decoded inspection)
"""
import base64
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class InjectionScanResult:
    """Result of prompt injection analysis."""

    is_injected: bool
    risk_score: float  # 0.0 to 1.0
    risk_level: str  # NONE, LOW, MEDIUM, HIGH
    detected_patterns: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class PromptInjectionScanner:
    """
    Sub-millisecond pattern and heuristic classifier for prompt injection.
    """

    # High-confidence injection patterns (weight = 0.5 to 0.8 each)
    HIGH_RISK_PATTERNS = [
        (
            re.compile(
                r"\b(?:ignore|disregard|forget|override|cancel)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|prompts|directives|rules|guidelines)\b",
                re.IGNORECASE,
            ),
            "INSTRUCTION_OVERRIDE_DIRECTIVE",
            0.85,
        ),
        (
            re.compile(
                r"\b(?:you\s+are\s+now|act\s+as)\s+(?:dan|developer\s+mode|unrestricted|jailbreak|evil\s+bot|evil\s+confidant)\b",
                re.IGNORECASE,
            ),
            "JAILBREAK_PERSONA_OVERRIDE",
            0.85,
        ),
        (
            re.compile(
                r"(?:---+\s*END\s+(?:OF\s+)?SYSTEM\s+PROMPT\s*---+|<\/system>|<\|im_start\|>system|\[SYSTEM\]\s*:)",
                re.IGNORECASE,
            ),
            "SYSTEM_DELIMITER_HIJACK",
            0.80,
        ),
        (
            re.compile(
                r"\b(?:print|repeat|output|show|reveal)\s+(?:the\s+)?(?:exact\s+)?(?:initial|original|full|system)\s+(?:prompt|instructions|directive)\s+(?:verbatim|word\s+for\s+word|above)\b",
                re.IGNORECASE,
            ),
            "PROMPT_LEAK_EXFILTRATION",
            0.75,
        ),
        (
            re.compile(
                r"!\[.*?\]\(https?:\/\/[^\s)]+\?[^\s)]*(?:key|token|secret|leak|exfil|data)=[^\s)]*\)",
                re.IGNORECASE,
            ),
            "MARKDOWN_IMAGE_EXFILTRATION",
            0.80,
        ),
    ]

    # Medium-confidence patterns (weight = 0.25 to 0.4 each)
    MEDIUM_RISK_PATTERNS = [
        (
            re.compile(
                r"\b(?:bypass|disable|turn\s+off|remove)\s+(?:safety|ethical|content)\s+(?:filters|guardrails|protocols|restrictions)\b",
                re.IGNORECASE,
            ),
            "SAFETY_FILTER_BYPASS_ATTEMPT",
            0.45,
        ),
        (
            re.compile(
                r"\b(?:simulate|pretend|hypothetically)\s+(?:you\s+have\s+no\s+(?:rules|restrictions|limits)|an\s+unfiltered\s+ai)\b",
                re.IGNORECASE,
            ),
            "HYPOTHETICAL_UNFILTERED_OVERRIDE",
            0.40,
        ),
        (
            re.compile(
                r"\b(?:for\s+educational\s+purposes\s+only|in\s+a\s+fictional\s+setting),\s*(?:how\s+to\s+hack|generate\s+exploit)\b",
                re.IGNORECASE,
            ),
            "EDUCATIONAL_PRETEXT_ATTACK",
            0.35,
        ),
    ]

    # Regex for potential base64 strings with reasonable length (min 16 chars)
    BASE64_PATTERN = re.compile(r"\b[A-Za-z0-9+/]{16,}={0,2}\b")

    @classmethod
    def scan(cls, text: str) -> InjectionScanResult:
        """
        Evaluate text for prompt injection, jailbreaks, and delimiter manipulation.
        """
        if not text or not text.strip():
            return InjectionScanResult(
                is_injected=False,
                risk_score=0.0,
                risk_level="NONE",
            )

        detected_patterns: List[str] = []
        raw_score = 0.0

        # 1. Check high-risk patterns
        for pattern, label, weight in cls.HIGH_RISK_PATTERNS:
            if pattern.search(text):
                detected_patterns.append(label)
                raw_score += weight

        # 2. Check medium-risk patterns
        for pattern, label, weight in cls.MEDIUM_RISK_PATTERNS:
            if pattern.search(text):
                detected_patterns.append(label)
                raw_score += weight

        # 3. Base64 decoded payload scan
        for match in cls.BASE64_PATTERN.finditer(text):
            candidate = match.group(0)
            try:
                decoded_bytes = base64.b64decode(candidate)
                decoded_text = decoded_bytes.decode("utf-8", errors="ignore").lower()
                # Check if decoded payload contains high-risk triggers
                for pattern, label, weight in cls.HIGH_RISK_PATTERNS:
                    if pattern.search(decoded_text):
                        detected_patterns.append(f"BASE64_OBFUSCATED_{label}")
                        raw_score += 0.85
                        break
            except Exception:
                pass

        # Normalize score between 0.0 and 1.0
        risk_score = round(min(1.0, raw_score), 3)

        if risk_score >= 0.70:
            risk_level = "HIGH"
            is_injected = True
        elif risk_score >= 0.40:
            risk_level = "MEDIUM"
            is_injected = True
        elif risk_score >= 0.20:
            risk_level = "LOW"
            is_injected = False
        else:
            risk_level = "NONE"
            is_injected = False

        return InjectionScanResult(
            is_injected=is_injected,
            risk_score=risk_score,
            risk_level=risk_level,
            detected_patterns=detected_patterns,
            details={"raw_weight": raw_score, "pattern_count": len(detected_patterns)},
        )
