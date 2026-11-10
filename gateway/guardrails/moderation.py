"""
gateway/guardrails/moderation.py
================================
Sub-millisecond content moderation and safety policy engine.

Classifies prompts against safety policy categories:
  - Dangerous Weapons & Explosives synthesis (C4, pipe bombs, sarin, ricin)
  - Cyberattack Exploitation (malware, ransomware, DDoS bots, keyloggers)
  - Self-Harm & Suicide encouragement
  - Hate Speech & Violent Extremism
"""
from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional


@dataclass
class ModerationResult:
    """Result of content moderation scan."""

    is_safe: bool
    flagged_categories: List[str] = field(default_factory=list)
    severity: str = "SAFE"  # SAFE, LOW, MEDIUM, CRITICAL
    matched_rules: List[str] = field(default_factory=list)


class ContentModerator:
    """
    Sub-millisecond rule-based safety policy validator for enterprise LLM gateways.
    """

    POLICY_RULES = [
        # Dangerous Weapons & Explosives
        (
            re.compile(
                r"\b(?:make|synthesize|build|construct|manufacture)\s+(?:a\s+)?(?:pipe\s+bomb|c4\s+explosive|dirty\s+bomb|sarin\s+gas|mustard\s+gas|ricin\s+toxin)\b",
                re.IGNORECASE,
            ),
            "DANGEROUS_WEAPONS_EXPLOSIVES",
            "CRITICAL",
        ),
        (
            re.compile(
                r"\b(?:recipe|instructions|guide)\s+for\s+(?:making|creating)\s+(?:explosives|bombs|biological\s+weapons)\b",
                re.IGNORECASE,
            ),
            "DANGEROUS_WEAPONS_EXPLOSIVES",
            "CRITICAL",
        ),
        # Cyber Exploits & Malware
        (
            re.compile(
                r"\b(?:write|code|create|generate)\s+(?:a\s+)?(?:ransomware|trojan\s+virus|keylogger|ddos\s+botnet|polymorphic\s+malware)\b",
                re.IGNORECASE,
            ),
            "CYBER_EXPLOITATION_MALWARE",
            "CRITICAL",
        ),
        (
            re.compile(
                r"\b(?:exploit\s+script\s+to\s+steal|zero-day\s+payload\s+for)\b",
                re.IGNORECASE,
            ),
            "CYBER_EXPLOITATION_MALWARE",
            "HIGH",
        ),
        # Self-Harm & Suicide
        (
            re.compile(
                r"\b(?:how\s+to\s+(?:commit\s+suicide|kill\s+myself|hang\s+myself)|painless\s+ways\s+to\s+die)\b",
                re.IGNORECASE,
            ),
            "SELF_HARM_SUICIDE",
            "CRITICAL",
        ),
        # Hate Speech & Violent Extremism
        (
            re.compile(
                r"\b(?:exterminate|eradicate|massacre)\s+(?:all\s+)?(?:members\s+of|people\s+of|jews|muslims|christians|black\s+people|white\s+people|minorities|ethnic\s+groups?)\b",
                re.IGNORECASE,
            ),
            "HATE_SPEECH_VIOLENT_EXTREMISM",
            "CRITICAL",
        ),
    ]

    @classmethod
    def scan(cls, text: str) -> ModerationResult:
        """
        Evaluate text against safety policy rules.
        """
        if not text or not text.strip():
            return ModerationResult(is_safe=True, severity="SAFE")

        flagged: List[str] = []
        matched: List[str] = []
        highest_severity = "SAFE"

        severity_rank = {"SAFE": 0, "LOW": 1, "HIGH": 2, "CRITICAL": 3}

        for pattern, category, severity in cls.POLICY_RULES:
            if pattern.search(text):
                if category not in flagged:
                    flagged.append(category)
                matched.append(f"{category}:{severity}")
                if severity_rank.get(severity, 0) > severity_rank.get(highest_severity, 0):
                    highest_severity = severity

        is_safe = len(flagged) == 0

        return ModerationResult(
            is_safe=is_safe,
            flagged_categories=flagged,
            severity=highest_severity,
            matched_rules=matched,
        )
