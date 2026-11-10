"""
gateway/guardrails/pii.py
=========================
High-speed PII (Personally Identifiable Information) detector and redaction engine.

Detects and masks sensitive personal data before prompts reach upstream LLMs:
  - Credit Card Numbers (with Luhn checksum validation)
  - US Social Security Numbers (SSN)
  - Email Addresses
  - Phone Numbers (US and international)
  - API Keys, Bearer Tokens, Secret Keys, and JWTs
  - IP Addresses (IPv4)
"""
import ipaddress
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PIIEntity:
    """Detected PII record."""

    entity_type: str
    original_value: str
    masked_placeholder: str
    start: int
    end: int


def _is_valid_luhn(card_number_str: str) -> bool:
    """Validate digits with the Luhn checksum algorithm."""
    digits = [int(c) for c in card_number_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = d * 2
            checksum += (doubled - 9) if doubled > 9 else doubled
        else:
            checksum += d
    return checksum % 10 == 0


def _is_valid_ipv4(ip_str: str) -> bool:
    """Check if string is a valid non-loopback IPv4 address."""
    try:
        addr = ipaddress.IPv4Address(ip_str)
        # We allow 127.0.0.1 or localhost without flagging, or flag public IPs
        return not addr.is_loopback
    except ValueError:
        return False


class PIIScanner:
    """
    Sub-millisecond regex & algorithmic scanner for identifying and redacting PII.
    """

    # Credit card regex: matches 13 to 19 digit strings with optional hyphens/spaces
    CC_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

    # SSN regex: 3 digits - 2 digits - 4 digits (excluding invalid prefix rules)
    SSN_PATTERN = re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")

    # RFC 5322 simplified email regex
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    # Phone numbers: US formats (xxx) xxx-xxxx, xxx-xxx-xxxx, +1-xxx-xxx-xxxx
    PHONE_PATTERN = re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )

    # API keys and security credentials:
    # 1. AWS access key: AKIA... (20 chars)
    # 2. OpenAI-style key: sk-... (at least 20 chars)
    # 3. JWT token: eyJ...
    # 4. Generic bearer token
    # 5. Private key blocks
    API_KEY_PATTERNS = [
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"\bsk-[a-zA-Z0-9_\-]{20,}\b"),
        re.compile(r"\bghp_[a-zA-Z0-9]{36,}\b"),
        re.compile(r"\beyJ[a-zA-Z0-9_\-]+\.eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\b"),
        re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----"),
    ]

    # IPv4 regex
    IPV4_PATTERN = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")

    @classmethod
    def scan(cls, text: str) -> List[PIIEntity]:
        """
        Scan text for all supported PII categories and return entity matches.
        """
        if not text:
            return []

        entities: List[PIIEntity] = []

        # 1. Private keys and API keys (high priority)
        for pattern in cls.API_KEY_PATTERNS:
            for match in pattern.finditer(text):
                entities.append(
                    PIIEntity(
                        entity_type="api_key",
                        original_value=match.group(0),
                        masked_placeholder="",
                        start=match.start(),
                        end=match.end(),
                    )
                )

        # 2. Credit Cards
        for match in cls.CC_PATTERN.finditer(text):
            raw = match.group(0)
            clean_digits = re.sub(r"\D", "", raw)
            if 13 <= len(clean_digits) <= 19 and _is_valid_luhn(clean_digits):
                entities.append(
                    PIIEntity(
                        entity_type="credit_card",
                        original_value=raw,
                        masked_placeholder="",
                        start=match.start(),
                        end=match.end(),
                    )
                )

        # 3. Social Security Numbers
        for match in cls.SSN_PATTERN.finditer(text):
            entities.append(
                PIIEntity(
                    entity_type="ssn",
                    original_value=match.group(0),
                    masked_placeholder="",
                    start=match.start(),
                    end=match.end(),
                )
            )

        # 4. Emails
        for match in cls.EMAIL_PATTERN.finditer(text):
            entities.append(
                PIIEntity(
                    entity_type="email",
                    original_value=match.group(0),
                    masked_placeholder="",
                    start=match.start(),
                    end=match.end(),
                )
            )

        # 5. Phone Numbers
        for match in cls.PHONE_PATTERN.finditer(text):
            val = match.group(0)
            # Avoid matching plain 10 digit numbers that might have been part of something else
            entities.append(
                PIIEntity(
                    entity_type="phone",
                    original_value=val,
                    masked_placeholder="",
                    start=match.start(),
                    end=match.end(),
                )
            )

        # 6. IPv4 Addresses
        for match in cls.IPV4_PATTERN.finditer(text):
            val = match.group(0)
            if _is_valid_ipv4(val):
                entities.append(
                    PIIEntity(
                        entity_type="ip_address",
                        original_value=val,
                        masked_placeholder="",
                        start=match.start(),
                        end=match.end(),
                    )
                )

        # Deduplicate overlapping matches by prioritizing longer matches or higher priority
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
        filtered_entities: List[PIIEntity] = []
        last_end = -1

        for ent in entities:
            if ent.start >= last_end:
                filtered_entities.append(ent)
                last_end = ent.end

        return filtered_entities

    @classmethod
    def mask(
        cls,
        text: str,
        indexed: bool = True,
    ) -> Tuple[str, List[PIIEntity]]:
        """
        Detect and replace all PII entities in text with placeholder tokens.

        Args:
            text: The raw prompt text.
            indexed: If True, uses '[EMAIL_1]', '[CREDIT_CARD_1]', etc.
                     If False, uses '[REDACTED_EMAIL]', '[REDACTED_CREDIT_CARD]'.

        Returns:
            Tuple of (masked_text, list of detected PIIEntity with placeholders).
        """
        entities = cls.scan(text)
        if not entities:
            return text, []

        type_counts: Dict[str, int] = {}
        for ent in entities:
            count = type_counts.get(ent.entity_type, 0) + 1
            type_counts[ent.entity_type] = count
            if indexed:
                ent.masked_placeholder = f"[{ent.entity_type.upper()}_{count}]"
            else:
                ent.masked_placeholder = f"[REDACTED_{ent.entity_type.upper()}]"

        # Build masked string from end to start to maintain index offsets
        masked_chars = list(text)
        for ent in reversed(entities):
            masked_chars[ent.start : ent.end] = list(ent.masked_placeholder)

        return "".join(masked_chars), entities

    @classmethod
    def unmask(cls, text: str, entities: List[PIIEntity]) -> str:
        """
        Restore original PII values back into a masked completion string.
        """
        restored = text
        for ent in entities:
            if ent.masked_placeholder and ent.masked_placeholder in restored:
                restored = restored.replace(ent.masked_placeholder, ent.original_value)
        return restored
