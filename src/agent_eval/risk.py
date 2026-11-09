"""Deterministic safety-risk classifier that runs before any LLM call.

Kept rule-based on purpose: it is fast, cheap, explainable and fully testable,
which matters more for a guardrail than squeezing out the last bit of recall.
"""

from __future__ import annotations

import re
from enum import StrEnum


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_HIGH_PATTERNS = [
    r"\b(kill|hurt|harm)\s+(myself|me|someone|him|her|them)\b",
    r"\bsuicid\w*",
    r"\bself[- ]?harm\b",
    r"\b(make|build)\s+(a\s+)?(bomb|explosive|weapon)\b",
    r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
    r"\b(reveal|print|show)\s+(your\s+)?(system\s+prompt|hidden\s+instructions)\b",
]

_MEDIUM_PATTERNS = [
    r"\bpassword\b",
    r"\bapi[_ ]?key\b",
    r"\bcredit\s+card\b",
    r"\bdelete\s+(all|every)\b",
    r"\bbypass\b",
]


def classify_risk(text: str) -> RiskLevel:
    """Classify a user message into a risk level using ordered regex rules."""
    lowered = text.lower()
    if any(re.search(p, lowered) for p in _HIGH_PATTERNS):
        return RiskLevel.HIGH
    if any(re.search(p, lowered) for p in _MEDIUM_PATTERNS):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW
