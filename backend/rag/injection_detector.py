"""
Task 4 Phase 3 — Document-Level Prompt Injection Detector
===========================================================
Detects malicious directives embedded within knowledge-base document chunks,
ensuring document content is treated strictly as untrusted DATA and never as
executable instructions.

Key capabilities:
- Normalized, case-insensitive regex pattern detection
- Machine-readable result (is_injection, reason, risk_level, matched_patterns)
- Content sanitization: strips malicious instructions while preserving
  legitimate clean factual content if present
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .models import InjectionScanResult


# ---------------------------------------------------------------------------
# Injection signature categories & compiled rules
# ---------------------------------------------------------------------------

INJECTION_RULES: List[Tuple[str, re.Pattern, str]] = [
    # 1. Instruction override / disregard
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|system|above|earlier)\s+(?:instructions|rules|prompts|directives)\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        "instruction_override",
        re.compile(
            r"\bdisregard\s+(?:all\s+)?(?:rules|instructions|safety\s+rules|guidelines|retrieval\s+rules)\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    # 2. System prompt extraction
    (
        "system_prompt_extraction",
        re.compile(
            r"\b(?:reveal|show|display|print|leak|output|dump)\s+(?:the\s+)?(?:system\s+prompt|developer\s+instructions|hidden\s+instructions|system\s+instructions|initial\s+prompt)\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    # 3. Secret & credential disclosure
    (
        "credential_disclosure",
        re.compile(
            r"\b(?:disclose|leak|send|steal|exfiltrate|reveal)\s+(?:confidential|customer|user|credentials?|passwords?|tokens?|secrets?|api\s*keys?)\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        "credential_disclosure",
        re.compile(
            r"\bsend\s+(?:the\s+)?user(?:'s)?\s+credentials\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    # 4. Instruction hierarchy manipulation
    (
        "hierarchy_manipulation",
        re.compile(
            r"\buse\s+this\s+document\s+as\s+(?:the\s+)?highest[- ]priority\s+instruction\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        "hierarchy_manipulation",
        re.compile(
            r"\b(?:system\s+override|follow\s+these\s+instructions\s+instead)\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        "hierarchy_manipulation",
        re.compile(
            r"\b(?:you\s+are\s+now\s+in\s+developer\s+mode|act\s+as\s+(?:an?\s+)?unrestricted)\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    # 5. Retrieval & policy bypass
    (
        "policy_bypass",
        re.compile(
            r"\b(?:ignore|bypass)\s+(?:the\s+)?(?:retrieval\s+rules|access\s+control|policy\s+rules|authorization)\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    # 6. Malicious command / jailbreak
    (
        "malicious_command",
        re.compile(
            r"\b(?:jailbreak|admin\s+override|execute\s+command|run\s+shell)\b",
            re.IGNORECASE,
        ),
        "high",
    ),
]


def normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace and standardizing punctuation."""
    if not text:
        return ""
    # Normalize unicode spaces, multiple spaces/newlines
    cleaned = re.sub(r"[\r\t\f\v ]+", " ", text)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def scan_document_injection(text: str) -> InjectionScanResult:
    """
    Scans document text for malicious prompt-injection patterns.

    Returns an InjectionScanResult containing:
    - is_injection: True if any pattern matched
    - reason: primary category of injection
    - matched_patterns: strings that triggered detection
    - sanitized_content: legitimate content with injection statements stripped
    """
    if not text or not isinstance(text, str):
        return InjectionScanResult(
            is_injection=False,
            reason=None,
            risk_level="none",
            matched_patterns=[],
            categories=[],
            sanitized_content=text,
        )

    norm_text = normalize_text(text)
    matched_patterns: List[str] = []
    categories: List[str] = []
    highest_risk = "none"

    # Scan overall text
    for cat, pattern, risk in INJECTION_RULES:
        for match in pattern.finditer(norm_text):
            matched_text = match.group(0)
            if matched_text not in matched_patterns:
                matched_patterns.append(matched_text)
            if cat not in categories:
                categories.append(cat)
            if risk == "high":
                highest_risk = "high"
            elif risk == "medium" and highest_risk != "high":
                highest_risk = "medium"

    is_injection = len(matched_patterns) > 0
    primary_reason = categories[0] if categories else None

    # Line/sentence-level sanitization: preserve clean sentences
    sanitized_content: Optional[str] = None
    if is_injection:
        # Split by lines and sentence-like boundaries
        raw_lines = re.split(r"[\n.]+", text)
        clean_parts: List[str] = []
        for line in raw_lines:
            line_str = line.strip()
            if not line_str:
                continue
            line_norm = normalize_text(line_str)
            # Check if this specific line/sentence triggers injection
            line_has_injection = any(
                pattern.search(line_norm) for _, pattern, _ in INJECTION_RULES
            )
            if not line_has_injection:
                clean_parts.append(line_str)

        if clean_parts:
            sanitized_content = ". ".join(clean_parts).strip()
            if not sanitized_content.endswith("."):
                sanitized_content += "."
        else:
            sanitized_content = None
    else:
        sanitized_content = text

    return InjectionScanResult(
        is_injection=is_injection,
        reason=primary_reason,
        risk_level=highest_risk,
        matched_patterns=matched_patterns,
        categories=categories,
        sanitized_content=sanitized_content,
    )
