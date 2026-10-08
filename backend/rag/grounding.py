"""
Task 4 Phase 3 — Grounding, Unsupported Claim Detection & Evidence Sufficiency
================================================================================
Evaluates whether sufficient, unambiguous evidence exists to answer a query,
and verifies that factual claims in candidate answers are strictly grounded in
retrieved evidence without hallucinating prices, dates, deadlines, or policy terms.

Key guarantees:
- Distinguishes SUPPORTED, INSUFFICIENT_EVIDENCE, AMBIGUOUS_EVIDENCE, UNSUPPORTED_CLAIM.
- Detects hallucinated prices, dates, percentages, refund windows, and terms.
- Deterministic, rule-based verification suitable for regression testing.
"""
from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from .models import (
    ClaimStatus,
    ClaimVerificationResult,
    DocumentType,
    EvidenceSufficiency,
    RetrievedChunk,
)


# ---------------------------------------------------------------------------
# Factual value extraction patterns (prices, dates, deadlines, percentages)
# ---------------------------------------------------------------------------

PATTERNS_VALUES = [
    # Currency / prices: $50, $100.50, 50 USD, 50 EUR, Rs 500
    re.compile(r"\$\s*\d+(?:\.\d{2})?|\b\d+\s*(?:dollars|usd|eur|euros|inr|gbp)\b", re.IGNORECASE),
    # Percentages: 100%, 50 percent
    re.compile(r"\b\d+%(?!\w)|\b\d+\s*percent\b", re.IGNORECASE),
    # Deadlines & time periods: 7 days, 14-day, 24 hours, 30 days, within 7 days
    re.compile(r"\b(?:within\s+)?\d+[- ](?:days?|weeks?|months?|hours?|years?)\b", re.IGNORECASE),
    # Exact ISO dates: 2026-03-15
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # Month Day, Year: March 15, 2026 or 15 March 2026
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\b",
        re.IGNORECASE,
    ),
]


def extract_factual_values(text: str) -> List[str]:
    """
    Extracts specific numerical, financial, temporal, or date assertions from text.
    """
    if not text:
        return []
    values: List[str] = []
    # Ignore citation lines when extracting values to avoid false claim checking on file metadata
    lines = [
        line for line in text.splitlines()
        if not re.match(r"(?i)^\s*(?:Source:|\[(?:FAQ|POLICY|HOW_TO|TROUBLESHOOTING|GENERAL)\])", line)
    ]
    cleaned_text = "\n".join(lines)

    for pattern in PATTERNS_VALUES:
        for match in pattern.finditer(cleaned_text):
            val = match.group(0).strip()
            if val not in values:
                values.append(val)
    return values


def extract_claims(text: str) -> List[str]:
    """
    Splits text into individual factual sentences or claims, excluding citations.
    """
    if not text:
        return []
    # Strip citation lines
    lines = [
        line for line in text.splitlines()
        if not re.match(r"(?i)^\s*(?:Source:|\[(?:FAQ|POLICY|HOW_TO|TROUBLESHOOTING|GENERAL)\])", line)
    ]
    body = " ".join(lines).strip()
    if not body:
        return []

    # Split by sentence end markers
    raw_sentences = re.split(r"(?<=[.!?])\s+", body)
    claims: List[str] = []
    for s in raw_sentences:
        s_clean = s.strip()
        if len(s_clean) > 5 and not s_clean.lower().startswith("source:"):
            claims.append(s_clean)
    return claims


def normalize_value(val: str) -> str:
    """Normalizes value string for matching (collapses spaces, removes hyphens)."""
    return re.sub(r"[\s\-]+", " ", val.lower().strip())


# ---------------------------------------------------------------------------
# Evidence sufficiency evaluator
# ---------------------------------------------------------------------------

def evaluate_evidence_sufficiency(
    query: str,
    chunks: List[RetrievedChunk],
    product: Optional[str] = None,
    region: Optional[str] = None,
) -> Tuple[EvidenceSufficiency, Optional[str]]:
    """
    Evaluates whether the retrieved chunks provide sufficient and unambiguous
    evidence to answer `query`.

    Returns (status, refusal_or_clarification_message).
    If status == SUPPORTED, message is None.
    """
    if not chunks:
        return (
            EvidenceSufficiency.INSUFFICIENT_EVIDENCE,
            "I don't have enough information in the available knowledge base to answer that accurately.",
        )

    # Check for ambiguous evidence:
    # If the user did not specify product or region, and retrieved evidence contains
    # conflicting policies from multiple distinct products or distinct regions:
    policy_chunks = [
        c for c in chunks
        if c.metadata.document_type == DocumentType.POLICY or "policy" in c.metadata.source_file.lower()
    ]
    if len(policy_chunks) >= 2:
        distinct_products = {
            c.metadata.product for c in policy_chunks
            if c.metadata.product != "global"
        }
        distinct_regions = {
            c.metadata.region for c in policy_chunks
            if c.metadata.region != "global"
        }
        # If user didn't specify product and retrieved policies are for multiple products
        if product is None and len(distinct_products) > 1:
            return (
                EvidenceSufficiency.AMBIGUOUS_EVIDENCE,
                "I found conflicting information for this request across multiple products. "
                "Could you provide the product so I can identify the applicable policy?",
            )
        # If user didn't specify region and retrieved policies are for multiple regions
        if region is None and len(distinct_regions) > 1:
            return (
                EvidenceSufficiency.AMBIGUOUS_EVIDENCE,
                "I found conflicting information for this request across multiple regions. "
                "Could you provide the region so I can identify the applicable policy?",
            )

    return (EvidenceSufficiency.SUPPORTED, None)


# ---------------------------------------------------------------------------
# Grounding & claim verification
# ---------------------------------------------------------------------------

def verify_claims(
    answer: str,
    evidence_chunks: List[RetrievedChunk],
) -> ClaimVerificationResult:
    """
    Verifies that all factual claims in `answer` are supported by `evidence_chunks`.
    Detects hallucinated values (prices, dates, deadlines, percentages).
    """
    if not answer or not answer.strip():
        return ClaimVerificationResult(
            status=ClaimStatus.UNSUPPORTED,
            is_grounded=False,
            claims=[],
            supported_claims=[],
            unsupported_claims=[],
            hallucinated_values=[],
        )

    # 1. Build unified normalized evidence text
    evidence_text = " ".join(c.content for c in evidence_chunks)
    evidence_norm = normalize_value(evidence_text)

    # 2. Extract and check factual values
    answer_values = extract_factual_values(answer)
    hallucinated_values: List[str] = []

    for val in answer_values:
        val_norm = normalize_value(val)
        # Direct check
        if val_norm not in evidence_norm:
            # Check numbers: e.g. "7 days" vs "7-day"
            nums = re.findall(r"\d+", val_norm)
            if not all(num in evidence_norm for num in nums):
                hallucinated_values.append(val)

    # 3. Check individual claims / sentences
    claims = extract_claims(answer)
    supported_claims: List[str] = []
    unsupported_claims: List[str] = []

    for claim in claims:
        claim_norm = normalize_value(claim)
        # If this sentence contains any hallucinated value, it is unsupported
        sentence_has_hallucination = any(
            normalize_value(hv) in claim_norm for hv in hallucinated_values
        )
        if sentence_has_hallucination:
            unsupported_claims.append(claim)
            continue

        # Check keyword presence from the claim in evidence
        # Filter stopwords
        words = re.findall(r"[a-z0-9]{3,}", claim_norm)
        stopwords = {
            "the", "and", "for", "with", "this", "that", "from", "are", "you",
            "can", "will", "have", "has", "our", "all", "any", "not", "source",
            "page", "version", "available", "please", "contact",
        }
        content_words = [w for w in words if w not in stopwords]

        if not content_words:
            # Short generic phrase
            supported_claims.append(claim)
            continue

        # Calculate word overlap ratio with evidence
        present_count = sum(1 for w in content_words if w in evidence_norm)
        overlap = present_count / len(content_words)

        if overlap >= 0.5:
            supported_claims.append(claim)
        else:
            unsupported_claims.append(claim)

    is_grounded = len(hallucinated_values) == 0 and len(unsupported_claims) == 0
    status = ClaimStatus.SUPPORTED if is_grounded else ClaimStatus.UNSUPPORTED

    return ClaimVerificationResult(
        status=status,
        is_grounded=is_grounded,
        claims=claims,
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        hallucinated_values=hallucinated_values,
    )
