"""
Task 6 Phase 1 — Spelling Normalizer
====================================
Normalizes common customer service spelling variations and errors in non-entity
words without altering business-critical identifiers.
"""
from __future__ import annotations

import re
from typing import Dict


COMMON_TYPOS: Dict[str, str] = {
    # Order variations
    "oder": "order",
    "ordr": "order",
    "odr": "order",
    "ordrs": "orders",
    "oders": "orders",
    # Refund variations
    "refnd": "refund",
    "refun": "refund",
    "refnded": "refunded",
    "refnding": "refunding",
    "rfund": "refund",
    # Status variations
    "staus": "status",
    "statas": "status",
    "stts": "status",
    # Cancellation variations
    "cancle": "cancel",
    "cncl": "cancel",
    "cancl": "cancel",
    "canceld": "canceled",
    "cancelling": "cancelling",
    "cancelation": "cancellation",
    # Payment / billing variations
    "pyment": "payment",
    "paymnt": "payment",
    "pymnt": "payment",
    "invoce": "invoice",
    "invice": "invoice",
    # Account variations
    "acount": "account",
    "accnt": "account",
    "acct": "account",
    # Course / learning variations
    "cours": "course",
    "curse": "course",
    "certifcate": "certificate",
    "certficate": "certificate",
    "enrolmnt": "enrollment",
    # Support / assistance
    "suport": "support",
    "asistance": "assistance",
    "recieved": "received",
    "subscripton": "subscription",
}


class SpellingNormalizer:
    """Safe, non-destructive normalizer for frequent customer service typos."""

    @classmethod
    def normalize_word(cls, word: str) -> str:
        """Corrects a single word if it matches a known typo, preserving case."""
        lower = word.lower()
        if lower in COMMON_TYPOS:
            target = COMMON_TYPOS[lower]
            if word.isupper():
                return target.upper()
            elif word.istitle():
                return target.capitalize()
            return target
        return word

    @classmethod
    def normalize_text(cls, text: str, enabled: bool = True) -> str:
        """
        Normalizes typos in text while leaving placeholders and intact tokens alone.
        If enabled is False, returns text as-is.
        """
        if not text or not enabled:
            return text

        def _replace_match(match: re.Match) -> str:
            word = match.group(0)
            # Do not touch placeholder tokens
            if word.startswith("__PRESERVED_") and word.endswith("__"):
                return word
            return cls.normalize_word(word)

        # Match alphanumeric words or existing entity placeholders
        pattern = re.compile(r"__PRESERVED_[A-Z_]+_\d+__|\b[A-Za-z]+\b")
        return pattern.sub(_replace_match, text)
