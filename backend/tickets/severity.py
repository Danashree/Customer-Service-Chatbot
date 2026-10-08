"""
Task 3 Phase 2: Deterministic Severity and Customer Impact Classifiers.

Both classifiers use configurable keyword-based rules.
Rules are defined as ordered lists of (label, patterns) tuples;
the first matching rule wins (highest to lowest severity order).

Domain: online-course / e-learning customer support.
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional, Tuple

from .priority_enums import Severity, CustomerImpact

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Type alias for rule entries
# ──────────────────────────────────────────────────────────────────────────────

# (label, [keyword/phrase patterns])
SeverityRule = Tuple[Severity, List[str]]
ImpactRule = Tuple[CustomerImpact, List[str]]


# ──────────────────────────────────────────────────────────────────────────────
# Default Severity rules — ordered CRITICAL → LOW (first match wins)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SEVERITY_RULES: List[SeverityRule] = [
    (
        Severity.CRITICAL,
        [
            "security breach", "data breach", "account hacked", "unauthorized access",
            "identity theft", "fraud", "scam", "charged multiple times",
            "double charged", "money stolen", "widespread", "service down",
            "platform down", "all users affected", "critical error",
        ],
    ),
    (
        Severity.HIGH,
        [
            "major course access", "blocked from course", "blocked from completing",
            "unable to complete", "payment not processed", "payment failed repeatedly",
            "important payment issue", "important deadline", "losing progress",
            "repeated issue", "repeated unresolved", "unresolved for days",
            "escalate immediately", "urgent escalation",
        ],
    ),
    (
        Severity.MEDIUM,
        [
            "cannot access course", "can't access course", "course not accessible",
            "cannot access", "can't access", "access issue", "login issue",
            "certificate not issued", "certificate issue", "certificate",
            "payment clarification", "payment", "refund", "account issue",
            "normal account issue", "course not loading", "video not playing",
            "quiz not working", "assignment not submitted", "progress not saved",
        ],
    ),
    # LOW is the fallback (matched if nothing above fires)
]


# ──────────────────────────────────────────────────────────────────────────────
# Default Customer Impact rules — ordered CRITICAL → LOW
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_IMPACT_RULES: List[ImpactRule] = [
    (
        CustomerImpact.CRITICAL,
        [
            "security breach", "data breach", "account hacked", "fraud", "scam",
            "charged multiple times", "double charged", "money stolen",
            "widespread", "service down", "all users", "platform down",
        ],
    ),
    (
        CustomerImpact.HIGH,
        [
            "cannot access", "can't access", "paid but no access",
            "unable to complete", "blocked", "certificate not issued",
            "lost progress", "cannot submit assignment", "urgent deadline",
        ],
    ),
    (
        CustomerImpact.MEDIUM,
        [
            "video not playing", "quiz not working", "slow loading",
            "assignment issue", "progress not saved", "minor access issue",
            "login problem", "account issue", "payment clarification",
        ],
    ),
    # LOW is the fallback
]


# ──────────────────────────────────────────────────────────────────────────────
# Classifier implementation
# ──────────────────────────────────────────────────────────────────────────────

def _match_rules(text: str, rules: list, fallback) -> object:
    """
    Apply ordered rules to `text` (case-insensitive phrase matching).
    Returns the label of the first matching rule, or `fallback` if none match.
    """
    if not text:
        return fallback

    text_lower = text.lower()
    for label, patterns in rules:
        for pattern in patterns:
            if re.search(r"\b" + re.escape(pattern.lower()) + r"\b", text_lower):
                logger.debug("Matched rule %s on pattern %r", label, pattern)
                return label
    return fallback


def classify_severity(
    issue: Optional[str],
    conversation: Optional[str] = None,
    rules: Optional[List[SeverityRule]] = None,
) -> Severity:
    """
    Classify ticket severity from the issue description and/or conversation.

    Rules are evaluated in order (CRITICAL first). The first match wins.
    Falls back to LOW if no rule matches.

    Parameters
    ----------
    issue:        Extracted issue description (primary signal).
    conversation: Full conversation text (secondary signal, used if issue alone
                  doesn't match anything).
    rules:        Custom rule list; defaults to DEFAULT_SEVERITY_RULES.
    """
    active_rules = rules or DEFAULT_SEVERITY_RULES

    # Primary: issue description
    if issue:
        result = _match_rules(issue, active_rules, None)
        if result is not None:
            return result

    # Secondary: full conversation text
    if conversation:
        result = _match_rules(conversation, active_rules, None)
        if result is not None:
            return result

    return Severity.LOW


def classify_customer_impact(
    issue: Optional[str],
    conversation: Optional[str] = None,
    rules: Optional[List[ImpactRule]] = None,
) -> CustomerImpact:
    """
    Classify customer impact from the issue description and/or conversation.

    Same matching strategy as classify_severity.
    Falls back to LOW if no rule matches.
    """
    active_rules = rules or DEFAULT_IMPACT_RULES

    if issue:
        result = _match_rules(issue, active_rules, None)
        if result is not None:
            return result

    if conversation:
        result = _match_rules(conversation, active_rules, None)
        if result is not None:
            return result

    return CustomerImpact.LOW
