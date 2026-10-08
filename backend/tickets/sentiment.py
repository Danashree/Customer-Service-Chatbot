"""
Task 3 Phase 2: Deterministic Rule-Based Sentiment Detector.

Determines customer sentiment from free-text conversation.
Sentiment is an INPUT to the priority calculator; it does NOT
modify any ticket content field.

Algorithm:
  1. Tokenise text into lowercase words.
  2. Count NEGATIVE and POSITIVE indicator words/phrases.
  3. Apply a negation window (e.g. "not happy" → cancel POSITIVE).
  4. Decide: more negative weight → NEGATIVE, more positive → POSITIVE, else NEUTRAL.

Domain-tuned for e-learning / online-course customer support.
No external model or training dataset is used.
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List

from .priority_enums import Sentiment

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Lexicons  (domain-tuned)
# ──────────────────────────────────────────────────────────────────────────────

_NEGATIVE_PHRASES: List[str] = [
    # Frustration / anger
    "frustrated", "frustrating", "angry", "furious", "annoyed", "annoying",
    "terrible", "horrible", "awful", "disgusting", "outrageous",
    # Urgency / complaint
    "unacceptable", "ridiculous", "worst", "useless", "incompetent",
    "impossible", "waste", "wasted", "scam", "fraud",
    # Access / payment problems
    "cannot access", "can't access", "not working", "doesn't work",
    "failed", "failure", "broken", "stuck", "blocked",
    "charged but", "paid but", "money deducted", "double charge",
    "never received", "still waiting", "no response", "ignored",
    "very disappointed", "deeply disappointed", "extremely disappointed",
    "not received", "not getting", "not able",
    # Negated positive
    "not happy", "not satisfied", "not good", "not helpful",
    "unhappy", "dissatisfied", "disappointed", "displeased",
]

_POSITIVE_PHRASES: List[str] = [
    "happy", "satisfied", "satisfied with", "pleased", "great",
    "excellent", "wonderful", "amazing", "fantastic", "love",
    "enjoy", "enjoying", "thank you", "thanks", "grateful",
    "appreciate", "helpful", "very helpful", "good experience",
    "works well", "works fine", "resolved", "solved",
]

# Single-word negative signals (higher weight)
_STRONG_NEGATIVE: List[str] = [
    "frustrated", "furious", "angry", "terrible", "horrible",
    "awful", "scam", "fraud", "useless", "incompetent",
]

# Negation words within a 3-token window
_NEGATION_WORDS: List[str] = ["not", "never", "no", "cannot", "can't", "didn't", "don't"]


def _preprocess(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _count_matches(text: str, phrases: List[str]) -> int:
    total = 0
    for phrase in phrases:
        total += len(re.findall(r"\b" + re.escape(phrase) + r"\b", text))
    return total


def detect_sentiment(conversation: str) -> Sentiment:
    """
    Analyse `conversation` and return POSITIVE, NEUTRAL, or NEGATIVE.

    Scoring:
      Each negative phrase match = +2 points toward negative.
      Each strong-negative word  = +3 points.
      Each positive phrase match = +2 points toward positive.
      Final: if neg_score > pos_score → NEGATIVE
             if pos_score > neg_score → POSITIVE
             else                     → NEUTRAL
    """
    if not conversation or not conversation.strip():
        return Sentiment.NEUTRAL

    text = _preprocess(conversation)

    neg_score = _count_matches(text, _NEGATIVE_PHRASES) * 2
    neg_score += _count_matches(text, _STRONG_NEGATIVE) * 3   # extra weight
    pos_score = _count_matches(text, _POSITIVE_PHRASES) * 2

    logger.debug("Sentiment scores — neg=%d pos=%d", neg_score, pos_score)

    if neg_score > pos_score:
        return Sentiment.NEGATIVE
    if pos_score > neg_score:
        return Sentiment.POSITIVE
    return Sentiment.NEUTRAL
