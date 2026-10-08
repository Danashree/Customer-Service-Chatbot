"""
Task 4 Phase 1: Conversation-level sentiment analyser.

Analyses the current message AND available conversation history to produce
a holistic ConversationSentimentResult.

Key behaviours:
  - Analyses current message separately via sentiment_multilingual.analyse_message()
  - Analyses each history message individually
  - Counts repeated negative/frustrated messages
  - If history is unavailable, analyses current message only
  - Repeated negative interactions flag `repeated_negative = True`
  - The overall sentiment considers trends, not just the last message
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .task4_models import (
    AnalysisMethod,
    ConversationSentimentResult,
    MessageSentimentResult,
    SentimentLabel,
    SentimentAnalysisResponse,
    ResponseTone,
)
from .sentiment_multilingual import analyse_message
from .tone_controller import get_tone  # late import to avoid circular

logger = logging.getLogger(__name__)

# Labels considered "negative-leaning" for repeated-negative detection
_NEGATIVE_LABELS = {
    SentimentLabel.NEGATIVE,
    SentimentLabel.FRUSTRATED,
    SentimentLabel.URGENT,
}

# Minimum consecutive negative customer messages to set repeated_negative flag
DEFAULT_NEGATIVE_THRESHOLD = 2


def analyse_conversation(
    current_message: str,
    conversation_history: Optional[List[str]] = None,
    negative_threshold: int = DEFAULT_NEGATIVE_THRESHOLD,
) -> SentimentAnalysisResponse:
    """
    Produce a full SentimentAnalysisResponse for a message + optional history.

    Parameters
    ----------
    current_message : str
        The latest customer message.
    conversation_history : list[str] | None
        Prior customer messages (oldest first). If None or empty, only the
        current message is analysed.
    negative_threshold : int
        How many consecutive negative/frustrated messages trigger
        `repeated_negative = True`. Default is 2.

    Returns
    -------
    SentimentAnalysisResponse
        message_analysis   — analysis of current_message only
        conversation_analysis — holistic analysis over history + current
        tone_recommendation — response tone based on worst-case sentiment
    """
    from .tone_controller import get_tone
    from .task4_models import RiskLevel

    # ── Analyse current message ────────────────────────────────────────────────
    msg_result: MessageSentimentResult = analyse_message(current_message)

    # ── Build conversation analysis ────────────────────────────────────────────
    history = conversation_history or []
    all_messages = list(history) + [current_message]

    # Analyse each message individually
    individual_results: List[MessageSentimentResult] = [
        analyse_message(m) for m in all_messages
    ]

    def _to_label(s: Any) -> SentimentLabel:
        if isinstance(s, SentimentLabel):
            return s
        try:
            return SentimentLabel(str(s))
        except ValueError:
            return SentimentLabel.NEUTRAL

    total = len(individual_results)
    neg_count = sum(
        1 for r in individual_results
        if _to_label(r.sentiment) in _NEGATIVE_LABELS
    )

    # Repeated-negative: count consecutive negatives from the end
    consecutive_neg = 0
    for r in reversed(individual_results):
        if _to_label(r.sentiment) in _NEGATIVE_LABELS:
            consecutive_neg += 1
        else:
            break

    repeated_negative = consecutive_neg >= negative_threshold

    # Overall sentiment: use the most severe recent signal
    # Priority: SARCASTIC > URGENT > FRUSTRATED > NEGATIVE > NEUTRAL > POSITIVE
    priority_order = [
        SentimentLabel.SARCASTIC,
        SentimentLabel.URGENT,
        SentimentLabel.FRUSTRATED,
        SentimentLabel.NEGATIVE,
        SentimentLabel.NEUTRAL,
        SentimentLabel.POSITIVE,
    ]

    # Consider last 3 messages for overall (avoid dilution by old positives)
    recent = individual_results[-3:] if len(individual_results) >= 3 else individual_results
    overall = SentimentLabel.NEUTRAL
    for candidate_label in priority_order:
        if any(_to_label(r.sentiment) == candidate_label for r in recent):
            overall = candidate_label
            break


    # Confidence: average of recent results
    if recent:
        avg_conf = sum(r.confidence for r in recent) / len(recent)
    else:
        avg_conf = msg_result.confidence

    # Aggregate flags
    frustration = any(r.frustration for r in recent) or overall == SentimentLabel.FRUSTRATED
    urgency = any(r.urgency for r in recent) or overall == SentimentLabel.URGENT
    sarcasm = any(r.sarcasm for r in recent) or overall == SentimentLabel.SARCASTIC

    # Language: from current message
    language = msg_result.language

    # Method: ML if any message used ML; RULE_BASED otherwise
    methods = {r.analysis_method for r in individual_results}
    method = (
        AnalysisMethod.ML_MODEL
        if AnalysisMethod.ML_MODEL in methods
        else AnalysisMethod.RULE_BASED
    )

    conv_result = ConversationSentimentResult(
        overall_sentiment=overall,
        confidence=round(avg_conf, 4),
        frustration=frustration,
        urgency=urgency,
        sarcasm=sarcasm,
        repeated_negative=repeated_negative,
        negative_message_count=neg_count,
        total_messages_analysed=total,
        language=language,
        analysis_method=method,
    )

    # ── Tone recommendation ────────────────────────────────────────────────────
    tone = get_tone(overall, RiskLevel.NORMAL)

    return SentimentAnalysisResponse(
        message_analysis=msg_result,
        conversation_analysis=conv_result,
        tone_recommendation=tone,
    )
