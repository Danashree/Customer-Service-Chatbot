"""
Task 4 Phase 6: Response Tone Controller.

Maps sentiment and risk levels to appropriate customer-service response tones.

Critical rule:
  The sentiment analysis and tone controller adjust RESPONSE TONE ONLY.
  They must NEVER change:
    - refund policy
    - payment policy
    - course policy
    - account policy
    - SLA policy
    - business rules
    - eligibility decisions
    - any existing customer-service policy
  They must NEVER invent promises or policy exceptions.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from .task4_models import (
    ResponseTone,
    RiskLevel,
    SentimentLabel,
    ToneControlResult,
)

logger = logging.getLogger(__name__)

# Base sentiment -> tone mapping
_SENTIMENT_TONE_MAP: Dict[SentimentLabel, Tuple[ResponseTone, str]] = {
    SentimentLabel.POSITIVE: (
        ResponseTone.FRIENDLY_POSITIVE,
        "Customer sentiment is positive; maintain a warm, friendly, and appreciative tone.",
    ),
    SentimentLabel.NEUTRAL: (
        ResponseTone.PROFESSIONAL_NEUTRAL,
        "Customer sentiment is neutral; respond professionally, clearly, and concisely.",
    ),
    SentimentLabel.NEGATIVE: (
        ResponseTone.EMPATHETIC_CALM,
        "Customer sentiment is negative; acknowledge frustration calmly and empathetically.",
    ),
    SentimentLabel.FRUSTRATED: (
        ResponseTone.EMPATHETIC_DEESCALATING,
        "Customer is frustrated; use de-escalating, reassuring, and solution-oriented phrasing.",
    ),
    SentimentLabel.URGENT: (
        ResponseTone.CONCISE_ACTION_FOCUSED,
        "Customer query is urgent; provide direct, immediate, and actionable guidance without unnecessary pleasantries.",
    ),
    SentimentLabel.SARCASTIC: (
        ResponseTone.CALM_PROFESSIONAL,
        "Customer is using sarcasm; maintain an entirely calm, non-confrontational, and polite stance.",
    ),
}


def get_tone(sentiment: SentimentLabel, risk_level: RiskLevel = RiskLevel.NORMAL) -> ResponseTone:
    """
    Determine the response tone given sentiment and risk level.

    Priority:
      HIGH / CRITICAL risk overrides base sentiment tone to ensure
      the response is calm, professional, and escalation-focused.
    """
    if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        return ResponseTone.CALM_ESCALATION_FOCUSED

    tone, _ = _SENTIMENT_TONE_MAP.get(
        sentiment,
        (ResponseTone.PROFESSIONAL_NEUTRAL, "Default tone"),
    )
    return tone


def evaluate_tone(
    sentiment: SentimentLabel,
    risk_level: RiskLevel = RiskLevel.NORMAL,
) -> ToneControlResult:
    """
    Evaluate and return a structured ToneControlResult with rationale.
    """
    if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        return ToneControlResult(
            tone=ResponseTone.CALM_ESCALATION_FOCUSED,
            rationale=(
                f"Risk level is {risk_level.value}; tone prioritized as calm, professional, "
                "and escalation-focused to handle sensitive security/financial/legal context."
            ),
            sentiment_input=sentiment,
            risk_input=risk_level,
        )

    tone, rationale = _SENTIMENT_TONE_MAP.get(
        sentiment,
        (ResponseTone.PROFESSIONAL_NEUTRAL, "Default neutral tone."),
    )

    return ToneControlResult(
        tone=tone,
        rationale=rationale,
        sentiment_input=sentiment,
        risk_input=risk_level,
    )
