"""
Task 3 Phase 2: Transparent, Explainable Priority Calculator.

Scoring formula (application-specific, not industry-standard):

  Component        | LOW | MEDIUM | HIGH | CRITICAL
  ─────────────────|─────|────────|──────|─────────
  Severity         |  10 |   25   |  40  |   50
  Customer Impact  |   5 |   10   |  20  |   30

  Sentiment        | POSITIVE | NEUTRAL | NEGATIVE
                   |     0    |    5    |    10

  % SLA consumed   | < 25% | 25–49% | 50–74% | 75–99% | ≥ 100%
                   |   0   |   5    |   10   |   20   |   30

  SLA urgency bonus| IN_PROGRESS | WARNING | BREACHED
                   |      0      |    5    |   10

  Final score → Priority
    < 30  → LOW
    30–49 → MEDIUM
    50–74 → HIGH
    ≥  75 → CRITICAL

The breakdown dict maps each component to its exact point contribution.
No score is ever invented; every field is traceable to the formula above.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from .priority_enums import (
    CustomerImpact, EscalationStatus, Priority, Sentiment, Severity, SLAStatus,
)
from .business_hours import calculate_business_minutes, BusinessHoursConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Score tables — change these to tune the formula
# ──────────────────────────────────────────────────────────────────────────────

SEVERITY_SCORES: Dict[str, int] = {
    Severity.LOW.value: 10,
    Severity.MEDIUM.value: 25,
    Severity.HIGH.value: 40,
    Severity.CRITICAL.value: 50,
}

IMPACT_SCORES: Dict[str, int] = {
    CustomerImpact.LOW.value: 5,
    CustomerImpact.MEDIUM.value: 10,
    CustomerImpact.HIGH.value: 20,
    CustomerImpact.CRITICAL.value: 30,
}

SENTIMENT_SCORES: Dict[str, int] = {
    Sentiment.POSITIVE.value: 0,
    Sentiment.NEUTRAL.value: 5,
    Sentiment.NEGATIVE.value: 10,
}

SLA_URGENCY_SCORES: Dict[str, int] = {
    SLAStatus.NOT_STARTED.value: 0,
    SLAStatus.IN_PROGRESS.value: 0,
    SLAStatus.WARNING.value: 5,
    SLAStatus.BREACHED.value: 10,
    SLAStatus.RESOLVED.value: 0,
}

# % SLA consumed → waiting-time score (upper-bound inclusive)
_WAITING_TIERS: list = [
    (24.99, 0),    # < 25%
    (49.99, 5),    # 25–49%
    (74.99, 10),   # 50–74%
    (99.99, 20),   # 75–99%
    (float("inf"), 30),  # ≥ 100%
]

# Priority thresholds
_PRIORITY_THRESHOLDS: list = [
    (30, Priority.LOW),
    (50, Priority.MEDIUM),
    (75, Priority.HIGH),
    (float("inf"), Priority.CRITICAL),
]


# ──────────────────────────────────────────────────────────────────────────────
# Breakdown dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PriorityBreakdown:
    severity: int = 0
    sentiment: int = 0
    customer_impact: int = 0
    waiting_time: int = 0
    sla_urgency: int = 0

    @property
    def total(self) -> int:
        return (
            self.severity
            + self.sentiment
            + self.customer_impact
            + self.waiting_time
            + self.sla_urgency
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "severity": self.severity,
            "sentiment": self.sentiment,
            "customer_impact": self.customer_impact,
            "waiting_time": self.waiting_time,
            "sla_urgency": self.sla_urgency,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Calculator
# ──────────────────────────────────────────────────────────────────────────────

def _waiting_score(pct_consumed: float) -> int:
    for threshold, score in _WAITING_TIERS:
        if pct_consumed <= threshold:
            return score
    return 30


def _priority_from_score(score: int) -> Priority:
    for threshold, priority in _PRIORITY_THRESHOLDS:
        if score < threshold:
            return priority
    return Priority.CRITICAL


def calculate_priority(
    severity: Severity,
    sentiment: Sentiment,
    customer_impact: CustomerImpact,
    sla_status: SLAStatus,
    sla_hours: Optional[float] = None,
    sla_started_at: Optional[str] = None,
    now: Optional[datetime] = None,
    business_hours_cfg: Optional[BusinessHoursConfig] = None,
) -> tuple[Priority, int, PriorityBreakdown]:
    """
    Compute priority, priority_score, and a full breakdown.

    Parameters
    ----------
    severity:          Classified ticket severity.
    sentiment:         Detected customer sentiment.
    customer_impact:   Classified customer impact.
    sla_status:        Current SLA state.
    sla_hours:         Total SLA duration in business hours (for % consumed calc).
    sla_started_at:    ISO-format string of SLA start time.
    now:               Current time (injectable for testing); defaults to UTC now.
    business_hours_cfg: BusinessHoursConfig for business-minutes calculation.

    Returns
    -------
    (Priority, score, PriorityBreakdown)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    breakdown = PriorityBreakdown()
    breakdown.severity = SEVERITY_SCORES.get(severity.value, 10)
    breakdown.sentiment = SENTIMENT_SCORES.get(sentiment.value, 5)
    breakdown.customer_impact = IMPACT_SCORES.get(customer_impact.value, 5)
    breakdown.sla_urgency = SLA_URGENCY_SCORES.get(sla_status.value, 0)

    # Waiting-time score based on % of SLA consumed
    pct_consumed = 0.0
    if sla_hours and sla_started_at:
        try:
            started = datetime.fromisoformat(sla_started_at)
            # Make both tz-aware if needed
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)

            if business_hours_cfg:
                consumed_mins = calculate_business_minutes(started, now, business_hours_cfg)
            else:
                # Fallback: wall-clock minutes (only if no BH config)
                consumed_mins = max(0, int((now - started).total_seconds() / 60))

            total_sla_mins = sla_hours * 60
            pct_consumed = (consumed_mins / total_sla_mins * 100) if total_sla_mins > 0 else 0.0
        except Exception as exc:
            logger.warning("Could not compute waiting-time score: %s", exc)

    breakdown.waiting_time = _waiting_score(pct_consumed)

    score = breakdown.total
    priority = _priority_from_score(score)

    logger.debug(
        "Priority calc: severity=%s sentiment=%s impact=%s wait=%.1f%% sla=%s → score=%d → %s",
        severity, sentiment, customer_impact, pct_consumed, sla_status, score, priority,
    )

    return priority, score, breakdown
