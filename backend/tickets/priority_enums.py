"""
Task 3 Phase 2: Priority, SLA, and Escalation Enums.
All enums used in priority calculation, SLA tracking, and escalation management.
"""

from enum import Enum


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Sentiment(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class CustomerImpact(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Priority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SLAStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    WARNING = "WARNING"       # >= 75% SLA consumed
    BREACHED = "BREACHED"     # >= 100% SLA consumed
    RESOLVED = "RESOLVED"


class EscalationStatus(str, Enum):
    NOT_ESCALATED = "NOT_ESCALATED"
    ESCALATED = "ESCALATED"
