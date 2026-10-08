"""
Task 4: Pydantic models for multilingual sentiment analysis, risk detection,
escalation management, and response tone control.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Sentiment Labels
# ──────────────────────────────────────────────────────────────────────────────

class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    FRUSTRATED = "FRUSTRATED"
    URGENT = "URGENT"
    SARCASTIC = "SARCASTIC"


class AnalysisMethod(str, Enum):
    ML_MODEL = "ML_MODEL"
    RULE_BASED = "RULE_BASED"   # deterministic fallback


# ──────────────────────────────────────────────────────────────────────────────
# Risk Levels
# ──────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskCondition(str, Enum):
    NONE = "none"
    ACCOUNT_COMPROMISE = "account_compromise_detected"
    DUPLICATE_PAYMENT = "duplicate_payment_detected"
    LEGAL_THREAT = "legal_threat_detected"
    CUSTOM_HIGH_RISK = "custom_high_risk_detected"


# ──────────────────────────────────────────────────────────────────────────────
# Queue Types
# ──────────────────────────────────────────────────────────────────────────────

class QueueType(str, Enum):
    NORMAL = "NORMAL"
    ON_CALL_QUEUE = "ON_CALL_QUEUE"
    NEXT_WORKING_DAY = "NEXT_WORKING_DAY"


# ──────────────────────────────────────────────────────────────────────────────
# Escalation Status
# ──────────────────────────────────────────────────────────────────────────────

class EscalationRecordStatus(str, Enum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    IN_PROGRESS = "IN_PROGRESS"


# ──────────────────────────────────────────────────────────────────────────────
# Response Tone
# ──────────────────────────────────────────────────────────────────────────────

class ResponseTone(str, Enum):
    FRIENDLY_POSITIVE = "friendly/positive"
    PROFESSIONAL_NEUTRAL = "professional/neutral"
    EMPATHETIC_CALM = "empathetic/calm"
    EMPATHETIC_DEESCALATING = "empathetic/de-escalating"
    CONCISE_ACTION_FOCUSED = "concise/action-focused"
    CALM_PROFESSIONAL = "calm/professional/non-confrontational"
    CALM_ESCALATION_FOCUSED = "calm/professional/escalation-focused"


# ──────────────────────────────────────────────────────────────────────────────
# Sentiment Analysis Result
# ──────────────────────────────────────────────────────────────────────────────

class MessageSentimentResult(BaseModel):
    """Result of analysing a single message."""

    sentiment: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    frustration: bool = False
    urgency: bool = False
    sarcasm: bool = False
    language: Optional[str] = None          # ISO 639-1 code if detected
    analysis_method: AnalysisMethod = AnalysisMethod.RULE_BASED


class ConversationSentimentResult(BaseModel):
    """Conversation-level sentiment summary (considers history)."""

    overall_sentiment: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    frustration: bool = False
    urgency: bool = False
    sarcasm: bool = False
    repeated_negative: bool = False         # multiple consecutive negatives
    negative_message_count: int = 0
    total_messages_analysed: int = 0
    language: Optional[str] = None
    analysis_method: AnalysisMethod = AnalysisMethod.RULE_BASED


class SentimentAnalysisResponse(BaseModel):
    """Full response returned by POST /sentiment/analyze."""

    message_analysis: MessageSentimentResult
    conversation_analysis: Optional[ConversationSentimentResult] = None
    tone_recommendation: ResponseTone = ResponseTone.PROFESSIONAL_NEUTRAL


# ──────────────────────────────────────────────────────────────────────────────
# Risk Detection
# ──────────────────────────────────────────────────────────────────────────────

class RiskAssessment(BaseModel):
    """Result of risk detection on a message / conversation."""

    risk_level: RiskLevel = RiskLevel.NORMAL
    risk_condition: RiskCondition = RiskCondition.NONE
    risk_reason: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Escalation Evaluation
# ──────────────────────────────────────────────────────────────────────────────

class EscalationRequest(BaseModel):
    """Input for POST /escalations/evaluate."""

    conversation_id: Optional[str] = None
    ticket_id: Optional[str] = None
    message: str
    conversation_history: Optional[List[str]] = None

    # Timing fields (for the 15-minute unresolved timer)
    conversation_started_at: Optional[datetime] = None
    last_customer_message_at: Optional[datetime] = None
    unresolved_since: Optional[datetime] = None
    current_time: Optional[datetime] = None  # injectable for tests

    # Customer-level metadata (optional)
    customer_name: Optional[str] = None


class EscalationEvaluation(BaseModel):
    """Output of POST /escalations/evaluate."""

    should_escalate: bool
    reason: Optional[str] = None
    activated_condition: str = "none"
    risk_level: RiskLevel = RiskLevel.NORMAL
    queue_type: QueueType = QueueType.NORMAL
    sentiment: SentimentLabel = SentimentLabel.NEUTRAL
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    frustration: bool = False
    urgency: bool = False
    sarcasm: bool = False
    repeated_negative: bool = False
    business_hours_status: str = "unknown"   # "open" | "closed"
    escalation_id: Optional[str] = None      # set if a record was created


# ──────────────────────────────────────────────────────────────────────────────
# Escalation Record (persisted)
# ──────────────────────────────────────────────────────────────────────────────

class EscalationRecord(BaseModel):
    """Persistent escalation record stored to JSON."""

    escalation_id: str
    ticket_id: Optional[str] = None
    conversation_id: Optional[str] = None
    created_at: str                     # ISO-8601
    escalation_reason: str
    activated_condition: str
    sentiment: str
    sentiment_confidence: float = 0.0
    risk_level: str = RiskLevel.NORMAL.value
    urgency: bool = False
    frustration: bool = False
    sarcasm: bool = False
    business_hours_status: str = "unknown"
    queue_type: str = QueueType.NORMAL.value
    conversation_summary: str = ""      # PII-masked
    status: str = EscalationRecordStatus.OPEN.value
    escalation_timestamp: str = ""      # same as created_at by default


# ──────────────────────────────────────────────────────────────────────────────
# Tone Controller Output
# ──────────────────────────────────────────────────────────────────────────────

class ToneControlResult(BaseModel):
    """Computed tone recommendation with rationale."""

    tone: ResponseTone
    rationale: str
    sentiment_input: SentimentLabel
    risk_input: RiskLevel


# ──────────────────────────────────────────────────────────────────────────────
# Sentiment Analyze API request
# ──────────────────────────────────────────────────────────────────────────────

class SentimentAnalyzeRequest(BaseModel):
    message: str
    conversation_history: Optional[List[str]] = None
