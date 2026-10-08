"""
Task 6 Phase 1 — Multilingual Foundation Data Models
=====================================================
Typed data models for:
  - Entity preservation (Order IDs, Product Codes, Dates, Names, Emails, Phones, URLs)
  - Language detection (Supported languages, mixed-language, transliteration, confidence)
  - Overall multilingual analysis result
"""
from __future__ import annotations

from enum import Enum
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of sensitive or business-critical entities to preserve."""
    NAME = "name"
    ORDER_ID = "order_id"
    DATE = "date"
    PRODUCT_CODE = "product_code"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"


class PreservedEntity(BaseModel):
    """An identified entity that must not be modified or translated."""
    entity_type: EntityType
    value: str
    start: int
    end: int
    placeholder: str


class EntityPreservationResult(BaseModel):
    """Result of entity extraction and placeholder shielding."""
    original_text: str
    protected_text: str
    entities: List[PreservedEntity] = Field(default_factory=list)


class LanguageDetectionResult(BaseModel):
    """Structured result of language detection."""
    primary_language: Optional[str] = Field(
        default=None,
        description="ISO 639-1 code of primary detected language (e.g. en, ta, hi, ml)."
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score between 0.0 and 1.0."
    )
    detected_languages: List[str] = Field(
        default_factory=list,
        description="List of all detected language codes."
    )
    is_mixed: bool = Field(
        default=False,
        description="True if the message contains multiple languages."
    )
    is_transliterated: bool = Field(
        default=False,
        description="True if non-English language is written in Roman/Latin script."
    )
    transliterated_language: Optional[str] = Field(
        default=None,
        description="Language code of transliterated content if applicable."
    )
    requires_clarification: bool = Field(
        default=False,
        description="True if confidence is too low or language is unsupported/ambiguous."
    )
    reason: Optional[str] = Field(
        default=None,
        description="Explanation when clarification is required (e.g. LOW_LANGUAGE_CONFIDENCE, UNSUPPORTED_LANGUAGE)."
    )


class MultilingualAnalysisResult(BaseModel):
    """Consolidated multilingual analysis outcome for a single incoming message."""
    original_text: str
    normalized_text: str
    language: LanguageDetectionResult
    entities: EntityPreservationResult
    requires_clarification: bool = False
    clarification_reason: Optional[str] = None


# =======================================================================
# Task 6 Phase 2 — Context, Intent & Correction Data Models
# =======================================================================

class IntentCandidate(BaseModel):
    """A detected customer intent candidate with confidence and parameters."""
    intent: str = Field(description="Intent name (e.g. order_status, refund_request, course_access).")
    confidence: float = Field(default=0.0, description="Confidence score for this specific intent.")
    matched_text: Optional[str] = Field(default=None, description="Clause or segment of text that triggered this intent.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameters/entities for this intent.")


class ClarificationResult(BaseModel):
    """Structured requirement for customer clarification."""
    requires_clarification: bool = False
    reason: Optional[str] = Field(default=None, description="e.g. LOW_INTENT_CONFIDENCE, AMBIGUOUS_INTENT, AMBIGUOUS_ENTITY.")
    clarification_question: Optional[str] = Field(default=None, description="Concise question to present to user.")
    candidate_intents: List[str] = Field(default_factory=list, description="Candidate intents under consideration.")
    candidate_entities: List[str] = Field(default_factory=list, description="Candidate entities if resolving ambiguity.")


class IntentResult(BaseModel):
    """Outcome of intent analysis across single or multi-intent messages."""
    intents: List[IntentCandidate] = Field(default_factory=list, description="All detected intent candidates.")
    primary_intent: Optional[str] = Field(default=None, description="Primary detected intent name.")
    confidence: float = Field(default=0.0, description="Overall or primary intent confidence.")
    is_multi_intent: bool = Field(default=False, description="True if multiple distinct intents detected.")
    requires_clarification: bool = Field(default=False, description="True if confidence is too low or query is ambiguous.")
    clarification_reason: Optional[str] = Field(default=None, description="Reason for clarification request.")
    clarification_question: Optional[str] = Field(default=None, description="Clarification question to ask user.")


class CorrectionRecord(BaseModel):
    """Record of an explicit entity correction performed in conversation context."""
    entity_type: EntityType = Field(description="Type of entity that was corrected.")
    original_value: str = Field(description="Previous/superseded entity value.")
    corrected_value: str = Field(description="New/active entity value.")
    message_id: Optional[str] = Field(default=None, description="ID of the message where correction occurred.")
    timestamp: float = Field(default_factory=time.time, description="Timestamp of the correction.")


class ConversationMessage(BaseModel):
    """A single turn in the conversation retaining original and processed representations."""
    message_id: str = Field(description="Unique identifier for the message.")
    role: str = Field(description="Speaker role: 'user' or 'assistant'.")
    original_text: str = Field(description="Unmodified raw text of the message.")
    detected_language: Optional[str] = Field(default=None, description="Primary detected language code (e.g. en, ta, hi, ml).")
    detected_languages: List[str] = Field(default_factory=list, description="All detected language codes.")
    normalized_text: str = Field(default="", description="Spelling-normalized and entity-restored text.")
    preserved_entities: List[PreservedEntity] = Field(default_factory=list, description="Entities found in this message.")
    intents: List[IntentCandidate] = Field(default_factory=list, description="Intents identified in this message.")
    intent_confidence: float = Field(default=0.0, description="Confidence of primary intent.")
    timestamp: float = Field(default_factory=time.time, description="Creation timestamp.")


class ConversationContext(BaseModel):
    """Stateful active conversation context within configured window."""
    conversation_id: str = Field(description="Unique conversation / session identifier.")
    messages: List[ConversationMessage] = Field(default_factory=list, description="Active messages within window.")
    active_entities: Dict[str, str] = Field(
        default_factory=dict,
        description="Currently active entity values keyed by EntityType value (e.g. {'order_id': 'ORD54321'})."
    )
    entity_history: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Historical entity values seen in conversation keyed by EntityType value."
    )
    corrections: List[CorrectionRecord] = Field(
        default_factory=list,
        description="List of explicit corrections performed during this conversation."
    )


# =======================================================================
# Task 6 Phase 3 — Session Management Data Models
# =======================================================================

class SessionStatus(str, Enum):
    """Lifecycle status of a customer session."""
    ACTIVE = "active"
    EXPIRED = "expired"
    RESTORED = "restored"
    NEW = "new"


class SessionSummary(BaseModel):
    """Persisted summary of an expired session for 24-hour restoration."""
    summary_id: str = Field(description="Unique ID of this summary record.")
    session_id: str = Field(description="Originating session identifier.")
    customer_id: str = Field(description="Customer/user identifier.")
    primary_intent: Optional[str] = Field(default=None, description="Primary detected intent of session.")
    detected_languages: List[str] = Field(default_factory=list, description="Languages observed in session.")
    active_entities: Dict[str, str] = Field(default_factory=dict, description="Final active entities at session expiry.")
    summary_text: str = Field(description="Concise, non-PII summary of discussion and unresolved issues.")
    unresolved_issues: List[str] = Field(default_factory=list, description="Unresolved topics requiring continuation.")
    generated_at: float = Field(description="Timestamp when summary was generated.")
    expired_at: float = Field(description="Timestamp when originating session expired.")


class CustomerSession(BaseModel):
    """Session record tracking lifecycle, isolation, timestamps, and context link."""
    session_id: str = Field(description="Unique session identifier.")
    customer_id: str = Field(description="Customer / User identifier.")
    conversation_id: str = Field(description="Associated conversation identifier.")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Current lifecycle state.")
    created_at: float = Field(description="Epoch creation timestamp.")
    last_activity_at: float = Field(description="Epoch timestamp of last interaction.")
    expired_at: Optional[float] = Field(default=None, description="Timestamp when session was expired.")
    restored_from_session_id: Optional[str] = Field(default=None, description="Source session ID if restored.")
    summary: Optional[SessionSummary] = Field(default=None, description="Summary if session has expired.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional session metadata.")


