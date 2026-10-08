"""
Task 6 — Multilingual Support (Phases 1, 2 & 3)
==============================================
Package exposing multilingual foundation components:
  - config: MultilingualConfig
  - models: EntityType, PreservedEntity, EntityPreservationResult, LanguageDetectionResult,
            MultilingualAnalysisResult, IntentCandidate, IntentResult, ClarificationResult,
            CorrectionRecord, ConversationMessage, ConversationContext,
            SessionStatus, SessionSummary, CustomerSession
  - clock: Clock, SystemClock, SimulatedClock
  - entity_preserver: EntityPreserver
  - spelling_normalizer: SpellingNormalizer
  - language_detector: LanguageDetector
  - analyzer: analyze_multilingual_message
  - intent_handler: IntentHandler
  - context_manager: ConversationContextManager
  - session_manager: SessionManager
"""

from .config import MultilingualConfig, DEFAULT_SUPPORTED_LANGUAGES
from .models import (
    EntityType,
    PreservedEntity,
    EntityPreservationResult,
    LanguageDetectionResult,
    MultilingualAnalysisResult,
    IntentCandidate,
    IntentResult,
    ClarificationResult,
    CorrectionRecord,
    ConversationMessage,
    ConversationContext,
    SessionStatus,
    SessionSummary,
    CustomerSession,
)
from .clock import Clock, SystemClock, SimulatedClock
from .entity_preserver import EntityPreserver
from .spelling_normalizer import SpellingNormalizer
from .language_detector import LanguageDetector
from .analyzer import analyze_multilingual_message
from .intent_handler import IntentHandler
from .context_manager import ConversationContextManager
from .session_manager import SessionManager
from .orchestrator import MultilingualOrchestrator

__all__ = [
    "MultilingualConfig",
    "DEFAULT_SUPPORTED_LANGUAGES",
    "EntityType",
    "PreservedEntity",
    "EntityPreservationResult",
    "LanguageDetectionResult",
    "MultilingualAnalysisResult",
    "IntentCandidate",
    "IntentResult",
    "ClarificationResult",
    "CorrectionRecord",
    "ConversationMessage",
    "ConversationContext",
    "SessionStatus",
    "SessionSummary",
    "CustomerSession",
    "Clock",
    "SystemClock",
    "SimulatedClock",
    "EntityPreserver",
    "SpellingNormalizer",
    "LanguageDetector",
    "analyze_multilingual_message",
    "IntentHandler",
    "ConversationContextManager",
    "SessionManager",
    "MultilingualOrchestrator",
]

