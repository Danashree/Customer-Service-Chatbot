"""
Task 6 Phase 4 — Multilingual Conversational Orchestrator
=========================================================
Coordinates end-to-end request handling:
  1. Session resolution (active, 30-min expiry, 24-hr restoration, clean new session)
  2. Simultaneous session isolation per customer
  3. Multilingual analysis (Phase 1: language detection, spelling/transliteration normalization, entity shielding)
  4. Context management (Phase 2: 10-message sliding window, entity continuity, correction tracking)
  5. Intent handling & decomposition (Phase 2: multi-intent, confidence check, follow-up ambiguity resolution)
  6. Clarification enforcement for low confidence / ambiguous requests
  7. Downstream routing to Task 1 FAISS / Task 4 RAG / Task 5 Sentiment & Escalation
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .analyzer import analyze_multilingual_message
from .clock import Clock, SystemClock, SimulatedClock
from .config import MultilingualConfig
from .context_manager import ConversationContextManager
from .intent_handler import IntentHandler
from .models import (
    ConversationContext,
    ConversationMessage,
    CustomerSession,
    IntentCandidate,
    IntentResult,
    MultilingualAnalysisResult,
    SessionStatus,
)
from .session_manager import SessionManager


class MultilingualOrchestrator:
    """
    Unified orchestrator connecting SessionManager, ConversationContextManager,
    MultilingualAnalyzer, and IntentHandler for production chatbot flow.
    """

    def __init__(
        self,
        config: Optional[MultilingualConfig] = None,
        clock: Optional[Clock] = None,
        session_manager: Optional[SessionManager] = None,
    ):
        self.config = config or MultilingualConfig.from_env()
        self.clock = clock or SystemClock()
        self.session_manager = session_manager or SessionManager(
            config=self.config,
            clock=self.clock,
        )

    def process_incoming_request(
        self,
        question: str,
        customer_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        current_time: Optional[datetime] = None,
    ) -> Tuple[CustomerSession, MultilingualAnalysisResult, IntentResult, Optional[str]]:
        """
        Coordinates full pre-processing:
          1. Synchronizes simulated clock if current_time passed
          2. Identifies customer session, evaluates 30-min inactivity expiry and 24-hr restoration
          3. Executes multilingual analysis (detection, transliteration, spelling, entities)
          4. Checks language confidence (clarification if low or unsupported)
          5. Checks customer intent & ambiguity (clarification if low confidence or ambiguous entity)
          6. Records user turn in conversation context

        Returns:
            Tuple of:
              - session: CustomerSession
              - analysis: MultilingualAnalysisResult
              - intent_res: IntentResult
              - clarification_prompt: Optional[str] (non-None if chatbot must ask clarification)
        """
        # Step 1: Clock synchronization
        if current_time and isinstance(self.clock, SimulatedClock):
            self.clock.set_time(current_time.timestamp())

        # Step 2: Determine customer and session identity
        cid = customer_id or (f"cust_{session_id}" if session_id else "default_customer")
        session = self._resolve_session(customer_id=cid, requested_session_id=session_id)

        # Step 3: Multilingual Analysis (Phase 1)
        analysis = analyze_multilingual_message(question, config=self.config)

        # Step 4: Check Language Confidence Threshold
        lang_res = analysis.language
        if lang_res.requires_clarification or lang_res.confidence < self.config.language_confidence_threshold:
            clarification_text = self._build_language_clarification(lang_res)
            # Record turn in context
            self.session_manager.context_manager.add_message(
                conversation_id=session.conversation_id,
                role="user",
                text=question,
                analysis=analysis,
                timestamp=self.clock.now(),
            )
            self.session_manager.context_manager.add_message(
                conversation_id=session.conversation_id,
                role="assistant",
                text=clarification_text,
                timestamp=self.clock.now(),
            )
            self.session_manager.touch_session(session.session_id)
            dummy_intent = IntentResult(
                requires_clarification=True,
                clarification_reason=lang_res.reason or "LOW_LANGUAGE_CONFIDENCE",
                clarification_question=clarification_text,
            )
            return session, analysis, dummy_intent, clarification_text

        # Step 5: Intent Detection & Ambiguity Resolution (Phase 2)
        ctx = self.session_manager.context_manager.get_context(session.conversation_id)
        intent_res = self.session_manager.context_manager.intent_handler.detect_intents(
            text=analysis.normalized_text,
            context=ctx,
        )

        if intent_res.requires_clarification:
            clarification_text = (
                intent_res.clarification_question
                or "Could you please clarify your request so I can assist you better?"
            )
            # Record turn in context
            self.session_manager.context_manager.add_message(
                conversation_id=session.conversation_id,
                role="user",
                text=question,
                analysis=analysis,
                intents=intent_res,
                timestamp=self.clock.now(),
            )
            self.session_manager.context_manager.add_message(
                conversation_id=session.conversation_id,
                role="assistant",
                text=clarification_text,
                timestamp=self.clock.now(),
            )
            self.session_manager.touch_session(session.session_id)
            return session, analysis, intent_res, clarification_text

        # Step 6: Valid request without clarification needed
        # Record user message in context
        self.session_manager.context_manager.add_message(
            conversation_id=session.conversation_id,
            role="user",
            text=question,
            analysis=analysis,
            intents=intent_res,
            timestamp=self.clock.now(),
        )
        self.session_manager.touch_session(session.session_id)
        return session, analysis, intent_res, None

    def record_assistant_turn(self, session: CustomerSession, answer_text: str) -> None:
        """Records the chatbot's generated response in the conversation context."""
        self.session_manager.context_manager.add_message(
            conversation_id=session.conversation_id,
            role="assistant",
            text=answer_text,
            timestamp=self.clock.now(),
        )
        self.session_manager.touch_session(session.session_id)

    def build_response_metadata(
        self,
        session: CustomerSession,
        analysis: MultilingualAnalysisResult,
        intent_res: IntentResult,
        clarification_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Constructs safe, non-sensitive Task 6 metadata for the API payload."""
        ctx = self.session_manager.context_manager.get_context(session.conversation_id)
        return {
            "language": analysis.language.primary_language or self.config.default_language,
            "language_confidence": round(analysis.language.confidence, 2),
            "is_mixed": analysis.language.is_mixed,
            "is_transliterated": analysis.language.is_transliterated,
            "session_id": session.session_id,
            "customer_id": session.customer_id,
            "session_status": session.status.value,
            "context_messages": len(ctx.messages),
            "restored": session.status == SessionStatus.RESTORED,
            "clarification_required": clarification_text is not None,
            "clarification_reason": (
                analysis.clarification_reason
                if analysis.requires_clarification
                else (intent_res.clarification_reason if intent_res and intent_res.requires_clarification else None)
            ),
            "active_entities": dict(ctx.active_entities),
            "primary_intent": intent_res.primary_intent if intent_res else None,
            "is_multi_intent": intent_res.is_multi_intent if intent_res else False,
        }

    # -------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------

    def _resolve_session(
        self,
        customer_id: str,
        requested_session_id: Optional[str] = None,
    ) -> CustomerSession:
        """
        Retrieves or instantiates an active session adhering strictly to:
          - Inactivity expiry (>= 30 min)
          - 24-hr summary restoration into new RESTORED session
          - Clean NEW session if > 24h
        """
        if requested_session_id:
            sess = self.session_manager.get_session(requested_session_id)
            if sess:
                if sess.customer_id == customer_id:
                    if sess.status == SessionStatus.EXPIRED:
                        # Attempt restore within 24h window
                        return self.session_manager.restore_session(
                            customer_id=customer_id,
                            previous_session_id=requested_session_id,
                        )
                    return sess
                else:
                    # Security isolation: session belongs to another customer!
                    # Do NOT allow access or overwriting. Initialize new isolated session.
                    return self.session_manager.create_session(customer_id=customer_id)

            # If session doesn't exist under this ID, create it
            return self.session_manager.create_session(
                customer_id=customer_id,
                session_id=requested_session_id,
            )

        # No session_id specified: check if customer has active session
        cust_sessions = self.session_manager.get_customer_sessions(customer_id)
        for s in reversed(cust_sessions):
            active_check = self.session_manager.get_session(s.session_id)
            if active_check and active_check.status in (
                SessionStatus.ACTIVE,
                SessionStatus.RESTORED,
                SessionStatus.NEW,
            ):
                return active_check

        # Customer has no active session: check if eligible for restoration
        return self.session_manager.restore_session(customer_id=customer_id)

    def _build_language_clarification(self, lang_res: Any) -> str:
        """Constructs safe clarification message for low confidence or unsupported language."""
        if lang_res.reason == "UNSUPPORTED_LANGUAGE":
            return (
                "I currently only support English, Tamil, Hindi, and Malayalam. "
                "Could you please rephrase your query in one of these languages?"
            )
        return (
            "I'm having trouble understanding your message clearly. "
            "Could you please rephrase your question in English, Tamil, Hindi, or Malayalam?"
        )
