"""
Task 6 Phase 3 — Production-Grade Session Manager
=================================================
Manages customer session lifecycles with:
  - Strict multi-customer simultaneous session isolation
  - Configurable 30-minute inactivity expiry (boundary-tested)
  - Non-destructive conversation summary generation
  - 24-hour summary restoration into new active sessions
  - Completely clean new session initialization after > 24 hours
  - Deterministic simulated-time clock support
  - Thread-safe concurrency control
"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .clock import Clock, SystemClock
from .config import MultilingualConfig
from .context_manager import ConversationContextManager
from .models import (
    ConversationContext,
    ConversationMessage,
    CustomerSession,
    EntityType,
    SessionStatus,
    SessionSummary,
)


class SessionManager:
    """
    Manages session lifecycle, inactivity timeouts, summary generation,
    and cross-session restoration with strict customer isolation.
    """

    def __init__(
        self,
        config: Optional[MultilingualConfig] = None,
        clock: Optional[Clock] = None,
        context_manager: Optional[ConversationContextManager] = None,
    ):
        self.config = config or MultilingualConfig.from_env()
        self.clock = clock or SystemClock()
        self.context_manager = context_manager or ConversationContextManager(
            config=self.config,
            clock=self.clock,
        )
        # Lock for thread safety during simultaneous customer sessions
        self._lock = threading.RLock()
        # Storage: session_id -> CustomerSession
        self._sessions: Dict[str, CustomerSession] = {}
        # Customer index: customer_id -> List[session_id] in chronological order
        self._customer_sessions: Dict[str, List[str]] = {}
        # Summary storage: session_id -> SessionSummary
        self._summaries: Dict[str, SessionSummary] = {}

    # ===================================================================
    # Session Creation & Retrieval
    # ===================================================================

    def create_session(
        self,
        customer_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CustomerSession:
        """
        Creates a new active customer session with unique identifier
        and isolated conversation context.
        """
        with self._lock:
            sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
            cid = f"conv_{sid}"
            now = self.clock.now()

            session = CustomerSession(
                session_id=sid,
                customer_id=customer_id,
                conversation_id=cid,
                status=SessionStatus.ACTIVE,
                created_at=now,
                last_activity_at=now,
                metadata=metadata or {},
            )

            self._sessions[sid] = session
            if customer_id not in self._customer_sessions:
                self._customer_sessions[customer_id] = []
            self._customer_sessions[customer_id].append(sid)

            # Ensure conversation context is initialized
            self.context_manager.get_or_create_context(cid)
            return session

    def get_session(self, session_id: str) -> Optional[CustomerSession]:
        """
        Retrieves a session by ID and automatically evaluates inactivity timeout.
        Returns the session with updated status if expired.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            # Check if active session has timed out due to inactivity
            if session.status == SessionStatus.ACTIVE:
                timeout_seconds = self.config.session_inactivity_minutes * 60.0
                elapsed = self.clock.now() - session.last_activity_at
                if elapsed >= timeout_seconds:
                    self._expire_session(session)

            return session

    def is_session_active(self, session_id: str) -> bool:
        """Returns True if the session exists and is currently active."""
        session = self.get_session(session_id)
        if not session:
            return False
        return session.status in (SessionStatus.ACTIVE, SessionStatus.RESTORED, SessionStatus.NEW)

    def touch_session(self, session_id: str) -> bool:
        """
        Updates last_activity_at timestamp for an active session.
        Returns False if session does not exist or is expired.
        """
        with self._lock:
            session = self.get_session(session_id)
            if not session or session.status == SessionStatus.EXPIRED:
                return False

            session.last_activity_at = self.clock.now()
            return True

    def add_message(
        self,
        session_id: str,
        role: str,
        text: str,
        message_id: Optional[str] = None,
    ) -> ConversationMessage:
        """
        Appends a message to the active session's conversation context
        and touches the session activity timestamp.
        Raises ValueError if session is expired or not found.
        """
        with self._lock:
            session = self.get_session(session_id)
            if not session:
                raise ValueError(f"Session '{session_id}' not found.")
            if session.status == SessionStatus.EXPIRED:
                raise ValueError(
                    f"Session '{session_id}' has expired due to inactivity. "
                    f"Please restore or start a new session."
                )

            # Update activity
            session.last_activity_at = self.clock.now()

            # Delegate to conversation context manager with synchronized time
            msg = self.context_manager.add_message(
                conversation_id=session.conversation_id,
                role=role,
                text=text,
                message_id=message_id,
                timestamp=self.clock.now(),
            )
            return msg

    # ===================================================================
    # Inactivity Expiry & Summary Generation
    # ===================================================================

    def expire_session(self, session_id: str) -> Optional[SessionSummary]:
        """Manually or explicitly expires an active session and generates its summary."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            return self._expire_session(session)

    def _expire_session(self, session: CustomerSession) -> Optional[SessionSummary]:
        """Internal helper to transition session to EXPIRED and persist summary."""
        session.status = SessionStatus.EXPIRED
        timeout_seconds = self.config.session_inactivity_minutes * 60.0
        # Calculate precise expiration timestamp based on last activity + timeout
        natural_expiry = session.last_activity_at + timeout_seconds
        session.expired_at = min(self.clock.now(), natural_expiry)

        summary = None
        if self.config.summary_enabled:
            summary = self._generate_summary(session)
            session.summary = summary
            self._summaries[session.session_id] = summary

        return summary

    def expire_inactive_sessions(self) -> List[str]:
        """
        Scans all registered sessions and expires any that exceed inactivity timeout.
        Returns list of newly expired session IDs.
        """
        with self._lock:
            expired_ids = []
            timeout_seconds = self.config.session_inactivity_minutes * 60.0
            now = self.clock.now()

            for sid, session in list(self._sessions.items()):
                if session.status == SessionStatus.ACTIVE:
                    if (now - session.last_activity_at) >= timeout_seconds:
                        self._expire_session(session)
                        expired_ids.append(sid)

            return expired_ids

    def _generate_summary(self, session: CustomerSession) -> SessionSummary:
        """
        Generates a concise, entity-accurate summary from conversation context.
        Preserves active corrected entities, customer intents, and observed languages.
        """
        ctx = self.context_manager.get_context(session.conversation_id)
        all_msgs = self.context_manager.get_all_messages(session.conversation_id)

        # 1. Determine primary intent and observed languages
        primary_intent = None
        languages: List[str] = []
        for msg in all_msgs:
            if msg.detected_language and msg.detected_language not in languages:
                languages.append(msg.detected_language)
            for intent_cand in msg.intents:
                if not primary_intent and intent_cand.intent not in ("greeting", "gratitude"):
                    primary_intent = intent_cand.intent

        # 2. Extract active entities (which already reflect the latest corrections!)
        active_entities = dict(ctx.active_entities)

        # 3. Build human-readable concise summary
        parts: List[str] = []
        if primary_intent:
            parts.append(f"Intent: {primary_intent}")
        if active_entities:
            ent_list = [f"{k}={v}" for k, v in active_entities.items()]
            parts.append(f"Entities: [{', '.join(ent_list)}]")
        if ctx.corrections:
            corr_list = [
                f"{c.entity_type.value} corrected from {c.original_value} to {c.corrected_value}"
                for c in ctx.corrections
            ]
            parts.append(f"Corrections: [{'; '.join(corr_list)}]")
        if languages:
            parts.append(f"Languages: {', '.join(languages)}")

        summary_text = " | ".join(parts) if parts else "Customer support discussion."

        unresolved = [primary_intent] if primary_intent else []

        return SessionSummary(
            summary_id=f"sum_{uuid.uuid4().hex[:12]}",
            session_id=session.session_id,
            customer_id=session.customer_id,
            primary_intent=primary_intent,
            detected_languages=languages,
            active_entities=active_entities,
            summary_text=summary_text,
            unresolved_issues=unresolved,
            generated_at=self.clock.now(),
            expired_at=session.expired_at or self.clock.now(),
        )

    # ===================================================================
    # 24-Hour Summary Restoration & Clean New Session
    # ===================================================================

    def get_restorable_session(
        self,
        customer_id: str,
        session_id: Optional[str] = None,
    ) -> Optional[CustomerSession]:
        """
        Determines whether an expired session is eligible for summary restoration
        within the configured restoration window (default 24 hours).

        Boundary Rules:
          - (current_time - expired_at) < restore_hours * 3600  -> Restorable
          - (current_time - expired_at) >= restore_hours * 3600 -> Restoration window closed
          - Strictly enforces customer isolation: customer A cannot restore customer B's session.
        """
        with self._lock:
            candidate_session = None

            if session_id:
                sess = self.get_session(session_id)
                # Verify customer identity
                if sess and sess.customer_id == customer_id:
                    candidate_session = sess
            else:
                # Find the most recent expired session for this customer
                cust_sids = self._customer_sessions.get(customer_id, [])
                for sid in reversed(cust_sids):
                    sess = self.get_session(sid)
                    if sess and sess.status == SessionStatus.EXPIRED:
                        candidate_session = sess
                        break

            if not candidate_session or candidate_session.status != SessionStatus.EXPIRED:
                return None

            if not candidate_session.expired_at:
                return None

            restore_window_seconds = self.config.session_restore_hours * 3600.0
            elapsed_since_expiry = self.clock.now() - candidate_session.expired_at

            if 0 <= elapsed_since_expiry < restore_window_seconds:
                return candidate_session

            return None

    def restore_session(
        self,
        customer_id: str,
        previous_session_id: Optional[str] = None,
    ) -> CustomerSession:
        """
        Restores previous conversation summary into a new active session if within
        the 24-hour window. Otherwise, initiates a completely clean new session.
        """
        with self._lock:
            eligible = self.get_restorable_session(
                customer_id=customer_id,
                session_id=previous_session_id,
            )

            if eligible and eligible.summary:
                # 1. Create a NEW active session linked to the expired session
                new_session = self.create_session(customer_id=customer_id)
                new_session.status = SessionStatus.RESTORED
                new_session.restored_from_session_id = eligible.session_id

                # 2. Inject preserved active entities into new conversation context
                self.context_manager.update_context(
                    conversation_id=new_session.conversation_id,
                    active_entities=eligible.summary.active_entities,
                )

                # 3. Add context restoration record to conversation history
                summary_note = (
                    f"[Restored from previous session {eligible.session_id}]: "
                    f"{eligible.summary.summary_text}"
                )
                self.context_manager.add_message(
                    conversation_id=new_session.conversation_id,
                    role="assistant",
                    text=summary_note,
                    timestamp=self.clock.now(),
                )

                return new_session

            # Fallback: start completely clean new session
            new_session = self.create_session(customer_id=customer_id)
            new_session.status = SessionStatus.NEW
            return new_session

    def create_new_session(self, customer_id: str) -> CustomerSession:
        """Explicitly starts a brand new session without restoring previous summaries."""
        with self._lock:
            sess = self.create_session(customer_id=customer_id)
            sess.status = SessionStatus.NEW
            return sess

    # ===================================================================
    # Convenience Inspection Helpers
    # ===================================================================

    def get_session_summary(self, session_id: str) -> Optional[SessionSummary]:
        """Retrieves stored summary for a session."""
        with self._lock:
            return self._summaries.get(session_id)

    def get_customer_sessions(self, customer_id: str) -> List[CustomerSession]:
        """Retrieves all sessions belonging to a specific customer."""
        with self._lock:
            sids = self._customer_sessions.get(customer_id, [])
            return [self._sessions[sid] for sid in sids if sid in self._sessions]
