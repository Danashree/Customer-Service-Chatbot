"""
Task 6 Phase 2 — Conversation Context Manager
=============================================
Manages sliding-window conversation history (at least 10 messages),
multilingual message retention, entity continuity across turns,
explicit correction tracking, and per-conversation isolation.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any, Dict, List, Optional

from .analyzer import analyze_multilingual_message
from .clock import Clock, SystemClock
from .config import MultilingualConfig
from .intent_handler import IntentHandler
from .models import (
    ConversationContext,
    ConversationMessage,
    CorrectionRecord,
    EntityType,
    IntentCandidate,
    IntentResult,
    MultilingualAnalysisResult,
    PreservedEntity,
)

# Correction detection indicators across languages
CORRECTION_PATTERNS = [
    r"\bsorry\b",
    r"\bactually\b",
    r"\bcorrect(?:ion|ed)?\b",
    r"\bmistake\b",
    r"\bnot\b.*\bbut\b",
    r"\binstead of\b",
    r"\bupdate(?:d)?\b",
    r"\bwrong\b",
    # Tamil / Tanglish
    r"மன்னிக்கவும்",
    r"தவறாக",
    r"சரியான",
    r"இல்லை",
    r"\bmannikkavum\b",
    r"\bthavaraaga\b",
    r"\bsariyana\b",
    r"\billai\b",
    # Hindi / Hinglish
    r"माफ़ करना",
    r"गलती से",
    r"सही",
    r"नहीं",
    r"\bmaaf karna\b",
    r"\bgalti se\b",
    r"\bsahi\b",
    r"\bnahi\b",
    # Malayalam / Manglish
    r"ക്ഷമിക്കണം",
    r"തെറ്റായി",
    r"ശരിയായ",
    r"അല്ല",
    r"\bkshamikkanam\b",
    r"\bthettayi\b",
    r"\bshariyaya\b",
    r"\balla\b",
]


class ConversationContextManager:
    """
    Manages active context per conversation_id / session_id with:
      - Sliding window retention (default 10 messages)
      - Non-destructive full message history
      - Cross-language context continuity
      - Entity continuity & correction handling
    """

    def __init__(
        self,
        config: Optional[MultilingualConfig] = None,
        clock: Optional[Clock] = None,
    ):
        self.config = config or MultilingualConfig.from_env()
        self.clock = clock or SystemClock()
        self.intent_handler = IntentHandler(config=self.config)
        # Storage mapped by conversation_id -> conversation state
        self._conversations: Dict[str, Dict[str, Any]] = {}

    def get_or_create_context(self, conversation_id: str) -> ConversationContext:
        """Retrieves or initializes isolated conversation context."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = {
                "all_messages": [],
                "active_entities": {},
                "entity_history": {},
                "corrections": [],
            }
        return self.get_context(conversation_id)

    def add_message(
        self,
        conversation_id: str,
        role: str,
        text: str,
        analysis: Optional[MultilingualAnalysisResult] = None,
        intents: Optional[IntentResult] = None,
        message_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> ConversationMessage:
        """
        Adds a new message to the conversation context:
          1. Performs multilingual analysis (entity preservation, spelling normalization, language detection)
          2. Performs intent detection and multi-intent decomposition
          3. Checks for and processes explicit entity corrections
          4. Updates active entities and entity history
          5. Stores message and enforces the active sliding window
        """
        self.get_or_create_context(conversation_id)
        conv_data = self._conversations[conversation_id]

        mid = message_id or f"msg_{uuid.uuid4().hex[:12]}"
        now = timestamp if timestamp is not None else self.clock.now()

        # Step 1: Multilingual analysis if not provided
        if analysis is None:
            analysis = analyze_multilingual_message(text, config=self.config)

        # Step 2: Intent detection if not provided
        current_ctx = self.get_context(conversation_id)
        if intents is None and role == "user":
            intents = self.intent_handler.detect_intents(text, context=current_ctx)

        detected_intents: List[IntentCandidate] = intents.intents if intents else []
        intent_confidence = intents.confidence if intents else 0.0

        # Step 3: Entity continuity and correction tracking (for user messages)
        if role == "user":
            self._process_entities_and_corrections(
                conversation_id=conversation_id,
                message_id=mid,
                text=text,
                entities=analysis.entities.entities,
                timestamp=now,
            )

        # Step 4: Construct conversation message
        msg = ConversationMessage(
            message_id=mid,
            role=role,
            original_text=text,
            detected_language=analysis.language.primary_language,
            detected_languages=analysis.language.detected_languages,
            normalized_text=analysis.normalized_text,
            preserved_entities=analysis.entities.entities,
            intents=detected_intents,
            intent_confidence=intent_confidence,
            timestamp=now,
        )

        conv_data["all_messages"].append(msg)
        return msg

    def get_context(self, conversation_id: str) -> ConversationContext:
        """
        Returns the active ConversationContext for the conversation_id,
        containing the most recent `context_window` messages.
        """
        if conversation_id not in self._conversations:
            return ConversationContext(
                conversation_id=conversation_id,
                messages=[],
                active_entities={},
                entity_history={},
                corrections=[],
            )

        conv_data = self._conversations[conversation_id]
        all_msgs = conv_data["all_messages"]

        # Sliding window of at most context_window messages
        window_size = self.config.context_window
        active_msgs = all_msgs[-window_size:] if len(all_msgs) > window_size else list(all_msgs)

        return ConversationContext(
            conversation_id=conversation_id,
            messages=active_msgs,
            active_entities=dict(conv_data["active_entities"]),
            entity_history={k: list(v) for k, v in conv_data["entity_history"].items()},
            corrections=list(conv_data["corrections"]),
        )

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[ConversationMessage]:
        """Returns the most recent messages up to the specified limit or context window."""
        ctx = self.get_context(conversation_id)
        if limit is None:
            return ctx.messages
        return ctx.messages[-limit:] if len(ctx.messages) > limit else ctx.messages

    def get_all_messages(self, conversation_id: str) -> List[ConversationMessage]:
        """Returns full historical messages stored for the conversation."""
        if conversation_id not in self._conversations:
            return []
        return list(self._conversations[conversation_id]["all_messages"])

    def clear_context(self, conversation_id: str) -> None:
        """Resets the context for a given conversation ID."""
        if conversation_id in self._conversations:
            self._conversations[conversation_id] = {
                "all_messages": [],
                "active_entities": {},
                "entity_history": {},
                "corrections": [],
            }

    def update_context(
        self,
        conversation_id: str,
        active_entities: Optional[Dict[str, str]] = None,
    ) -> ConversationContext:
        """Directly updates context active entities if needed."""
        self.get_or_create_context(conversation_id)
        if active_entities:
            self._conversations[conversation_id]["active_entities"].update(active_entities)
        return self.get_context(conversation_id)

    def get_active_entity(
        self,
        conversation_id: str,
        entity_type: EntityType,
    ) -> Optional[str]:
        """Retrieves the currently active entity value for a specific entity type."""
        ctx = self.get_context(conversation_id)
        return ctx.active_entities.get(entity_type.value)

    def get_entity_history(
        self,
        conversation_id: str,
        entity_type: EntityType,
    ) -> List[str]:
        """Retrieves all historical values for an entity type in this conversation."""
        ctx = self.get_context(conversation_id)
        return ctx.entity_history.get(entity_type.value, [])

    # -------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------

    def _is_correction(self, text: str) -> bool:
        """Determines if the message contains explicit correction indicators."""
        for pattern in CORRECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _process_entities_and_corrections(
        self,
        conversation_id: str,
        message_id: str,
        text: str,
        entities: List[PreservedEntity],
        timestamp: Optional[float] = None,
    ) -> None:
        """Updates active entities, entity history, and records explicit corrections."""
        conv_data = self._conversations[conversation_id]
        active = conv_data["active_entities"]
        history = conv_data["entity_history"]
        corrections = conv_data["corrections"]
        ts = timestamp if timestamp is not None else self.clock.now()

        is_correction_msg = self._is_correction(text)

        for entity in entities:
            etype_str = entity.entity_type.value
            val = entity.value

            # Initialize history list if needed
            if etype_str not in history:
                history[etype_str] = []

            # Check if this is an explicit correction
            if is_correction_msg and etype_str in active and active[etype_str] != val:
                old_val = active[etype_str]
                record = CorrectionRecord(
                    entity_type=entity.entity_type,
                    original_value=old_val,
                    corrected_value=val,
                    message_id=message_id,
                    timestamp=ts,
                )
                corrections.append(record)
                active[etype_str] = val
                history[etype_str].append(val)
            else:
                # Standard entity registration / update
                if val not in history[etype_str]:
                    history[etype_str].append(val)
                active[etype_str] = val
