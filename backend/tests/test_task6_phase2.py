"""
Task 6 Phase 2 — Test Suite: Conversation Context & Intent Handling
===================================================================
36 Comprehensive Behavioral Tests covering:
  - Group A: Context (create, add, retrieve, 10-msg window, 11th evicts, ordering, empty, clear)
  - Group B: Multilingual Context (En->Ta, Ta->En, Hi->Ta, Ml->En, mixed, per-msg language)
  - Group C: Entity Continuity (order ID, name, date, product code, multiple entities)
  - Group D: Corrections (order ID, date, product code, preserves history, latest is active)
  - Group E: Intent Detection (simple, multiple, multilingual, transliterated, low conf, clarification)
  - Group F: Ambiguity Resolution (context-resolved follow-up, ambiguous follow-up, multiple competing entities)
  - Group G: Configuration & Isolation (custom window, custom threshold, per-session isolation)
"""
import pytest

from backend.multilingual import (
    ConversationContextManager,
    EntityType,
    IntentCandidate,
    IntentHandler,
    MultilingualConfig,
)


# =======================================================================
# Group A: Context Lifecycle & Sliding Window
# =======================================================================

class TestGroupAContextLifecycle:
    """Tests conversation context lifecycle and sliding window mechanics."""

    def test_01_create_context(self):
        manager = ConversationContextManager()
        ctx = manager.get_or_create_context("test_conv_01")
        assert ctx.conversation_id == "test_conv_01"
        assert len(ctx.messages) == 0
        assert ctx.active_entities == {}
        assert ctx.entity_history == {}
        assert ctx.corrections == []

    def test_02_add_one_message(self):
        manager = ConversationContextManager()
        msg = manager.add_message(
            conversation_id="test_conv_02",
            role="user",
            text="Where is my order?",
        )
        assert msg.role == "user"
        assert msg.original_text == "Where is my order?"
        assert msg.detected_language == "en"
        assert len(manager.get_context("test_conv_02").messages) == 1

    def test_03_retrieve_message(self):
        manager = ConversationContextManager()
        msg = manager.add_message(
            conversation_id="test_conv_03",
            role="user",
            text="Hello support team",
            message_id="msg_fixed_03",
        )
        recent = manager.get_recent_messages("test_conv_03")
        assert len(recent) == 1
        assert recent[0].message_id == "msg_fixed_03"
        assert recent[0].original_text == "Hello support team"
        assert recent[0].role == "user"

    def test_04_retain_10_messages(self):
        manager = ConversationContextManager()
        for i in range(10):
            manager.add_message("test_conv_04", "user", f"Message number {i}")

        ctx = manager.get_context("test_conv_04")
        assert len(ctx.messages) == 10
        assert ctx.messages[0].original_text == "Message number 0"
        assert ctx.messages[9].original_text == "Message number 9"

    def test_05_11th_message_evicts_oldest_active_message(self):
        manager = ConversationContextManager()
        # Add 12 messages
        for i in range(12):
            manager.add_message("test_conv_05", "user", f"Turn {i}")

        ctx = manager.get_context("test_conv_05")
        # Active context must retain exactly 10 latest messages
        assert len(ctx.messages) == 10
        assert ctx.messages[0].original_text == "Turn 2"
        assert ctx.messages[-1].original_text == "Turn 11"

        # Full non-destructive history retains all 12
        all_msgs = manager.get_all_messages("test_conv_05")
        assert len(all_msgs) == 12
        assert all_msgs[0].original_text == "Turn 0"

    def test_06_ordering_preserved(self):
        manager = ConversationContextManager()
        messages_sent = ["First", "Second", "Third", "Fourth", "Fifth"]
        for txt in messages_sent:
            manager.add_message("test_conv_06", "user", txt)

        ctx = manager.get_context("test_conv_06")
        retrieved_texts = [m.original_text for m in ctx.messages]
        assert retrieved_texts == messages_sent

    def test_07_empty_context(self):
        manager = ConversationContextManager()
        ctx = manager.get_context("non_existent_conv")
        assert ctx.conversation_id == "non_existent_conv"
        assert len(ctx.messages) == 0
        assert ctx.active_entities == {}

    def test_08_clear_context(self):
        manager = ConversationContextManager()
        manager.add_message("test_conv_08", "user", "My order is ORD-123")
        assert len(manager.get_context("test_conv_08").messages) == 1

        manager.clear_context("test_conv_08")
        ctx = manager.get_context("test_conv_08")
        assert len(ctx.messages) == 0
        assert ctx.active_entities == {}
        assert ctx.entity_history == {}


# =======================================================================
# Group B: Multilingual Context
# =======================================================================

class TestGroupBMultilingualContext:
    """Tests multilingual continuity and language tracking across turns."""

    def test_09_english_to_tamil(self):
        manager = ConversationContextManager()
        m1 = manager.add_message("test_conv_09", "user", "Where is my order?")
        m2 = manager.add_message("test_conv_09", "user", "என் order எங்கே இருக்கு?")
        assert m1.detected_language == "en"
        assert m2.detected_language == "ta"

    def test_10_tamil_to_english(self):
        manager = ConversationContextManager()
        m1 = manager.add_message("test_conv_10", "user", "வணக்கம், உதவி வேண்டும்")
        m2 = manager.add_message("test_conv_10", "user", "Can you help me with login?")
        assert m1.detected_language == "ta"
        assert m2.detected_language == "en"

    def test_11_hindi_to_tamil(self):
        manager = ConversationContextManager()
        m1 = manager.add_message("test_conv_11", "user", "नमस्ते, मुझे मदद चाहिए")
        m2 = manager.add_message("test_conv_11", "user", "என் பெயர் தனாஸ்ரீ")
        assert m1.detected_language == "hi"
        assert m2.detected_language == "ta"

    def test_12_malayalam_to_english(self):
        manager = ConversationContextManager()
        m1 = manager.add_message("test_conv_12", "user", "എന്റെ ഓർഡർ എവിടെയാണ്?")
        m2 = manager.add_message("test_conv_12", "user", "Check tracking status please")
        assert m1.detected_language == "ml"
        assert m2.detected_language == "en"

    def test_13_mixed_language_context(self):
        manager = ConversationContextManager()
        m1 = manager.add_message(
            "test_conv_13",
            "user",
            "Mera course access nahi ho raha, please help with ORD-1002",
        )
        assert "hi" in m1.detected_languages or m1.detected_language == "hi"

    def test_14_language_stored_per_message(self):
        manager = ConversationContextManager()
        manager.add_message("test_conv_14", "user", "Hello support")
        manager.add_message("test_conv_14", "user", "என் order எங்கே?")
        manager.add_message("test_conv_14", "user", "मेरा सवाल है")
        manager.add_message("test_conv_14", "user", "നന്ദി")

        ctx = manager.get_context("test_conv_14")
        langs = [m.detected_language for m in ctx.messages]
        assert langs == ["en", "ta", "hi", "ml"]


# =======================================================================
# Group C: Entity Continuity Across Context
# =======================================================================

class TestGroupCEntityContinuity:
    """Tests preservation and availability of entities across turns."""

    def test_15_order_id_retained_across_messages(self):
        manager = ConversationContextManager()
        manager.add_message("test_conv_15", "user", "My order is ORD-2026-999")
        manager.add_message("test_conv_15", "user", "When will it arrive?")

        active_oid = manager.get_active_entity("test_conv_15", EntityType.ORDER_ID)
        assert active_oid == "ORD-2026-999"

    def test_16_name_retained_across_messages(self):
        manager = ConversationContextManager()
        manager.add_message("test_conv_16", "user", "My name is Danashree.")
        manager.add_message("test_conv_16", "user", "Can you check my account?")

        active_name = manager.get_active_entity("test_conv_16", EntityType.NAME)
        assert active_name == "Danashree"

    def test_17_date_retained_across_messages(self):
        manager = ConversationContextManager()
        manager.add_message("test_conv_17", "user", "I registered on 2026-09-28.")
        manager.add_message("test_conv_17", "user", "Is my certificate ready?")

        active_date = manager.get_active_entity("test_conv_17", EntityType.DATE)
        assert active_date == "2026-09-28"

    def test_18_product_code_retained_across_messages(self):
        manager = ConversationContextManager()
        manager.add_message("test_conv_18", "user", "I have an issue with PROD-AX21.")
        manager.add_message("test_conv_18", "user", "The videos won't play.")

        active_code = manager.get_active_entity("test_conv_18", EntityType.PRODUCT_CODE)
        assert active_code == "PROD-AX21"

    def test_19_multiple_entities_retained(self):
        manager = ConversationContextManager()
        manager.add_message(
            "test_conv_19",
            "user",
            "My name is Danashree and my order is ORD-77889 for PROD-BC99 on 2026-09-25",
        )
        ctx = manager.get_context("test_conv_19")
        assert ctx.active_entities.get("name") == "Danashree"
        assert ctx.active_entities.get("order_id") == "ORD-77889"
        assert ctx.active_entities.get("product_code") == "PROD-BC99"
        assert ctx.active_entities.get("date") == "2026-09-25"


# =======================================================================
# Group D: Information Corrections
# =======================================================================

class TestGroupDCorrections:
    """Tests handling of customer corrections and maintaining historical accuracy."""

    def test_20_corrected_order_id(self):
        manager = ConversationContextManager()
        manager.add_message("test_conv_20", "user", "My order is ORD12345.")
        manager.add_message("test_conv_20", "user", "Sorry, the correct order ID is ORD54321.")

        active_order = manager.get_active_entity("test_conv_20", EntityType.ORDER_ID)
        assert active_order == "ORD54321"

    def test_21_corrected_date(self):
        manager = ConversationContextManager()
        manager.add_message("test_conv_21", "user", "My delivery date is 2026-09-20.")
        manager.add_message("test_conv_21", "user", "Actually, it is 2026-09-25.")

        active_date = manager.get_active_entity("test_conv_21", EntityType.DATE)
        assert active_date == "2026-09-25"

    def test_22_corrected_product_code(self):
        manager = ConversationContextManager()
        manager.add_message("test_conv_22", "user", "I enrolled in PROD-AX21.")
        manager.add_message("test_conv_22", "user", "Correction, the course is PROD-BX99.")

        active_code = manager.get_active_entity("test_conv_22", EntityType.PRODUCT_CODE)
        assert active_code == "PROD-BX99"

    def test_23_correction_preserves_history(self):
        manager = ConversationContextManager()
        manager.add_message("test_conv_23", "user", "My order is ORD12345.")
        manager.add_message("test_conv_23", "user", "Sorry, the correct order ID is ORD54321.")

        ctx = manager.get_context("test_conv_23")
        history = ctx.entity_history.get("order_id", [])
        assert "ORD12345" in history
        assert "ORD54321" in history
        assert len(ctx.corrections) == 1
        assert ctx.corrections[0].original_value == "ORD12345"
        assert ctx.corrections[0].corrected_value == "ORD54321"

    def test_24_latest_value_becomes_active_value(self):
        manager = ConversationContextManager()
        manager.add_message("test_conv_24", "user", "My order is ORD11111.")
        manager.add_message("test_conv_24", "user", "Actually, order is ORD22222.")
        manager.add_message("test_conv_24", "user", "Sorry, my mistake, the right order is ORD33333.")

        assert manager.get_active_entity("test_conv_24", EntityType.ORDER_ID) == "ORD33333"
        history = manager.get_entity_history("test_conv_24", EntityType.ORDER_ID)
        assert history == ["ORD11111", "ORD22222", "ORD33333"]


# =======================================================================
# Group E: Intent Detection & Confidence
# =======================================================================

class TestGroupEIntentHandling:
    """Tests intent detection, multi-intent parsing, and confidence scoring."""

    def test_25_simple_intent_detection(self):
        handler = IntentHandler()
        res = handler.detect_intents("Where is my order?")
        assert res.primary_intent == "order_status"
        assert res.confidence >= 0.70
        assert not res.is_multi_intent
        assert not res.requires_clarification

    def test_26_multiple_intents(self):
        handler = IntentHandler()
        res = handler.detect_intents("Where is my order and can I get a refund?")
        assert res.is_multi_intent is True
        intents = [c.intent for c in res.intents]
        assert "order_status" in intents
        assert "refund_request" in intents

    def test_27_multilingual_intent(self):
        handler = IntentHandler()
        # Tamil compound request
        ta_res = handler.detect_intents("என் order எங்கே இருக்கு மற்றும் refund கிடைக்குமா?")
        ta_intents = [c.intent for c in ta_res.intents]
        assert "order_status" in ta_intents
        assert "refund_request" in ta_intents

        # Hindi compound request
        hi_res = handler.detect_intents("मेरा order कहाँ है और क्या मुझे refund मिल सकता है?")
        hi_intents = [c.intent for c in hi_res.intents]
        assert "order_status" in hi_intents
        assert "refund_request" in hi_intents

        # Malayalam compound request
        ml_res = handler.detect_intents("എന്റെ order എവിടെയാണ് കൂടാതെ refund ലഭിക്കുമോ?")
        ml_intents = [c.intent for c in ml_res.intents]
        assert "order_status" in ml_intents
        assert "refund_request" in ml_intents

    def test_28_transliterated_intent(self):
        handler = IntentHandler()
        # Hinglish
        res_hi = handler.detect_intents("mera order kaha hai")
        assert res_hi.primary_intent == "order_status"

        # Tanglish
        res_ta = handler.detect_intents("en order enga iruku")
        assert res_ta.primary_intent == "order_status"

        # Manglish
        res_ml = handler.detect_intents("ente order evide aanu")
        assert res_ml.primary_intent == "order_status"

    def test_29_low_confidence_intent(self):
        handler = IntentHandler()
        res = handler.detect_intents("xyz qwerty completely unknown sentence")
        assert res.confidence < 0.70
        assert res.requires_clarification is True
        assert res.clarification_reason == "LOW_INTENT_CONFIDENCE"

    def test_30_clarification_on_low_confidence(self):
        handler = IntentHandler()
        res = handler.detect_intents("do something now")
        assert res.requires_clarification is True
        assert res.clarification_question is not None


# =======================================================================
# Group F: Ambiguity & Follow-up Resolution
# =======================================================================

class TestGroupFAmbiguityResolution:
    """Tests resolution of context follow-ups and genuine ambiguity handling."""

    def test_31_context_resolved_follow_up(self):
        manager = ConversationContextManager()
        manager.add_message("test_conv_31", "user", "My order is ORD12345.")
        ctx = manager.get_context("test_conv_31")

        handler = IntentHandler()
        res = handler.detect_intents("When will it arrive?", context=ctx)
        assert res.primary_intent == "order_status"
        assert res.requires_clarification is False
        assert res.intents[0].parameters.get("order_id") == "ORD12345"

    def test_32_ambiguous_follow_up(self):
        handler = IntentHandler()
        res = handler.detect_intents("Can you fix it?")
        assert res.requires_clarification is True
        assert res.clarification_reason == "AMBIGUOUS_INTENT"
        assert "course access, payment, or order" in res.clarification_question

    def test_33_multiple_possible_entities_require_clarification(self):
        manager = ConversationContextManager()
        # Customer mentions two orders without correction
        manager.add_message("test_conv_33", "user", "I placed ORD12345 and ORD54321.")
        ctx = manager.get_context("test_conv_33")

        handler = IntentHandler()
        # Follow-up "When will it arrive?" is ambiguous regarding which order
        res = handler.detect_intents("When will it arrive?", context=ctx)
        assert res.requires_clarification is True
        assert res.clarification_reason == "AMBIGUOUS_ENTITY"
        assert "ORD12345 or ORD54321" in res.clarification_question


# =======================================================================
# Group G: Configuration & Isolation
# =======================================================================

class TestGroupGConfigAndIsolation:
    """Tests configurability and per-conversation session isolation."""

    def test_34_custom_context_window(self):
        cfg = MultilingualConfig(context_window=5)
        manager = ConversationContextManager(config=cfg)

        for i in range(8):
            manager.add_message("test_conv_34", "user", f"Turn {i}")

        ctx = manager.get_context("test_conv_34")
        assert len(ctx.messages) == 5
        assert ctx.messages[0].original_text == "Turn 3"
        assert ctx.messages[-1].original_text == "Turn 7"

    def test_35_custom_intent_threshold(self):
        # With very high threshold, an intent with 0.80 confidence triggers clarification
        cfg = MultilingualConfig(intent_confidence_threshold=0.98)
        handler = IntentHandler(config=cfg)

        res = handler.detect_intents("order delivery status")
        assert res.requires_clarification is True
        assert res.clarification_reason == "LOW_INTENT_CONFIDENCE"

    def test_36_separate_context_ids_do_not_share_messages(self):
        manager = ConversationContextManager()
        manager.add_message("user_session_A", "user", "My order is ORD-AAA")
        manager.add_message("user_session_B", "user", "My order is ORD-BBB")

        ctx_a = manager.get_context("user_session_A")
        ctx_b = manager.get_context("user_session_B")

        assert len(ctx_a.messages) == 1
        assert len(ctx_b.messages) == 1
        assert ctx_a.active_entities["order_id"] == "ORD-AAA"
        assert ctx_b.active_entities["order_id"] == "ORD-BBB"
        assert "ORD-BBB" not in ctx_a.entity_history.get("order_id", [])
        assert "ORD-AAA" not in ctx_b.entity_history.get("order_id", [])


# =======================================================================
# Group H: General Informational Inquiries & Intent Priority Regressions
# =======================================================================

class TestGroupHGeneralInquiryAndIntentRegression:
    """Tests general_inquiry intent handling, priority over generic phrasing, and ambiguous clarification."""

    def test_37_general_inquiry_emi_payments(self):
        handler = IntentHandler()
        res = handler.detect_intents("Do you offer EMI payments?")
        assert res.primary_intent == "general_inquiry"
        assert res.confidence >= 0.70
        assert res.requires_clarification is False
        assert not res.is_multi_intent

    def test_38_general_inquiry_what_courses(self):
        handler = IntentHandler()
        res = handler.detect_intents("What courses do you offer?")
        assert res.primary_intent == "general_inquiry"
        assert res.confidence >= 0.70
        assert res.requires_clarification is False
        assert not res.is_multi_intent

    def test_39_general_inquiry_tell_me_courses(self):
        handler = IntentHandler()
        res = handler.detect_intents("Tell me about the courses")
        assert res.primary_intent == "general_inquiry"
        assert res.confidence >= 0.70
        assert res.requires_clarification is False
        assert not res.is_multi_intent

    def test_40_general_inquiry_payment_options(self):
        handler = IntentHandler()
        res = handler.detect_intents("What payment options are available?")
        assert res.primary_intent == "general_inquiry"
        assert res.confidence >= 0.70
        assert res.requires_clarification is False
        assert not res.is_multi_intent

    def test_41_regression_payment_failed_remains_payment_issue(self):
        handler = IntentHandler()
        res = handler.detect_intents("My payment failed")
        assert res.primary_intent == "payment_issue"
        assert res.confidence >= 0.70
        assert res.requires_clarification is False
        assert not res.is_multi_intent

    def test_42_regression_course_access_remains_course_access(self):
        handler = IntentHandler()
        res = handler.detect_intents("I cannot access my course")
        assert res.primary_intent == "course_access"
        assert res.confidence >= 0.70
        assert res.requires_clarification is False
        assert not res.is_multi_intent

    def test_43_regression_ambiguous_questions_still_request_clarification(self):
        handler = IntentHandler()
        ambiguous_inputs = [
            "Can you fix it?",
            "course",
            "payment",
            "what",
            "how",
            "xyz completely unknown query",
        ]
        for query in ambiguous_inputs:
            res = handler.detect_intents(query)
            assert res.requires_clarification is True, f"Expected clarification for: {query}"
            assert res.confidence < 0.70 or res.clarification_reason in ("AMBIGUOUS_INTENT", "LOW_INTENT_CONFIDENCE")
            assert res.clarification_question is not None

    def test_44_substantive_unmatched_questions_route_to_general_inquiry(self):
        handler = IntentHandler()
        questions = [
            "Should I learn Power BI or Tableau?",
            "Can I use Power BI on a Mac?",
            "Why should I trust Nullclass?",
        ]
        for query in questions:
            res = handler.detect_intents(query)
            assert res.requires_clarification is False, f"Unexpected clarification for: {query}"
            assert res.primary_intent == "general_inquiry"

    def test_45_gibberish_and_vague_imperatives_still_request_clarification(self):
        handler = IntentHandler()
        for query in [
            "xyz qwerty completely unknown sentence",
            "do something now",
            "repair the thing quickly",
        ]:
            res = handler.detect_intents(query)
            assert res.requires_clarification is True, f"Expected clarification for: {query}"
            assert res.clarification_reason == "LOW_INTENT_CONFIDENCE"
