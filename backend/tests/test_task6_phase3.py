"""
Task 6 Phase 3 — Test Suite: Session Management, Isolation, Expiry & Summary Restoration
========================================================================================
47 Comprehensive Behavioral Tests covering:
  - Group A: Session Creation (new session, unique IDs, timestamps, initial ACTIVE status)
  - Group B: Session Isolation (simultaneous customers, no leakage, different entities/languages)
  - Group C: Activity Tracking (last_activity_at updates, touch_session, continuous activity)
  - Group D: 30-Minute Expiry (29:59 active, 30:00 expired, 30:01 expired, expired session rejects msgs)
  - Group E: Configurable Inactivity Timeout (custom 15m, 45m, default 30m)
  - Group F: Deterministic Simulated Clock (advance seconds/minutes/hours/days, set_time, no sleep)
  - Group G: Summary Generation (intents, active entities, latest corrections, languages, non-PII)
  - Group H: 24-Hour Summary Restoration (RESTORED status, restored_from_session_id, boundaries at 23:59:59, 24:00:00, 24:00:01)
  - Group I: New Session After 24 Hours (status NEW, zero context leakage, clean entity state)
  - Group J: Security & Multi-Customer Isolation on Restoration (cross-customer restore rejected, latest session picked)
  - Group K: Configuration & Custom Windows (summary toggle, custom restore window)
  - Group L: Regression & End-to-End Compatibility (Phase 1 detection, Phase 2 multi-intent in session)
"""
import pytest

from backend.multilingual import (
    ConversationContextManager,
    CustomerSession,
    EntityType,
    MultilingualConfig,
    SessionManager,
    SessionStatus,
    SessionSummary,
    SimulatedClock,
)


# =======================================================================
# Group A: Session Creation
# =======================================================================

class TestGroupASessionCreation:
    """Tests session creation, unique identification, and initial state."""

    def test_01_create_new_session(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        session = manager.create_session(customer_id="cust_101")

        assert session is not None
        assert session.customer_id == "cust_101"
        assert session.status == SessionStatus.ACTIVE
        assert session.created_at == 1700000000.0
        assert session.last_activity_at == 1700000000.0
        assert session.session_id.startswith("sess_")
        assert session.conversation_id.startswith("conv_")

    def test_02_unique_session_ids(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s1 = manager.create_session(customer_id="cust_102")
        s2 = manager.create_session(customer_id="cust_102")

        assert s1.session_id != s2.session_id
        assert s1.conversation_id != s2.conversation_id

    def test_03_custom_session_id_and_metadata(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        meta = {"channel": "web_chat", "region": "IN"}
        session = manager.create_session(
            customer_id="cust_103",
            session_id="custom_sess_999",
            metadata=meta,
        )

        assert session.session_id == "custom_sess_999"
        assert session.metadata["channel"] == "web_chat"
        assert session.metadata["region"] == "IN"

    def test_04_initial_active_status(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        session = manager.create_session(customer_id="cust_104")

        assert manager.is_session_active(session.session_id) is True
        assert session.expired_at is None
        assert session.summary is None


# =======================================================================
# Group B: Simultaneous Session Isolation
# =======================================================================

class TestGroupBSessionIsolation:
    """Verifies complete data and context isolation between simultaneous customer sessions."""

    def test_05_two_customers_have_separate_sessions(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)

        s_a = manager.create_session(customer_id="cust_alice")
        s_b = manager.create_session(customer_id="cust_bob")

        manager.add_message(s_a.session_id, "user", "My order is ORD1001")
        manager.add_message(s_b.session_id, "user", "My order is ORD2002")

        ctx_a = manager.context_manager.get_context(s_a.conversation_id)
        ctx_b = manager.context_manager.get_context(s_b.conversation_id)

        assert ctx_a.active_entities.get("order_id") == "ORD1001"
        assert ctx_b.active_entities.get("order_id") == "ORD2002"

    def test_06_customer_a_order_id_never_leaked_to_customer_b(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)

        s_a = manager.create_session(customer_id="cust_alice")
        s_b = manager.create_session(customer_id="cust_bob")

        manager.add_message(s_a.session_id, "user", "My order is ORD1001")
        manager.add_message(s_b.session_id, "user", "My order is ORD2002")

        # Bob asks follow-up: "When will it arrive?"
        # The intent handler must resolve Bob's order ORD2002, NEVER Alice's ORD1001
        msg_b = manager.add_message(s_b.session_id, "user", "When will it arrive?")
        assert msg_b.intents[0].parameters.get("order_id") == "ORD2002"

        # Alice asks follow-up: "When will it arrive?"
        msg_a = manager.add_message(s_a.session_id, "user", "When will it arrive?")
        assert msg_a.intents[0].parameters.get("order_id") == "ORD1001"

    def test_07_same_customer_multiple_sessions_are_isolated(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)

        s1 = manager.create_session(customer_id="cust_multi")
        s2 = manager.create_session(customer_id="cust_multi")

        manager.add_message(s1.session_id, "user", "Order is ORD-1111")
        manager.add_message(s2.session_id, "user", "Order is ORD-2222")

        ctx1 = manager.context_manager.get_context(s1.conversation_id)
        ctx2 = manager.context_manager.get_context(s2.conversation_id)

        assert ctx1.active_entities.get("order_id") == "ORD-1111"
        assert ctx2.active_entities.get("order_id") == "ORD-2222"

    def test_08_different_languages_in_isolated_sessions(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)

        s_ta = manager.create_session(customer_id="cust_tamil")
        s_hi = manager.create_session(customer_id="cust_hindi")

        m_ta = manager.add_message(s_ta.session_id, "user", "என் order எங்கே?")
        m_hi = manager.add_message(s_hi.session_id, "user", "मेरा सवाल है")

        assert m_ta.detected_language == "ta"
        assert m_hi.detected_language == "hi"

    def test_09_different_corrections_in_simultaneous_sessions(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)

        s1 = manager.create_session(customer_id="cust_X")
        s2 = manager.create_session(customer_id="cust_Y")

        manager.add_message(s1.session_id, "user", "Order is ORD-AAA")
        manager.add_message(s1.session_id, "user", "Actually order is ORD-BBB")

        manager.add_message(s2.session_id, "user", "Order is ORD-XXX")

        ctx1 = manager.context_manager.get_context(s1.conversation_id)
        ctx2 = manager.context_manager.get_context(s2.conversation_id)

        assert ctx1.active_entities.get("order_id") == "ORD-BBB"
        assert len(ctx1.corrections) == 1
        assert ctx2.active_entities.get("order_id") == "ORD-XXX"
        assert len(ctx2.corrections) == 0


# =======================================================================
# Group C: Activity Tracking
# =======================================================================

class TestGroupCActivityTracking:
    """Verifies that last_activity_at accurately tracks user interactions."""

    def test_10_last_activity_at_updates_on_message(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_act_1")

        assert s.last_activity_at == 1700000000.0

        # Advance 15 minutes
        clock.advance(minutes=15)
        manager.add_message(s.session_id, "user", "Still have a question")

        sess = manager.get_session(s.session_id)
        assert sess.last_activity_at == 1700000000.0 + (15 * 60)

    def test_11_touch_session_updates_activity(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_act_2")

        clock.advance(minutes=10)
        res = manager.touch_session(s.session_id)
        assert res is True
        assert manager.get_session(s.session_id).last_activity_at == 1700000000.0 + 600

    def test_12_continuous_activity_keeps_session_active_past_30m(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_act_3")

        # User interacts every 20 minutes for an hour
        for i in range(3):
            clock.advance(minutes=20)
            manager.add_message(s.session_id, "user", f"Turn {i}")
            assert manager.is_session_active(s.session_id) is True

        # Total elapsed time is 60 minutes, but session is still ACTIVE because activity was frequent
        sess = manager.get_session(s.session_id)
        assert sess.status == SessionStatus.ACTIVE


# =======================================================================
# Group D: 30-Minute Inactivity Expiry
# =======================================================================

class TestGroupDInactivityExpiry:
    """Verifies exact boundary conditions for 30-minute inactivity timeout."""

    def test_13_boundary_29m_59s_remains_active(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_exp_1")

        # Advance 29 minutes and 59 seconds (1799 seconds)
        clock.advance(seconds=1799)
        assert manager.is_session_active(s.session_id) is True
        sess = manager.get_session(s.session_id)
        assert sess.status == SessionStatus.ACTIVE

    def test_14_boundary_30m_00s_becomes_expired(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_exp_2")

        # Advance exactly 30 minutes (1800 seconds)
        clock.advance(seconds=1800)
        assert manager.is_session_active(s.session_id) is False
        sess = manager.get_session(s.session_id)
        assert sess.status == SessionStatus.EXPIRED
        assert sess.expired_at == 1700000000.0 + 1800

    def test_15_boundary_30m_01s_is_expired(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_exp_3")

        # Advance 30 minutes and 1 second (1801 seconds)
        clock.advance(seconds=1801)
        assert manager.is_session_active(s.session_id) is False
        sess = manager.get_session(s.session_id)
        assert sess.status == SessionStatus.EXPIRED

    def test_16_add_message_on_expired_session_fails(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_exp_4")

        clock.advance(minutes=31)
        with pytest.raises(ValueError, match="expired due to inactivity"):
            manager.add_message(s.session_id, "user", "Hello again")

    def test_17_touch_session_on_expired_session_returns_false(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_exp_5")

        clock.advance(minutes=35)
        res = manager.touch_session(s.session_id)
        assert res is False

    def test_18_expire_inactive_sessions_batch(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)

        s1 = manager.create_session(customer_id="c1")
        clock.advance(minutes=15)
        s2 = manager.create_session(customer_id="c2")

        # Advance another 16 minutes (total 31m for s1, 16m for s2)
        clock.advance(minutes=16)

        expired = manager.expire_inactive_sessions()
        assert s1.session_id in expired
        assert s2.session_id not in expired
        assert manager.is_session_active(s2.session_id) is True


# =======================================================================
# Group E: Configurable Inactivity Timeout
# =======================================================================

class TestGroupEConfigurableTimeout:
    """Tests custom inactivity timeouts configured via MultilingualConfig."""

    def test_19_custom_timeout_15_minutes(self):
        clock = SimulatedClock(1700000000.0)
        cfg = MultilingualConfig(session_inactivity_minutes=15)
        manager = SessionManager(config=cfg, clock=clock)
        s = manager.create_session(customer_id="cust_conf_1")

        clock.advance(minutes=14, seconds=59)
        assert manager.is_session_active(s.session_id) is True

        clock.advance(seconds=2)
        assert manager.is_session_active(s.session_id) is False

    def test_20_custom_timeout_45_minutes(self):
        clock = SimulatedClock(1700000000.0)
        cfg = MultilingualConfig(session_inactivity_minutes=45)
        manager = SessionManager(config=cfg, clock=clock)
        s = manager.create_session(customer_id="cust_conf_2")

        # At 35 minutes, standard 30-min would expire, but 45-min stays active
        clock.advance(minutes=35)
        assert manager.is_session_active(s.session_id) is True

        clock.advance(minutes=11)
        assert manager.is_session_active(s.session_id) is False


# =======================================================================
# Group F: Deterministic Simulated Clock
# =======================================================================

class TestGroupFSimulatedClock:
    """Tests SimulatedClock functionality ensuring no sleep() is ever needed."""

    def test_21_advance_seconds(self):
        clock = SimulatedClock(100.0)
        assert clock.advance(seconds=45) == 145.0
        assert clock.now() == 145.0

    def test_22_advance_minutes(self):
        clock = SimulatedClock(100.0)
        assert clock.advance(minutes=10) == 700.0

    def test_23_advance_hours(self):
        clock = SimulatedClock(100.0)
        assert clock.advance(hours=2) == 100.0 + 7200.0

    def test_24_advance_days(self):
        clock = SimulatedClock(100.0)
        assert clock.advance(days=1) == 100.0 + 86400.0

    def test_25_set_time(self):
        clock = SimulatedClock(100.0)
        clock.set_time(5000.0)
        assert clock.now() == 5000.0


# =======================================================================
# Group G: Summary Generation
# =======================================================================

class TestGroupGSummaryGeneration:
    """Verifies that expired sessions generate accurate, entity-preserving summaries."""

    def test_26_summary_generated_on_expiry(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_sum_1")

        manager.add_message(s.session_id, "user", "My order is ORD1234")
        manager.add_message(s.session_id, "user", "When will it arrive?")

        # Advance past 30 minutes to trigger expiry
        clock.advance(minutes=31)
        sess = manager.get_session(s.session_id)
        assert sess.status == SessionStatus.EXPIRED
        assert sess.summary is not None
        assert sess.summary.customer_id == "cust_sum_1"
        assert sess.summary.session_id == s.session_id

    def test_27_summary_preserves_latest_correction(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_sum_2")

        manager.add_message(s.session_id, "user", "My order is ORD1234.")
        manager.add_message(s.session_id, "user", "Actually, the order ID is ORD1235.")

        clock.advance(minutes=30)
        sess = manager.get_session(s.session_id)
        summary = sess.summary
        assert summary is not None
        assert summary.active_entities.get("order_id") == "ORD1235"
        assert "ORD1235" in summary.summary_text
        assert "corrected from ORD1234 to ORD1235" in summary.summary_text

    def test_28_summary_preserves_all_business_entities(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_sum_3")

        manager.add_message(
            s.session_id,
            "user",
            "My name is Danashree, order ORD-2026-001 for PROD-AX21 on 2026-09-28",
        )
        clock.advance(minutes=30)
        sess = manager.get_session(s.session_id)
        entities = sess.summary.active_entities

        assert entities.get("name") == "Danashree"
        assert entities.get("order_id") == "ORD-2026-001"
        assert entities.get("product_code") == "PROD-AX21"
        assert entities.get("date") == "2026-09-28"

    def test_29_summary_preserves_multilingual_context(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_sum_4")

        manager.add_message(s.session_id, "user", "My order is ORD9988")
        manager.add_message(s.session_id, "user", "என் order எங்கே?")

        clock.advance(minutes=30)
        sess = manager.get_session(s.session_id)
        assert "en" in sess.summary.detected_languages
        assert "ta" in sess.summary.detected_languages


# =======================================================================
# Group H: 24-Hour Summary Restoration Boundaries
# =======================================================================

class TestGroupHSummaryRestorationBoundaries:
    """Tests exact 24-hour restoration eligibility boundaries."""

    def test_30_restoration_within_24h_creates_restored_session(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s1 = manager.create_session(customer_id="cust_rst_1")
        manager.add_message(s1.session_id, "user", "My order is ORD-7777")

        # Expire session at 30 min
        clock.advance(minutes=30)
        assert manager.is_session_active(s1.session_id) is False

        # Customer returns 2 hours after expiry (well within 24h)
        clock.advance(hours=2)
        new_sess = manager.restore_session(customer_id="cust_rst_1")

        assert new_sess.status == SessionStatus.RESTORED
        assert new_sess.restored_from_session_id == s1.session_id
        assert new_sess.session_id != s1.session_id

        # Verify active entities restored
        ctx = manager.context_manager.get_context(new_sess.conversation_id)
        assert ctx.active_entities.get("order_id") == "ORD-7777"

    def test_31_boundary_23h_59m_59s_is_restorable(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s1 = manager.create_session(customer_id="cust_rst_2")
        manager.add_message(s1.session_id, "user", "Order is ORD-8888")

        # Expire at 30m
        clock.advance(minutes=30)
        sess = manager.get_session(s1.session_id)
        assert sess.status == SessionStatus.EXPIRED

        # Advance 23 hours, 59 minutes, 59 seconds after expiry (86399 seconds)
        clock.advance(seconds=86399)

        eligible = manager.get_restorable_session(customer_id="cust_rst_2")
        assert eligible is not None
        assert eligible.session_id == s1.session_id

        restored = manager.restore_session(customer_id="cust_rst_2")
        assert restored.status == SessionStatus.RESTORED

    def test_32_boundary_24h_00m_00s_restoration_window_closed(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s1 = manager.create_session(customer_id="cust_rst_3")
        manager.add_message(s1.session_id, "user", "Order is ORD-9999")

        clock.advance(minutes=30)
        sess = manager.get_session(s1.session_id)
        assert sess.status == SessionStatus.EXPIRED

        # Advance exactly 24 hours after expiry (86400 seconds)
        clock.advance(seconds=86400)

        eligible = manager.get_restorable_session(customer_id="cust_rst_3")
        assert eligible is None

        # Attempt restore -> initiates NEW session
        new_sess = manager.restore_session(customer_id="cust_rst_3")
        assert new_sess.status == SessionStatus.NEW
        assert new_sess.restored_from_session_id is None

    def test_33_boundary_24h_00m_01s_window_closed(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s1 = manager.create_session(customer_id="cust_rst_4")
        manager.add_message(s1.session_id, "user", "Order is ORD-0000")

        clock.advance(minutes=30)
        manager.get_session(s1.session_id)

        # Advance 24 hours and 1 second after expiry
        clock.advance(seconds=86401)
        eligible = manager.get_restorable_session(customer_id="cust_rst_4")
        assert eligible is None


# =======================================================================
# Group I: New Session After 24 Hours
# =======================================================================

class TestGroupINewSessionAfter24Hours:
    """Verifies that returning after >24h provides a completely clean session."""

    def test_34_new_session_after_24h_has_no_prior_entities(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s1 = manager.create_session(customer_id="cust_clean_1")
        manager.add_message(s1.session_id, "user", "My order is ORD-OLD123")

        # Expire at 30 min + wait 25 hours
        clock.advance(minutes=30)
        clock.advance(hours=25)

        # Customer returns
        new_sess = manager.restore_session(customer_id="cust_clean_1")
        assert new_sess.status == SessionStatus.NEW

        ctx = manager.context_manager.get_context(new_sess.conversation_id)
        assert ctx.active_entities == {}
        assert "order_id" not in ctx.active_entities
        assert ctx.corrections == []

    def test_35_follow_up_in_new_session_does_not_assume_old_order(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s1 = manager.create_session(customer_id="cust_clean_2")
        manager.add_message(s1.session_id, "user", "My order is ORD-OLD999")

        clock.advance(minutes=30)
        clock.advance(hours=26)

        new_sess = manager.restore_session(customer_id="cust_clean_2")
        # Customer says "When will it arrive?" in the new session
        msg = manager.add_message(new_sess.session_id, "user", "When will it arrive?")
        # Without order in context, order_id is not assumed from Day 1
        assert msg.intents[0].parameters.get("order_id") is None


# =======================================================================
# Group J: Security & Multi-Customer Isolation on Restoration
# =======================================================================

class TestGroupJSecurityAndIsolationOnRestore:
    """Verifies that cross-customer restoration is strictly denied."""

    def test_36_customer_a_cannot_restore_customer_b_session(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)

        s_b = manager.create_session(customer_id="cust_bob")
        manager.add_message(s_b.session_id, "user", "My secret order is ORD-BOB-99")
        clock.advance(minutes=30)
        manager.get_session(s_b.session_id)

        # Alice attempts to restore Bob's session ID
        clock.advance(hours=1)
        eligible = manager.get_restorable_session(
            customer_id="cust_alice",
            session_id=s_b.session_id,
        )
        assert eligible is None

        # Alice attempts restore_session pointing to Bob's ID
        new_alice_sess = manager.restore_session(
            customer_id="cust_alice",
            previous_session_id=s_b.session_id,
        )
        assert new_alice_sess.status == SessionStatus.NEW
        assert new_alice_sess.restored_from_session_id is None

        ctx_alice = manager.context_manager.get_context(new_alice_sess.conversation_id)
        assert "ORD-BOB-99" not in ctx_alice.active_entities.values()

    def test_37_customer_with_multiple_expired_sessions_restores_latest(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)

        # Session 1: Order AAA
        s1 = manager.create_session(customer_id="cust_multi_exp")
        manager.add_message(s1.session_id, "user", "Order is ORD-AAA")
        clock.advance(minutes=30)
        manager.get_session(s1.session_id)

        # Session 2: Order BBB (1 hour later)
        clock.advance(hours=1)
        s2 = manager.create_session(customer_id="cust_multi_exp")
        manager.add_message(s2.session_id, "user", "Order is ORD-BBB")
        clock.advance(minutes=30)
        manager.get_session(s2.session_id)

        # Return 2 hours later without specifying session_id
        clock.advance(hours=2)
        restored = manager.restore_session(customer_id="cust_multi_exp")
        assert restored.restored_from_session_id == s2.session_id

        ctx = manager.context_manager.get_context(restored.conversation_id)
        assert ctx.active_entities.get("order_id") == "ORD-BBB"


# =======================================================================
# Group K: Configuration & Custom Windows
# =======================================================================

class TestGroupKConfigurableWindows:
    """Tests custom restore window configurations and summary disable toggle."""

    def test_38_summary_disabled_toggle(self):
        clock = SimulatedClock(1700000000.0)
        cfg = MultilingualConfig(summary_enabled=False)
        manager = SessionManager(config=cfg, clock=clock)

        s = manager.create_session(customer_id="cust_nosum")
        manager.add_message(s.session_id, "user", "Where is my order?")
        clock.advance(minutes=30)

        sess = manager.get_session(s.session_id)
        assert sess.status == SessionStatus.EXPIRED
        assert sess.summary is None

    def test_39_custom_restore_hours_12(self):
        clock = SimulatedClock(1700000000.0)
        cfg = MultilingualConfig(session_restore_hours=12)
        manager = SessionManager(config=cfg, clock=clock)

        s = manager.create_session(customer_id="cust_12h")
        manager.add_message(s.session_id, "user", "Order is ORD-12H")
        clock.advance(minutes=30)
        manager.get_session(s.session_id)

        # 11 hours after expiry: restorable
        clock.advance(hours=11)
        assert manager.get_restorable_session(customer_id="cust_12h") is not None

        # 12 hours after expiry: closed
        clock.advance(hours=1, seconds=1)
        assert manager.get_restorable_session(customer_id="cust_12h") is None


# =======================================================================
# Group L: Regression & End-to-End Compatibility
# =======================================================================

class TestGroupLRegressionCompatibility:
    """Verifies that Phase 1 language analysis and Phase 2 intents function seamlessly in sessions."""

    def test_40_phase1_analysis_preserved_in_session_message(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_reg_1")

        msg = manager.add_message(
            s.session_id,
            "user",
            "Mera course access nahi ho raha for PROD-AX21",
        )
        assert msg.detected_language == "hi"
        assert len(msg.preserved_entities) > 0
        assert msg.preserved_entities[0].value == "PROD-AX21"

    def test_41_phase2_multi_intent_decomposition_in_session(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_reg_2")

        msg = manager.add_message(
            s.session_id,
            "user",
            "Where is my order and can I get a refund?",
        )
        intents = [c.intent for c in msg.intents]
        assert "order_status" in intents
        assert "refund_request" in intents

    def test_42_explicit_create_new_session(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_new_session(customer_id="cust_new_explicit")
        assert s.status == SessionStatus.NEW

    def test_43_get_customer_sessions_list(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s1 = manager.create_session(customer_id="cust_list")
        s2 = manager.create_session(customer_id="cust_list")
        s_other = manager.create_session(customer_id="cust_other")

        sessions = manager.get_customer_sessions("cust_list")
        assert len(sessions) == 2
        sids = [s.session_id for s in sessions]
        assert s1.session_id in sids
        assert s2.session_id in sids
        assert s_other.session_id not in sids

    def test_44_non_existent_session_queries(self):
        manager = SessionManager()
        assert manager.get_session("ghost_sess") is None
        assert manager.is_session_active("ghost_sess") is False
        assert manager.touch_session("ghost_sess") is False
        assert manager.get_session_summary("ghost_sess") is None

    def test_45_summary_without_corrections(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_nocorr")
        manager.add_message(s.session_id, "user", "I need help with login")
        clock.advance(minutes=30)

        sess = manager.get_session(s.session_id)
        assert sess.summary is not None
        assert "Corrections:" not in sess.summary.summary_text
        assert "Intent: course_access" in sess.summary.summary_text

    def test_46_empty_conversation_summary(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s = manager.create_session(customer_id="cust_empty")
        clock.advance(minutes=30)

        sess = manager.get_session(s.session_id)
        assert sess.status == SessionStatus.EXPIRED
        assert sess.summary is not None
        assert sess.summary.summary_text == "Customer support discussion."

    def test_47_restore_with_explicit_session_id(self):
        clock = SimulatedClock(1700000000.0)
        manager = SessionManager(clock=clock)
        s1 = manager.create_session(customer_id="cust_exp_id")
        manager.add_message(s1.session_id, "user", "My order is ORD-TARGET")
        clock.advance(minutes=30)
        manager.get_session(s1.session_id)

        clock.advance(hours=3)
        restored = manager.restore_session(
            customer_id="cust_exp_id",
            previous_session_id=s1.session_id,
        )
        assert restored.status == SessionStatus.RESTORED
        assert restored.restored_from_session_id == s1.session_id
        ctx = manager.context_manager.get_context(restored.conversation_id)
        assert ctx.active_entities.get("order_id") == "ORD-TARGET"
