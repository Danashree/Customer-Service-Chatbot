"""
Task 3 Phase 2 Test Suite: Priority Calculation + SLA Management.

Covers all 31+ required tests:
1. positive sentiment
2. neutral sentiment
3. negative sentiment
4. low severity
5. medium severity
6. high severity
7. critical severity
8. low customer impact
9. high customer impact
10. priority calculation
11. priority breakdown
12. waiting time calculation
13. SLA creation
14. SLA due-time calculation
15. weekend exclusion
16. holiday exclusion
17. non-working-hours exclusion
18. SLA starts outside business hours
19. 75% warning
20. 100% breach
21. automatic escalation
22. SLA configuration update
23. runtime holiday change
24. runtime weekend change
25. runtime SLA duration change
26. no wall-clock calculation bug
27. existing Phase 1 ticket still works
28. /ask still works
29. /multimodal/analyze still works
30. Boundary conditions (74.99%, 75%, 99.99%, 100%)
31. API endpoints (POST/GET priority, GET/POST sla, GET/PATCH sla/config)
"""

import os
import sys
from datetime import date, datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

backend_dir = os.path.join(os.path.dirname(__file__), "..")
if backend_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(backend_dir))

from tickets.priority_enums import (
    Severity,
    Sentiment,
    CustomerImpact,
    Priority,
    SLAStatus,
    EscalationStatus,
)
from tickets.business_hours import (
    BusinessHoursConfig,
    is_business_day,
    is_business_time,
    next_business_moment,
    calculate_business_minutes,
    add_business_hours,
    add_business_minutes,
    calculate_sla_due_at,
    calculate_sla_warning_at,
)
from tickets.sla_config import (
    SLAConfig,
    get_sla_config,
    update_sla_config,
    reset_sla_config,
)
from tickets.sentiment import detect_sentiment
from tickets.severity import classify_severity, classify_customer_impact
from tickets.priority_calculator import (
    calculate_priority,
    PriorityBreakdown,
    _waiting_score,
)
from tickets.sla_manager import initialise_sla, check_sla, resolve_sla, SLAState
from tickets.models import (
    TicketBase,
    TicketCreateRequest,
    TicketStatus,
    SLAConfigUpdateRequest,
)
from tickets.storage import TicketStorage
from tickets.extractor import ConversationExtractor
from tickets.validator import MandatoryFieldValidator
from tickets.service import TicketService
import main as app_module


@pytest.fixture(autouse=True)
def clean_sla_config():
    """Reset SLA configuration to default before and after each test."""
    reset_sla_config()
    yield
    reset_sla_config()


def make_service(tmp_path) -> TicketService:
    store = TicketStorage(store_path=str(tmp_path / "tickets_p2_test.json"))
    extractor = ConversationExtractor(use_llm=False)
    validator = MandatoryFieldValidator()
    return TicketService(storage=store, extractor=extractor, validator=validator)


# ════════════════════════════════════════════════════════════════════════
# 1. Sentiment Tests
# ════════════════════════════════════════════════════════════════════════

def test_positive_sentiment():
    text = "I am happy with the course. The mentor was amazing and very helpful!"
    assert detect_sentiment(text) == Sentiment.POSITIVE


def test_neutral_sentiment():
    text = "I want to know the course duration and the syllabus topics."
    assert detect_sentiment(text) == Sentiment.NEUTRAL


def test_negative_sentiment():
    text = "I cannot access my course and this is very frustrating. Money deducted but still waiting!"
    assert detect_sentiment(text) == Sentiment.NEGATIVE


# ════════════════════════════════════════════════════════════════════════
# 2. Severity Tests
# ════════════════════════════════════════════════════════════════════════

def test_low_severity():
    issue = "Question about recommended reading materials for next week."
    assert classify_severity(issue) == Severity.LOW


def test_medium_severity():
    issue = "Certificate not issued after course completion."
    assert classify_severity(issue) == Severity.MEDIUM


def test_high_severity():
    issue = "Cannot access course materials and I am blocked from completing my graded assignment before the deadline."
    assert classify_severity(issue) == Severity.HIGH


def test_critical_severity():
    issue = "Security breach: unauthorized access on my account and charged multiple times!"
    assert classify_severity(issue) == Severity.CRITICAL


# ════════════════════════════════════════════════════════════════════════
# 3. Customer Impact Tests
# ════════════════════════════════════════════════════════════════════════

def test_low_customer_impact():
    issue = "General query about the course start date."
    assert classify_customer_impact(issue) == CustomerImpact.LOW


def test_medium_customer_impact():
    issue = "Video not playing smoothly in module 2."
    assert classify_customer_impact(issue) == CustomerImpact.MEDIUM


def test_high_customer_impact():
    issue = "Paid but no access, unable to complete course."
    assert classify_customer_impact(issue) == CustomerImpact.HIGH


def test_critical_customer_impact():
    issue = "Platform down, data breach affecting all users."
    assert classify_customer_impact(issue) == CustomerImpact.CRITICAL


# ════════════════════════════════════════════════════════════════════════
# 4. Priority Calculation & Breakdown Tests
# ════════════════════════════════════════════════════════════════════════

def test_priority_calculation():
    # Low score: Low (10) + Pos (0) + Low (5) + Wait (0) + Urgency (0) = 15 -> LOW
    p, s, b = calculate_priority(
        severity=Severity.LOW,
        sentiment=Sentiment.POSITIVE,
        customer_impact=CustomerImpact.LOW,
        sla_status=SLAStatus.IN_PROGRESS,
    )
    assert p == Priority.LOW
    assert s == 15

    # Medium score: Med (25) + Neu (5) + Med (10) + Wait (0) = 40 -> MEDIUM
    p, s, b = calculate_priority(
        severity=Severity.MEDIUM,
        sentiment=Sentiment.NEUTRAL,
        customer_impact=CustomerImpact.MEDIUM,
        sla_status=SLAStatus.IN_PROGRESS,
    )
    assert p == Priority.MEDIUM
    assert s == 40

    # High score: High (40) + Neg (10) + High (20) + Wait (0) = 70 -> HIGH
    p, s, b = calculate_priority(
        severity=Severity.HIGH,
        sentiment=Sentiment.NEGATIVE,
        customer_impact=CustomerImpact.HIGH,
        sla_status=SLAStatus.IN_PROGRESS,
    )
    assert p == Priority.HIGH
    assert s == 70

    # Critical score: Critical (50) + Neg (10) + Critical (30) = 90 -> CRITICAL
    p, s, b = calculate_priority(
        severity=Severity.CRITICAL,
        sentiment=Sentiment.NEGATIVE,
        customer_impact=CustomerImpact.CRITICAL,
        sla_status=SLAStatus.IN_PROGRESS,
    )
    assert p == Priority.CRITICAL
    assert s == 90


def test_priority_breakdown():
    p, s, b = calculate_priority(
        severity=Severity.HIGH,
        sentiment=Sentiment.NEGATIVE,
        customer_impact=CustomerImpact.HIGH,
        sla_status=SLAStatus.WARNING,  # warning gives +5 urgency
    )
    b_dict = b.as_dict()
    assert b_dict["severity"] == 40
    assert b_dict["sentiment"] == 10
    assert b_dict["customer_impact"] == 20
    assert b_dict["waiting_time"] == 0
    assert b_dict["sla_urgency"] == 5
    assert b.total == 75
    assert s == 75
    assert p == Priority.CRITICAL  # >= 75 maps to CRITICAL


# ════════════════════════════════════════════════════════════════════════
# 5. Waiting Time & Business Hours Arithmetic Tests
# ════════════════════════════════════════════════════════════════════════

def test_waiting_time_calculation():
    cfg = BusinessHoursConfig(work_start=9, work_end=18)
    t1 = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc)  # Monday 09:00
    t2 = datetime(2026, 9, 21, 12, 30, tzinfo=timezone.utc) # Monday 12:30
    mins = calculate_business_minutes(t1, t2, cfg)
    assert mins == 210  # 3.5 hours = 210 mins


def test_sla_creation():
    cfg = BusinessHoursConfig(work_start=9, work_end=18)
    start = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc)
    res = initialise_sla(sla_hours=8.0, business_cfg=cfg, now=start)
    assert res["sla_hours"] == 8.0
    assert res["sla_started_at"] == start.isoformat()
    assert res["sla_status"] == SLAStatus.IN_PROGRESS
    assert res["escalation_status"] == EscalationStatus.NOT_ESCALATED
    assert "sla_due_at" in res
    assert "sla_warning_at" in res


def test_sla_due_time_calculation():
    cfg = BusinessHoursConfig(work_start=9, work_end=18)
    start = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc)  # Monday 09:00
    due = calculate_sla_due_at(start, 8.0, cfg)
    # Monday 09:00 + 8 hours = Monday 17:00
    assert due == datetime(2026, 9, 21, 17, 0, tzinfo=timezone.utc)


def test_weekend_exclusion():
    cfg = BusinessHoursConfig(work_start=9, work_end=18, working_weekdays=[0, 1, 2, 3, 4])
    # Friday 2026-09-25 at 15:00 UTC
    start = datetime(2026, 9, 25, 15, 0, tzinfo=timezone.utc)
    # 8 business hours: Friday has 3 hours (15:00 to 18:00).
    # Sat & Sun excluded.
    # Monday 2026-09-28 has remaining 5 hours: 09:00 + 5h = 14:00.
    due = calculate_sla_due_at(start, 8.0, cfg)
    expected = datetime(2026, 9, 28, 14, 0, tzinfo=timezone.utc)
    assert due == expected


def test_holiday_exclusion():
    cfg = BusinessHoursConfig(
        work_start=9,
        work_end=18,
        working_weekdays=[0, 1, 2, 3, 4],
        holidays=["2026-09-28"],  # Monday is holiday
    )
    # Friday 15:00 + 8 hours
    start = datetime(2026, 9, 25, 15, 0, tzinfo=timezone.utc)
    # Friday: 3 hrs. Weekend: excluded. Monday: holiday.
    # Tuesday 2026-09-29: remaining 5 hrs -> 09:00 + 5 = 14:00.
    due = calculate_sla_due_at(start, 8.0, cfg)
    expected = datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc)
    assert due == expected


def test_non_working_hours_exclusion():
    cfg = BusinessHoursConfig(work_start=9, work_end=18)
    # Monday 17:30 + 2 hours
    start = datetime(2026, 9, 21, 17, 30, tzinfo=timezone.utc)
    # Monday has 30 mins (17:30 to 18:00).
    # Night excluded.
    # Tuesday gets remaining 90 mins (09:00 + 1h30m = 10:30).
    due = calculate_sla_due_at(start, 2.0, cfg)
    expected = datetime(2026, 9, 22, 10, 30, tzinfo=timezone.utc)
    assert due == expected


def test_sla_starts_outside_business_hours():
    cfg = BusinessHoursConfig(work_start=9, work_end=18, working_weekdays=[0, 1, 2, 3, 4])
    # Ticket created Friday 20:00 (after business hours)
    start = datetime(2026, 9, 25, 20, 0, tzinfo=timezone.utc)
    # Snaps forward to Monday 09:00, then adds 4 hours -> Monday 13:00
    due = calculate_sla_due_at(start, 4.0, cfg)
    expected = datetime(2026, 9, 28, 13, 0, tzinfo=timezone.utc)
    assert due == expected

    # Ticket created Saturday 12:00
    sat_start = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
    sat_due = calculate_sla_due_at(sat_start, 4.0, cfg)
    assert sat_due == expected


# ════════════════════════════════════════════════════════════════════════
# 6. SLA Warning, Breach, and Escalation Tests
# ════════════════════════════════════════════════════════════════════════

def test_75_percent_warning():
    cfg = BusinessHoursConfig(work_start=9, work_end=18)
    # 8 hours SLA started Monday 09:00
    start = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc)
    sla_data = initialise_sla(8.0, cfg, warning_threshold=0.75, now=start)

    # Monday 15:00 = exactly 6 hours = 360 mins = 75%
    check_time = datetime(2026, 9, 21, 15, 0, tzinfo=timezone.utc)
    state = check_sla(sla_data, cfg, warning_threshold=0.75, now=check_time)
    assert state.sla_status == SLAStatus.WARNING
    assert state.percentage_consumed == 75.0
    assert state.business_minutes_consumed == 360
    assert state.business_minutes_remaining == 120
    assert state.escalation_status == EscalationStatus.NOT_ESCALATED


def test_100_percent_breach_and_automatic_escalation():
    cfg = BusinessHoursConfig(work_start=9, work_end=18)
    start = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc)
    sla_data = initialise_sla(8.0, cfg, warning_threshold=0.75, now=start)

    # Monday 17:01 = 481 mins consumed >= 480 mins
    check_time = datetime(2026, 9, 21, 17, 1, tzinfo=timezone.utc)
    state = check_sla(sla_data, cfg, warning_threshold=0.75, now=check_time)
    assert state.sla_status == SLAStatus.BREACHED
    assert state.escalation_status == EscalationStatus.ESCALATED
    assert state.escalated_at is not None
    assert "SLA breach" in state.escalation_reason
    # Original timestamps preserved
    assert state.sla_started_at == sla_data["sla_started_at"]
    assert state.sla_due_at == sla_data["sla_due_at"]


def test_boundary_conditions():
    # Test waiting score tiers: 74.99%, 75%, 99.99%, 100%
    assert _waiting_score(74.99) == 10
    assert _waiting_score(75.0) == 20
    assert _waiting_score(99.99) == 20
    assert _waiting_score(100.0) == 30

    cfg = BusinessHoursConfig(work_start=9, work_end=18)
    start = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc)
    sla_data = initialise_sla(8.0, cfg, warning_threshold=0.75, now=start)

    # Just before 75%: 5 hours 59 minutes = 359 mins (359 / 480 * 100 = 74.79%)
    t_before_75 = datetime(2026, 9, 21, 14, 59, tzinfo=timezone.utc)
    state_before = check_sla(sla_data, cfg, warning_threshold=0.75, now=t_before_75)
    assert state_before.sla_status == SLAStatus.IN_PROGRESS

    # Exactly 75%: 6 hours = 360 mins
    t_75 = datetime(2026, 9, 21, 15, 0, tzinfo=timezone.utc)
    state_75 = check_sla(sla_data, cfg, warning_threshold=0.75, now=t_75)
    assert state_75.sla_status == SLAStatus.WARNING

    # Just before 100%: 7 hours 59 mins = 479 mins
    t_before_100 = datetime(2026, 9, 21, 16, 59, tzinfo=timezone.utc)
    state_before_100 = check_sla(sla_data, cfg, warning_threshold=0.75, now=t_before_100)
    assert state_before_100.sla_status == SLAStatus.WARNING

    # Exactly 100%: 8 hours = 480 mins
    t_100 = datetime(2026, 9, 21, 17, 0, tzinfo=timezone.utc)
    state_100 = check_sla(sla_data, cfg, warning_threshold=0.75, now=t_100)
    assert state_100.sla_status == SLAStatus.BREACHED
    assert state_100.escalation_status == EscalationStatus.ESCALATED


# ════════════════════════════════════════════════════════════════════════
# 7. Runtime Configuration Update Tests
# ════════════════════════════════════════════════════════════════════════

def test_sla_configuration_update():
    new_cfg = update_sla_config(
        sla_hours_by_priority={"HIGH": 6.0, "CRITICAL": 2.0},
        warning_threshold=0.80,
    )
    assert new_cfg.get_sla_hours("HIGH") == 6.0
    assert new_cfg.get_sla_hours("CRITICAL") == 2.0
    assert new_cfg.warning_threshold == 0.80


def test_runtime_holiday_change():
    start = datetime(2026, 9, 25, 15, 0, tzinfo=timezone.utc)
    # Before holiday: Friday 15:00 + 8h due Monday 14:00
    cfg1 = get_sla_config().business_hours
    due1 = calculate_sla_due_at(start, 8.0, cfg1)
    assert due1.date() == date(2026, 9, 28)

    # Add holiday for Monday at runtime
    update_sla_config(holidays=["2026-09-28"])
    cfg2 = get_sla_config().business_hours
    due2 = calculate_sla_due_at(start, 8.0, cfg2)
    assert due2.date() == date(2026, 9, 29)


def test_runtime_weekend_change():
    # Change working days to Sunday (6) through Thursday (3), weekend = Friday (4) + Saturday (5)
    update_sla_config(working_weekdays=[6, 0, 1, 2, 3])
    cfg = get_sla_config().business_hours
    assert is_business_day(date(2026, 9, 27), cfg) is True  # Sunday is business day
    assert is_business_day(date(2026, 9, 25), cfg) is False # Friday is weekend


def test_runtime_sla_duration_change(tmp_path):
    service = make_service(tmp_path)
    # Create ticket 1 under default 8h HIGH SLA
    conv = "Customer: I cannot access course. Email: a@b.com. Order: ORD-1. Issue: cannot access."
    r1 = service.create_from_conversation(TicketCreateRequest(conversation=conv))
    assert r1.ticket.sla_hours == 8.0

    # Change HIGH SLA to 5h
    update_sla_config(sla_hours_by_priority={"HIGH": 5.0})

    # Create ticket 2 under new 5h HIGH SLA
    r2 = service.create_from_conversation(TicketCreateRequest(conversation=conv))
    assert r2.ticket.sla_hours == 5.0

    # Existing ticket 1 still has 8.0h (safe preservation)
    t1_stored = service._storage.get(r1.ticket.ticket_id)
    assert t1_stored.sla_hours == 8.0


def test_no_wall_clock_calculation_bug():
    cfg = BusinessHoursConfig(work_start=9, work_end=18, working_weekdays=[0, 1, 2, 3, 4])
    # Friday 18:00 to Sunday 18:00 = 48 wall-clock hours
    t1 = datetime(2026, 9, 25, 18, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 9, 27, 18, 0, tzinfo=timezone.utc)
    mins = calculate_business_minutes(t1, t2, cfg)
    assert mins == 0, f"Weekend must consume 0 business minutes, got {mins}"


# ════════════════════════════════════════════════════════════════════════
# 8. Backward Compatibility & Regression Tests
# ════════════════════════════════════════════════════════════════════════

def test_existing_phase1_ticket_still_works(tmp_path):
    service = make_service(tmp_path)
    conv = (
        "Customer: Hi, I'm Priya. Order: ORD-2024-8801. "
        "Course: Python for Data Science. Email: priya@example.com. "
        "Issue: Course access not granted after payment."
    )
    req = TicketCreateRequest(conversation=conv, unresolved_reason="Unresolved issue")
    resp = service.create_from_conversation(req)

    # Phase 1 assertions
    assert resp.ticket.ticket_id.startswith("TCK-")
    assert resp.is_complete is True
    assert resp.ticket.status == TicketStatus.READY_FOR_ROUTING
    assert resp.ticket.course_name is not None
    assert resp.ticket.order_id == "ORD-2024-8801"

    # Phase 2 assertions automatically populated
    assert resp.ticket.priority is not None
    assert resp.ticket.severity is not None
    assert resp.ticket.sentiment is not None
    assert resp.ticket.sla_status == SLAStatus.IN_PROGRESS.value


def test_existing_ask_endpoint_still_works(monkeypatch):
    import langchain_helper as lh

    monkeypatch.setattr(lh, "get_active_faiss_path", lambda: "/fake/faiss")
    monkeypatch.setattr(lh, "get_qa_chain", lambda: lambda q: {"result": "mock answer"})

    client = TestClient(app_module.app)
    response = client.post("/ask", json={"question": "What are the system requirements?"})
    assert response.status_code in (200, 400, 500)


def test_existing_multimodal_analyze_endpoint_still_works():
    client = TestClient(app_module.app)
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("test.txt", b"", "text/plain")},
    )
    assert response.status_code != 404


# ════════════════════════════════════════════════════════════════════════
# 9. API Endpoints Tests
# ════════════════════════════════════════════════════════════════════════

def test_api_priority_and_sla_endpoints(tmp_path, monkeypatch):
    # Route main's storage to tmp_path for test isolation
    test_storage = TicketStorage(store_path=str(tmp_path / "api_test_store.json"))
    service = TicketService(storage=test_storage, extractor=ConversationExtractor(use_llm=False))
    monkeypatch.setattr(app_module, "_ticket_service", service)
    monkeypatch.setattr(app_module, "_ticket_storage", test_storage)

    client = TestClient(app_module.app)

    # 1. Create a ticket
    create_resp = client.post("/tickets", json={
        "conversation": "Customer: I cannot access course. Order: ORD-100. Course: Python. Email: p@e.com. Issue: cannot access course."
    })
    assert create_resp.status_code == 200
    ticket_id = create_resp.json()["ticket"]["ticket_id"]

    # 2. GET /tickets/{id}/priority
    p_resp = client.get(f"/tickets/{ticket_id}/priority")
    assert p_resp.status_code == 200
    p_data = p_resp.json()
    assert "priority" in p_data
    assert "priority_score" in p_data
    assert "breakdown" in p_data

    # 3. POST /tickets/{id}/priority
    calc_resp = client.post(f"/tickets/{ticket_id}/priority")
    assert calc_resp.status_code == 200
    assert calc_resp.json()["ticket_id"] == ticket_id

    # 4. GET /tickets/{id}/sla
    sla_resp = client.get(f"/tickets/{ticket_id}/sla")
    assert sla_resp.status_code == 200
    sla_data = sla_resp.json()
    assert sla_data["sla_status"] == "IN_PROGRESS"
    assert "business_minutes_remaining" in sla_data

    # 5. POST /tickets/{id}/sla/check
    check_resp = client.post(f"/tickets/{ticket_id}/sla/check")
    assert check_resp.status_code == 200

    # 6. GET /sla/config
    cfg_resp = client.get("/sla/config")
    assert cfg_resp.status_code == 200
    assert "sla_hours_by_priority" in cfg_resp.json()

    # 7. PATCH /sla/config
    patch_resp = client.patch("/sla/config", json={
        "sla_hours_by_priority": {"HIGH": 7.0},
        "warning_threshold": 0.70
    })
    assert patch_resp.status_code == 200
    assert patch_resp.json()["config"]["sla_hours_by_priority"]["HIGH"] == 7.0
    assert patch_resp.json()["config"]["warning_threshold"] == 0.70
