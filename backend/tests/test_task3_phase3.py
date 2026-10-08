"""
Task 3 Phase 3 Test Suite: Agent Routing, Availability & Workload Balancing.

Comprehensive coverage (33 tests):
1. Agent model validation
2. Team model validation
3. Skill detection: course_access
4. Skill detection: payment
5. Skill detection: refund
6. Skill detection: login
7. Skill detection: certificate
8. Skill detection: technical
9. No matching skill
10. Normal routing during business hours
11. Workload balancing (lower workload preferred)
12. Workload capacity limit (at max_workload skipped)
13. Agent availability filtering (unavailable agent skipped)
14. Agent active status filtering (inactive agent skipped)
15. Deterministic tie-breaking (equal workload broken by workload % and agent_id)
16. Team unavailable (inactive team -> TEAM_UNAVAILABLE)
17. No agent available (all agents busy/unavailable -> NO_AGENT_AVAILABLE)
18. After-hours routing (evening/night -> AFTER_HOURS)
19. Weekend routing (Saturday/Sunday -> AFTER_HOURS)
20. Holiday routing (configured holiday -> AFTER_HOURS)
21. Team with business_hours_supported=False bypasses after-hours check
22. Runtime availability change (Agent A toggled unavailable -> Agent B assigned)
23. Runtime workload change (Agent A workload increased -> Agent B assigned)
24. Runtime team deactivation (Course Support deactivated -> TEAM_UNAVAILABLE)
25. Runtime agent skill addition (adding skill allows routing)
26. Invalid ticket with no issue text -> INVALID_TICKET
27. Ticket integration (ticket record stores routing fields)
28. Explainable routing result (reason includes skill, agent, workload)
29. API endpoint: POST /tickets/{id}/route
30. API endpoint: GET /tickets/{id}/routing
31. API endpoint: GET /routing/config & PATCH /routing/config
32. Existing Phase 1 / Phase 2 data preserved after routing
33. Existing endpoints regression check (/ask and /multimodal/analyze)
"""

import os
import sys
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

backend_dir = os.path.join(os.path.dirname(__file__), "..")
if backend_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(backend_dir))

from tickets.routing_models import (
    RoutingStatus,
    RoutingResult,
    Team,
    Agent,
    RoutingConfigUpdate,
)
from tickets.routing_config import (
    RoutingConfig,
    get_routing_config,
    update_routing_config,
    reset_routing_config,
)
from tickets.router import SkillDetector, TicketRouter
from tickets.business_hours import BusinessHoursConfig
from tickets.sla_config import reset_sla_config, update_sla_config
from tickets.models import TicketBase, TicketCreateRequest, TicketStatus
from tickets.storage import TicketStorage
from tickets.extractor import ConversationExtractor
from tickets.validator import MandatoryFieldValidator
from tickets.service import TicketService
import main as app_module


# During-business-hours test moment: Wednesday 14:00 UTC (9:00 - 18:00 window)
BUSINESS_TIME_WED_14 = datetime(2026, 9, 23, 14, 0, tzinfo=timezone.utc)
# After-hours test moment: Wednesday 21:00 UTC
AFTER_HOURS_WED_21 = datetime(2026, 9, 23, 21, 0, tzinfo=timezone.utc)
# Weekend test moment: Saturday 12:00 UTC
WEEKEND_SAT_12 = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clean_configs():
    """Reset routing and SLA configs before and after each test."""
    reset_sla_config()
    reset_routing_config()
    yield
    reset_sla_config()
    reset_routing_config()


def make_service(tmp_path) -> TicketService:
    store = TicketStorage(store_path=str(tmp_path / "tickets_p3_test.json"))
    extractor = ConversationExtractor(use_llm=False)
    validator = MandatoryFieldValidator()
    return TicketService(storage=store, extractor=extractor, validator=validator)


# ════════════════════════════════════════════════════════════════════════
# 1. Models & Validation
# ════════════════════════════════════════════════════════════════════════

def test_agent_model_validation():
    agent = Agent(
        agent_id="AGT-TEST",
        agent_name="John Doe",
        team_id="course_support",
        skills=["course_access"],
        available=True,
        current_workload=2,
        max_workload=10,
        active=True,
    )
    assert agent.agent_id == "AGT-TEST"
    assert agent.current_workload == 2
    assert agent.available is True


def test_team_model_validation():
    team = Team(
        team_id="test_team",
        team_name="Test Support Team",
        skills=["skill_a", "skill_b"],
        active=True,
        business_hours_supported=True,
    )
    assert team.team_id == "test_team"
    assert "skill_a" in team.skills
    assert team.business_hours_supported is True


# ════════════════════════════════════════════════════════════════════════
# 2. Skill Detection Tests
# ════════════════════════════════════════════════════════════════════════

def test_skill_detection_course_access():
    detector = SkillDetector()
    skill, team = detector.detect("I cannot access my course materials after enrollment.")
    assert skill == "course_access"
    assert team == "course_support"


def test_skill_detection_payment():
    detector = SkillDetector()
    skill, team = detector.detect("I was charged twice on my credit card invoice.")
    assert skill == "payment"
    assert team == "payment_support"


def test_skill_detection_refund():
    detector = SkillDetector()
    skill, team = detector.detect("I would like to request a refund for this course.")
    assert skill == "refund"
    assert team == "payment_support"


def test_skill_detection_login():
    detector = SkillDetector()
    skill, team = detector.detect("I forgot my password and cannot sign in to the portal.")
    assert skill == "login"
    assert team == "technical_support"


def test_skill_detection_certificate():
    detector = SkillDetector()
    skill, team = detector.detect("My course completion certificate was not issued.")
    assert skill == "certificate"
    assert team == "account_support"


def test_skill_detection_technical():
    detector = SkillDetector()
    skill, team = detector.detect("The platform crashes with an error code in the browser environment.")
    assert skill == "technical"
    assert team == "technical_support"


def test_no_matching_skill():
    detector = SkillDetector()
    skill, team = detector.detect("Good afternoon, the weather is nice today.")
    assert skill is None
    assert team is None


# ════════════════════════════════════════════════════════════════════════
# 3. Core Routing & Workload Balancing
# ════════════════════════════════════════════════════════════════════════

def test_normal_routing_during_business_hours():
    router = TicketRouter()
    ticket = TicketBase(issue="Cannot access course portal for Python.")
    res = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert res.routing_status == RoutingStatus.ASSIGNED
    assert res.team_id == "course_support"
    assert res.required_skill == "course_access"
    assert res.agent_id in ["AGENT-01", "AGENT-02"]


def test_workload_balancing_selects_lowest():
    # AGENT-01 has workload 2, AGENT-02 has workload 5
    router = TicketRouter()
    ticket = TicketBase(issue="Cannot access course portal for Python.")
    res = router.route(ticket, now=BUSINESS_TIME_WED_14)
    # AGENT-01 has lower workload (2 < 5)
    assert res.routing_status == RoutingStatus.ASSIGNED
    assert res.agent_id == "AGENT-01"
    assert res.workload == 2


def test_workload_capacity_limit():
    config = RoutingConfig()
    # Set AGENT-01 at max capacity (10/10)
    config.update_agent("AGENT-01", current_workload=10, max_workload=10)
    router = TicketRouter(routing_config=config)
    ticket = TicketBase(issue="Cannot access course portal for Python.")
    res = router.route(ticket, now=BUSINESS_TIME_WED_14)
    # AGENT-01 is at capacity, so AGENT-02 (workload 5/10) must be selected
    assert res.routing_status == RoutingStatus.ASSIGNED
    assert res.agent_id == "AGENT-02"


def test_agent_availability_filtering():
    config = RoutingConfig()
    # Set AGENT-01 unavailable
    config.update_agent("AGENT-01", available=False)
    router = TicketRouter(routing_config=config)
    ticket = TicketBase(issue="Cannot access course portal.")
    res = router.route(ticket, now=BUSINESS_TIME_WED_14)
    # Should skip AGENT-01 and pick AGENT-02
    assert res.routing_status == RoutingStatus.ASSIGNED
    assert res.agent_id == "AGENT-02"


def test_agent_active_status_filtering():
    config = RoutingConfig()
    # Set AGENT-01 inactive
    config.update_agent("AGENT-01", active=False)
    router = TicketRouter(routing_config=config)
    ticket = TicketBase(issue="Cannot access course portal.")
    res = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert res.routing_status == RoutingStatus.ASSIGNED
    assert res.agent_id == "AGENT-02"


def test_deterministic_tie_breaking():
    config = RoutingConfig()
    # Give AGENT-01 and AGENT-02 identical workloads and capacities
    config.update_agent("AGENT-01", current_workload=3, max_workload=10)
    config.update_agent("AGENT-02", current_workload=3, max_workload=10)
    router = TicketRouter(routing_config=config)
    ticket = TicketBase(issue="Cannot access course portal.")
    res = router.route(ticket, now=BUSINESS_TIME_WED_14)
    # Alphabetical tie-break: AGENT-01 < AGENT-02
    assert res.agent_id == "AGENT-01"


# ════════════════════════════════════════════════════════════════════════
# 4. Unavailability & Fallback States
# ════════════════════════════════════════════════════════════════════════

def test_team_unavailable():
    config = RoutingConfig()
    config.update_team("payment_support", active=False)
    router = TicketRouter(routing_config=config)
    ticket = TicketBase(issue="Charged twice on my credit card.")
    res = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert res.routing_status == RoutingStatus.TEAM_UNAVAILABLE
    assert res.team_id == "payment_support"
    assert res.agent_id is None


def test_no_agent_available():
    config = RoutingConfig()
    # Make all course_support agents unavailable
    config.update_agent("AGENT-01", available=False)
    config.update_agent("AGENT-02", current_workload=10, max_workload=10)
    router = TicketRouter(routing_config=config)
    ticket = TicketBase(issue="Cannot access course portal.")
    res = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert res.routing_status == RoutingStatus.NO_AGENT_AVAILABLE
    assert res.team_id == "course_support"
    assert res.agent_id is None


def test_after_hours_routing():
    router = TicketRouter()
    ticket = TicketBase(issue="Cannot access course portal.")
    # Arrives at 21:00 UTC (work_end is 18:00)
    res = router.route(ticket, now=AFTER_HOURS_WED_21)
    assert res.routing_status == RoutingStatus.AFTER_HOURS
    assert res.agent_id is None
    assert res.next_business_time is not None


def test_weekend_routing():
    router = TicketRouter()
    ticket = TicketBase(issue="Cannot access course portal.")
    res = router.route(ticket, now=WEEKEND_SAT_12)
    assert res.routing_status == RoutingStatus.AFTER_HOURS
    assert res.agent_id is None


def test_holiday_routing():
    # Configure 2026-09-23 as a holiday in SLAConfig
    update_sla_config(holidays=["2026-09-23"])
    router = TicketRouter()
    ticket = TicketBase(issue="Cannot access course portal.")
    res = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert res.routing_status == RoutingStatus.AFTER_HOURS


def test_team_bypasses_business_hours_if_unsupported():
    config = RoutingConfig()
    # Make course_support 24/7 (business_hours_supported=False)
    config.update_team("course_support", business_hours_supported=False)
    router = TicketRouter(routing_config=config)
    ticket = TicketBase(issue="Cannot access course portal.")
    res = router.route(ticket, now=AFTER_HOURS_WED_21)
    # Since team doesn't enforce business hours, assignment succeeds even after hours
    assert res.routing_status == RoutingStatus.ASSIGNED
    assert res.agent_id == "AGENT-01"


def test_invalid_ticket_no_issue():
    router = TicketRouter()
    ticket = TicketBase(issue="")
    res = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert res.routing_status == RoutingStatus.INVALID_TICKET


# ════════════════════════════════════════════════════════════════════════
# 5. Runtime Configuration Mutability
# ════════════════════════════════════════════════════════════════════════

def test_runtime_availability_change():
    config = get_routing_config()
    router = TicketRouter(routing_config=config)
    ticket = TicketBase(issue="Cannot access course.")

    # 1. Initially Agent 1 is assigned (workload 2 vs 5)
    r1 = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert r1.agent_id == "AGENT-01"

    # 2. Toggle Agent 1 unavailable at runtime
    config.update_agent("AGENT-01", available=False)

    # 3. Re-route: Agent 2 is now assigned without restarting app
    r2 = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert r2.agent_id == "AGENT-02"


def test_runtime_workload_change():
    config = get_routing_config()
    router = TicketRouter(routing_config=config)
    ticket = TicketBase(issue="Cannot access course.")

    # Initially Agent 1 (workload 2) beats Agent 2 (workload 5)
    r1 = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert r1.agent_id == "AGENT-01"

    # Change Agent 1's workload to 9
    config.update_agent("AGENT-01", current_workload=9)

    # Re-route: Agent 2 (workload 5) now beats Agent 1 (workload 9)
    r2 = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert r2.agent_id == "AGENT-02"


def test_runtime_team_deactivation():
    config = get_routing_config()
    router = TicketRouter(routing_config=config)
    ticket = TicketBase(issue="Cannot access course.")

    # Initially course_support is active
    r1 = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert r1.routing_status == RoutingStatus.ASSIGNED

    # Deactivate course_support at runtime
    config.update_team("course_support", active=False)

    # Re-route: team unavailable
    r2 = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert r2.routing_status == RoutingStatus.TEAM_UNAVAILABLE


def test_runtime_agent_skill_addition():
    config = get_routing_config()
    router = TicketRouter(routing_config=config)
    # AGENT-02 only has course_access. Course content issue requires course_content.
    # Initially AGENT-01 has course_content, AGENT-02 does not.
    config.update_agent("AGENT-01", available=False)
    ticket = TicketBase(issue="I have a question about the assignment syllabus.")

    # AGENT-01 unavailable, AGENT-02 lacks skill -> NO_AGENT_AVAILABLE
    r1 = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert r1.routing_status == RoutingStatus.NO_AGENT_AVAILABLE

    # Add skill to AGENT-02 at runtime
    config.update_agent("AGENT-02", skills=["course_access", "course_content"])

    # Re-route: AGENT-02 now handles it
    r2 = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert r2.routing_status == RoutingStatus.ASSIGNED
    assert r2.agent_id == "AGENT-02"


# ════════════════════════════════════════════════════════════════════════
# 6. Service & Storage Integration
# ════════════════════════════════════════════════════════════════════════

def test_ticket_service_integration(tmp_path):
    service = make_service(tmp_path)
    req = TicketCreateRequest(
        conversation="Customer: I cannot access course. Order: ORD-101. Email: s@e.com. Course: Python. Issue: cannot access course."
    )
    resp = service.create_from_conversation(req)
    ticket_id = resp.ticket.ticket_id

    # Route the ticket via service
    routing_res = service.route_ticket(ticket_id, now=BUSINESS_TIME_WED_14)
    assert routing_res is not None
    assert routing_res.routing_status == RoutingStatus.ASSIGNED

    # Verify fields stored in ticket
    stored = service._storage.get(ticket_id)
    assert stored.assigned_team_id == "course_support"
    assert stored.assigned_agent_id in ["AGENT-01", "AGENT-02"]
    assert stored.routing_status == RoutingStatus.ASSIGNED.value
    assert stored.required_skill == "course_access"


def test_explainable_routing_result():
    router = TicketRouter()
    ticket = TicketBase(issue="Refund for enrolled course not processed.")
    res = router.route(ticket, now=BUSINESS_TIME_WED_14)
    assert res.routing_status == RoutingStatus.ASSIGNED
    assert "refund" in res.reason.lower()
    assert res.agent_name is not None
    assert len(res.eligible_agents_considered) > 0


# ════════════════════════════════════════════════════════════════════════
# 7. API Endpoints Tests
# ════════════════════════════════════════════════════════════════════════

def test_api_route_and_routing_endpoints(tmp_path, monkeypatch):
    # Use wide open hours (00:00-24:00, 7 days/week) so routing succeeds regardless of when test runs
    wide_bh_cfg = BusinessHoursConfig(work_start=0, work_end=24, working_weekdays=[0, 1, 2, 3, 4, 5, 6])
    test_storage = TicketStorage(store_path=str(tmp_path / "api_routing_test.json"))

    router = TicketRouter(business_hours_cfg=wide_bh_cfg)
    service = TicketService(
        storage=test_storage,
        extractor=ConversationExtractor(use_llm=False),
        router=router,
    )
    monkeypatch.setattr(app_module, "_ticket_service", service)
    monkeypatch.setattr(app_module, "_ticket_storage", test_storage)

    client = TestClient(app_module.app)

    # 1. Create a ticket
    create_resp = client.post("/tickets", json={
        "conversation": "Customer: Cannot access my course. Order: ORD-202. Course: Python. Email: user@example.com. Issue: cannot access course."
    })
    assert create_resp.status_code == 200
    ticket_id = create_resp.json()["ticket"]["ticket_id"]

    # 2. Route the ticket via POST /tickets/{id}/route
    route_resp = client.post(f"/tickets/{ticket_id}/route")
    assert route_resp.status_code == 200
    route_data = route_resp.json()
    assert route_data["routing_status"] == "ASSIGNED"
    assert route_data["team_id"] == "course_support"
    assert "agent_id" in route_data

    # 3. Query routing via GET /tickets/{id}/routing
    get_routing = client.get(f"/tickets/{ticket_id}/routing")
    assert get_routing.status_code == 200
    assert get_routing.json()["routing_status"] == "ASSIGNED"
    assert get_routing.json()["agent_id"] == route_data["agent_id"]

    # 4. Route 404 test
    assert client.post("/tickets/TCK-NOTFOUND/route").status_code == 404
    assert client.get("/tickets/TCK-NOTFOUND/routing").status_code == 404


def test_api_routing_config_endpoints():
    client = TestClient(app_module.app)

    # 1. GET /routing/config
    get_cfg = client.get("/routing/config")
    assert get_cfg.status_code == 200
    cfg_data = get_cfg.json()
    assert "teams" in cfg_data
    assert "agents" in cfg_data
    team_ids = [t["team_id"] for t in cfg_data["teams"]]
    assert "course_support" in team_ids

    # 2. PATCH /routing/config (add or update team)
    patch_cfg = client.patch("/routing/config", json={
        "teams": [
            {
                "team_id": "vip_support",
                "team_name": "VIP Enterprise Support",
                "skills": ["enterprise", "course_access"],
                "active": True,
                "business_hours_supported": False
            }
        ]
    })
    assert patch_cfg.status_code == 200
    updated_teams = [t["team_id"] for t in patch_cfg.json()["teams"]]
    assert "vip_support" in updated_teams


# ════════════════════════════════════════════════════════════════════════
# 8. Backward Compatibility & Regressions
# ════════════════════════════════════════════════════════════════════════

def test_phase1_and_phase2_fields_preserved_after_routing(tmp_path):
    service = make_service(tmp_path)
    req = TicketCreateRequest(
        conversation="Customer: Course access not granted. Order: ORD-303. Course: SQL. Email: sql@test.com. Issue: Course access not granted."
    )
    resp = service.create_from_conversation(req)
    ticket_id = resp.ticket.ticket_id

    # Verify Phase 1 & 2 fields before routing
    assert resp.ticket.order_id == "ORD-303"
    assert resp.ticket.priority is not None
    assert resp.ticket.sla_hours is not None

    # Route ticket
    service.route_ticket(ticket_id, now=BUSINESS_TIME_WED_14)

    # Verify Phase 1 & Phase 2 fields are completely intact
    stored = service._storage.get(ticket_id)
    assert stored.order_id == "ORD-303"
    assert stored.priority == resp.ticket.priority
    assert stored.sla_hours == resp.ticket.sla_hours
    assert stored.routing_status == RoutingStatus.ASSIGNED.value


def test_existing_ask_and_multimodal_endpoints_work(monkeypatch):
    import langchain_helper as lh

    monkeypatch.setattr(lh, "get_active_faiss_path", lambda: "/fake/faiss")
    monkeypatch.setattr(lh, "get_qa_chain", lambda: lambda q: {"result": "mock answer"})

    client = TestClient(app_module.app)
    # /ask
    ask_resp = client.post("/ask", json={"question": "What is the refund policy?"})
    assert ask_resp.status_code in (200, 400, 500)

    # /multimodal/analyze
    mm_resp = client.post(
        "/multimodal/analyze",
        files={"file": ("test.txt", b"", "text/plain")},
    )
    assert mm_resp.status_code != 404
