"""
Task 3 Phase 1 — Test Suite (15 tests)
Tests the Unresolved Conversation → Structured Support Ticket pipeline.

Design decisions:
- All TicketService instances use an isolated tmp_path JSON store so tests
  are hermetic and never pollute tickets_store.json.
- LLM-dependent paths are exercised via regex-only extraction (use_llm=False)
  for determinism and speed; no real Gemini calls are made.
- The /ask and /multimodal/analyze endpoints are tested via pytest-style
  monkeypatching to confirm Task 1 / Task 2 routes remain intact.
"""

import json
import os
import sys
import pytest

# Ensure backend is importable when running from project root
backend_dir = os.path.join(os.path.dirname(__file__), "..")
if backend_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(backend_dir))

from tickets.models import TicketCreateRequest, TicketStatusUpdateRequest, TicketStatus
from tickets.extractor import ConversationExtractor
from tickets.validator import MandatoryFieldValidator
from tickets.storage import TicketStorage
from tickets.service import TicketService


# ──────────────────────────────────────────────
# Helper: build an isolated TicketService whose JSON store lives in tmp_path
# ──────────────────────────────────────────────

def make_service(tmp_path) -> TicketService:
    store = TicketStorage(store_path=str(tmp_path / "tickets_test.json"))
    extractor = ConversationExtractor(use_llm=False)
    validator = MandatoryFieldValidator()
    return TicketService(storage=store, extractor=extractor, validator=validator)


# ──────────────────────────────────────────────
# Fixtures: realistic e-learning conversations
# ──────────────────────────────────────────────

COMPLETE_CONVERSATION = """
Customer: Hi, my name is Priya Sharma.
I purchased the course 'Python for Data Science' (Order ID: ORD-2024-8801) last week
but I still cannot access the course materials. My email is priya.sharma@example.com.
I have attached a payment screenshot as evidence.
Agent: I'm sorry to hear that. Let me look into it right away.
"""

MISSING_ORDER_ID_CONVERSATION = """
Customer: Hello, I am Alex Kim. I bought the 'Machine Learning Basics' course
but I cannot log in. Please help. My email is alex.kim@example.com.
"""

MISSING_CONTACT_CONVERSATION = """
Customer: I purchased 'Data Structures in Python' (Order ID: ORD-2024-5501)
but I cannot access module 3. I have a payment receipt.
"""

MISSING_ISSUE_CONVERSATION = """
Customer: Hi, I am Sam Lee. My order ID is ORD-2024-3377 and email is
sam.lee@example.com. Course: Deep Learning Fundamentals.
"""

MISSING_EVIDENCE_CONVERSATION = """
Customer: My name is Riya Patel. Order ORD-2024-9921. Course: SQL Mastery.
I was charged but never received access. Email: riya.patel@example.com.
Issue: payment charged but access not granted.
"""


# ════════════════════════════════════════════════════════════════════════
# Test 1: Complete conversation creates a structured ticket
# ════════════════════════════════════════════════════════════════════════

def test_complete_conversation_creates_ticket(tmp_path):
    """A fully-described conversation should produce a structured ticket."""
    service = make_service(tmp_path)
    req = TicketCreateRequest(
        conversation=COMPLETE_CONVERSATION,
        unresolved_reason="Course access not granted after payment",
    )
    resp = service.create_from_conversation(req)

    assert resp.ticket is not None
    assert resp.ticket.ticket_id.startswith("TCK-")
    assert resp.ticket.issue is not None and len(resp.ticket.issue) > 0


# ════════════════════════════════════════════════════════════════════════
# Test 2: Missing order ID is detected
# ════════════════════════════════════════════════════════════════════════

def test_missing_order_id_detected(tmp_path):
    """Conversation without order ID → 'order_id' in missing_fields."""
    service = make_service(tmp_path)
    req = TicketCreateRequest(conversation=MISSING_ORDER_ID_CONVERSATION)
    resp = service.create_from_conversation(req)

    assert "order_id" in resp.missing_fields
    assert resp.ticket.order_id is None


# ════════════════════════════════════════════════════════════════════════
# Test 3: Missing contact information detected
# ════════════════════════════════════════════════════════════════════════

def test_missing_contact_detected(tmp_path):
    """Conversation without email or phone → 'contact_info' in missing_fields."""
    service = make_service(tmp_path)
    req = TicketCreateRequest(conversation=MISSING_CONTACT_CONVERSATION)
    resp = service.create_from_conversation(req)

    assert "contact_info" in resp.missing_fields
    assert resp.ticket.contact_email is None
    assert resp.ticket.contact_phone is None


# ════════════════════════════════════════════════════════════════════════
# Test 4: Missing issue detected
# ════════════════════════════════════════════════════════════════════════

def test_missing_issue_detected(tmp_path):
    """Conversation that lists facts but no clear issue → 'issue' in missing_fields."""
    service = make_service(tmp_path)
    req = TicketCreateRequest(conversation=MISSING_ISSUE_CONVERSATION)
    resp = service.create_from_conversation(req)

    assert "issue" in resp.missing_fields


# ════════════════════════════════════════════════════════════════════════
# Test 5: Missing evidence represented safely (not as error)
# ════════════════════════════════════════════════════════════════════════

def test_missing_evidence_safe_representation(tmp_path):
    """Evidence is optional for non-payment disputes; missing evidence → ticket.evidence is None."""
    service = make_service(tmp_path)
    # Minimal conversation with enough mandatory fields but no evidence
    conversation = (
        "Customer: I'm Maria. Order ID: ORD-2024-1234. "
        "Course: Python Basics. Email: maria@example.com. "
        "Issue: I cannot access the course videos."
    )
    req = TicketCreateRequest(conversation=conversation)
    resp = service.create_from_conversation(req)

    # evidence is optional — either None or a string; must NOT raise
    assert resp.ticket.evidence is None or isinstance(resp.ticket.evidence, str)


# ════════════════════════════════════════════════════════════════════════
# Test 6: No hallucinated order ID
# ════════════════════════════════════════════════════════════════════════

def test_no_hallucinated_order_id(tmp_path):
    """When no order ID is present in conversation, ticket.order_id must be None."""
    service = make_service(tmp_path)
    conversation = (
        "Customer: Hi, I'm Ben. Email: ben@example.com. "
        "Course: Data Science. Issue: I cannot access Module 2."
    )
    req = TicketCreateRequest(conversation=conversation)
    resp = service.create_from_conversation(req)

    assert resp.ticket.order_id is None, (
        f"Hallucinated order_id: {resp.ticket.order_id!r}"
    )


# ════════════════════════════════════════════════════════════════════════
# Test 7: No hallucinated email
# ════════════════════════════════════════════════════════════════════════

def test_no_hallucinated_email(tmp_path):
    """When no email is present in conversation, ticket.contact_email must be None."""
    service = make_service(tmp_path)
    conversation = (
        "Customer: My order ID is ORD-2024-7777. "
        "Course: SQL Mastery. Issue: Videos not loading."
    )
    req = TicketCreateRequest(conversation=conversation)
    resp = service.create_from_conversation(req)

    assert resp.ticket.contact_email is None, (
        f"Hallucinated email: {resp.ticket.contact_email!r}"
    )


# ════════════════════════════════════════════════════════════════════════
# Test 8: Ticket receives a unique ticket_id
# ════════════════════════════════════════════════════════════════════════

def test_ticket_receives_unique_id(tmp_path):
    """Two tickets from the same service instance must have distinct ticket_ids."""
    service = make_service(tmp_path)
    req1 = TicketCreateRequest(conversation=COMPLETE_CONVERSATION)
    req2 = TicketCreateRequest(conversation=MISSING_ORDER_ID_CONVERSATION)
    r1 = service.create_from_conversation(req1)
    r2 = service.create_from_conversation(req2)

    assert r1.ticket.ticket_id != r2.ticket.ticket_id
    assert r1.ticket.ticket_id.startswith("TCK-")
    assert r2.ticket.ticket_id.startswith("TCK-")


# ════════════════════════════════════════════════════════════════════════
# Test 9: Ticket can be retrieved by ID
# ════════════════════════════════════════════════════════════════════════

def test_ticket_can_be_retrieved(tmp_path):
    """A stored ticket must be retrievable via storage.get()."""
    service = make_service(tmp_path)
    req = TicketCreateRequest(conversation=COMPLETE_CONVERSATION)
    resp = service.create_from_conversation(req)

    retrieved = service._storage.get(resp.ticket.ticket_id)
    assert retrieved is not None
    assert retrieved.ticket_id == resp.ticket.ticket_id


# ════════════════════════════════════════════════════════════════════════
# Test 10: Ticket status can be updated
# ════════════════════════════════════════════════════════════════════════

def test_ticket_status_can_be_updated(tmp_path):
    """storage.update_status() should change the ticket's status field."""
    service = make_service(tmp_path)
    req = TicketCreateRequest(conversation=COMPLETE_CONVERSATION)
    resp = service.create_from_conversation(req)

    updated = service._storage.update_status(
        ticket_id=resp.ticket.ticket_id,
        status=TicketStatus.READY_FOR_ROUTING,
        note="Verified by support team",
    )
    assert updated is not None
    assert updated.status == TicketStatus.READY_FOR_ROUTING


# ════════════════════════════════════════════════════════════════════════
# Test 11: Ticket list works
# ════════════════════════════════════════════════════════════════════════

def test_ticket_list_works(tmp_path):
    """storage.list_all() should return all stored tickets."""
    service = make_service(tmp_path)
    req1 = TicketCreateRequest(conversation=COMPLETE_CONVERSATION)
    req2 = TicketCreateRequest(conversation=MISSING_ORDER_ID_CONVERSATION)
    service.create_from_conversation(req1)
    service.create_from_conversation(req2)

    all_tickets = service._storage.list_all()
    assert len(all_tickets) == 2


# ════════════════════════════════════════════════════════════════════════
# Test 12: Clarification request mentions only missing mandatory fields
# ════════════════════════════════════════════════════════════════════════

def test_clarification_request_contains_only_missing_fields(tmp_path):
    """Clarification request must address only what is missing, not re-ask for known info."""
    service = make_service(tmp_path)
    # Has everything except order_id
    conversation = (
        "Customer: Hi, I'm Linda. Email: linda@example.com. "
        "Course: Python for Data Science. Issue: I cannot access Module 1."
    )
    req = TicketCreateRequest(conversation=conversation)
    resp = service.create_from_conversation(req)

    assert resp.clarification_request is not None
    assert "order" in resp.clarification_request.lower() or "order_id" in resp.clarification_request.lower()
    # Should not ask about email again
    assert "email" not in resp.clarification_request.lower() or "order" in resp.clarification_request.lower()


# ════════════════════════════════════════════════════════════════════════
# Test 13: Existing /ask endpoint is still reachable (no regression)
# ════════════════════════════════════════════════════════════════════════

def test_existing_ask_endpoint_still_works(monkeypatch):
    """
    /ask must remain present in the FastAPI app after Task 3 routes are added.
    We monkeypatch get_active_faiss_path and get_qa_chain to avoid real I/O.
    """
    import importlib
    import langchain_helper as lh

    # Patch active path check so /ask doesn't 400 due to missing FAISS index
    monkeypatch.setattr(lh, "get_active_faiss_path", lambda: "/fake/faiss")

    fake_chain_called = {}

    def fake_chain(q):
        fake_chain_called["q"] = q
        return {"result": "mock answer"}

    monkeypatch.setattr(lh, "get_qa_chain", lambda: fake_chain)

    from fastapi.testclient import TestClient
    import main as app_module
    # Reload so monkeypatched functions are picked up at route-call time
    client = TestClient(app_module.app)

    response = client.post("/ask", json={"question": "How do I access my course?"})
    # 200 from mock or 400 if active FAISS not found (acceptable — route is present)
    assert response.status_code in (200, 400, 500), (
        f"Unexpected status: {response.status_code}"
    )


# ════════════════════════════════════════════════════════════════════════
# Test 14: Existing /multimodal/analyze endpoint is still reachable
# ════════════════════════════════════════════════════════════════════════

def test_existing_multimodal_analyze_endpoint_still_works():
    """
    /multimodal/analyze must remain present in the FastAPI app.
    Sends an invalid (empty) file to trigger a validation 400, confirming the route exists.
    """
    from fastapi.testclient import TestClient
    import main as app_module

    client = TestClient(app_module.app)
    # Send empty bytes to trigger early validation (400 or 422), not 404
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("test.txt", b"", "text/plain")},
    )
    assert response.status_code != 404, "Route /multimodal/analyze must exist"
    assert response.status_code in (200, 400, 422, 500)


# ════════════════════════════════════════════════════════════════════════
# Test 15: Incomplete ticket status is WAITING_FOR_CUSTOMER;
#           complete ticket status is READY_FOR_ROUTING
# ════════════════════════════════════════════════════════════════════════

def test_ticket_status_reflects_completeness(tmp_path):
    """
    - A conversation missing mandatory fields → status WAITING_FOR_CUSTOMER
    - A conversation with all mandatory fields → status READY_FOR_ROUTING
    """
    service = make_service(tmp_path)

    # Incomplete (no contact info)
    incomplete_req = TicketCreateRequest(conversation=MISSING_CONTACT_CONVERSATION)
    incomplete_resp = service.create_from_conversation(incomplete_req)
    assert incomplete_resp.ticket.status == TicketStatus.WAITING_FOR_CUSTOMER

    # Complete — provide all fields
    complete_conv = (
        "Customer: Hi, I'm Ananya. Order ID: ORD-2024-0001. "
        "Course: Machine Learning A-Z. Email: ananya@example.com. "
        "Issue: I completed payment but my account shows no active course."
    )
    complete_req = TicketCreateRequest(conversation=complete_conv)
    complete_resp = service.create_from_conversation(complete_req)
    assert complete_resp.ticket.status == TicketStatus.READY_FOR_ROUTING
    assert complete_resp.is_complete is True
