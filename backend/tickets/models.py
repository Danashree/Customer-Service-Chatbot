"""
Task 3 Phase 1 + Phase 2: Support Ticket Data Models and Enums.

Phase 1 models remain unchanged.
Phase 2 adds Optional priority/SLA/escalation fields to TicketBase —
all default to None so existing Phase 1 tickets deserialize without error.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 enums
# ──────────────────────────────────────────────────────────────────────────────

class TicketStatus(str, Enum):
    OPEN = "OPEN"
    WAITING_FOR_CUSTOMER = "WAITING_FOR_CUSTOMER"
    READY_FOR_ROUTING = "READY_FOR_ROUTING"


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 + Phase 2 ticket base
# ──────────────────────────────────────────────────────────────────────────────

class TicketBase(BaseModel):
    # ── Phase 1 core fields ─────────────────────────────────────────────────
    ticket_id: str = Field(default_factory=lambda: f"TCK-{uuid.uuid4().hex[:8].upper()}")
    conversation_id: Optional[str] = None
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    course_name: Optional[str] = None
    order_id: Optional[str] = None
    issue: Optional[str] = None
    evidence: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: TicketStatus = TicketStatus.OPEN
    missing_fields: List[str] = Field(default_factory=list)
    unresolved_reason: Optional[str] = None

    # ── Phase 2: severity / sentiment / impact ───────────────────────────────
    severity: Optional[str] = None          # Severity enum value
    sentiment: Optional[str] = None         # Sentiment enum value
    customer_impact: Optional[str] = None   # CustomerImpact enum value
    waiting_time: Optional[float] = None    # current business minutes consumed (dynamic)

    # ── Phase 2: priority ────────────────────────────────────────────────────
    priority: Optional[str] = None          # Priority enum value
    priority_score: Optional[int] = None

    # ── Phase 2: SLA ─────────────────────────────────────────────────────────
    sla_hours: Optional[float] = None
    sla_started_at: Optional[str] = None    # ISO datetime string
    sla_due_at: Optional[str] = None        # ISO datetime string
    sla_warning_at: Optional[str] = None    # ISO datetime string (75% threshold)
    sla_status: Optional[str] = None        # SLAStatus enum value

    # ── Phase 2: escalation ──────────────────────────────────────────────────
    escalation_status: Optional[str] = None  # EscalationStatus enum value
    escalated_at: Optional[str] = None       # ISO datetime string
    escalation_reason: Optional[str] = None

    # ── Phase 3: routing ─────────────────────────────────────────────────────
    assigned_team_id: Optional[str] = None
    assigned_agent_id: Optional[str] = None
    assigned_agent_name: Optional[str] = None
    routing_status: Optional[str] = None     # RoutingStatus enum value
    routed_at: Optional[str] = None          # ISO datetime string
    required_skill: Optional[str] = None
    routing_reason: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 request / response models (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

class TicketCreateRequest(BaseModel):
    conversation: str
    conversation_id: Optional[str] = None
    customer_id: Optional[str] = None
    unresolved_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TicketStatusUpdateRequest(BaseModel):
    status: TicketStatus
    note: Optional[str] = None


class TicketValidationResult(BaseModel):
    is_complete: bool
    missing_fields: List[str]
    clarification_request: Optional[str] = None


class TicketResponse(BaseModel):
    ticket: TicketBase
    is_complete: bool
    missing_fields: List[str]
    clarification_request: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 request / response models
# ──────────────────────────────────────────────────────────────────────────────

class PriorityResponse(BaseModel):
    ticket_id: str
    priority: str
    priority_score: int
    severity: str
    sentiment: str
    customer_impact: str
    sla_status: str
    breakdown: Dict[str, int]


class SLAResponse(BaseModel):
    ticket_id: str
    sla_status: str
    sla_hours: Optional[float]
    sla_started_at: Optional[str]
    sla_warning_at: Optional[str]
    sla_due_at: Optional[str]
    business_minutes_consumed: int
    business_minutes_remaining: int
    percentage_consumed: float
    escalation_status: str
    escalated_at: Optional[str] = None
    escalation_reason: Optional[str] = None


class SLAConfigUpdateRequest(BaseModel):
    sla_hours_by_priority: Optional[Dict[str, float]] = None
    warning_threshold: Optional[float] = None
    work_start: Optional[int] = None
    work_end: Optional[int] = None
    working_weekdays: Optional[List[int]] = None
    holidays: Optional[List[str]] = None
