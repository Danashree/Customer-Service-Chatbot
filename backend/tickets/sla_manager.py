"""
Task 3 Phase 2: SLA Lifecycle Manager.

Manages the full SLA lifecycle for a ticket:
  1. initialise_sla()  — compute started_at, warning_at, due_at
  2. check_sla()       — evaluate current state (IN_PROGRESS / WARNING / BREACHED)
  3. resolve_sla()     — mark as RESOLVED

SLA breaches trigger automatic escalation:
  escalation_status = ESCALATED
  escalated_at      = first-breach timestamp
  escalation_reason = "SLA breach: <priority> SLA of <N> hours exceeded."

All calculations use business-time utilities from business_hours.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .priority_enums import EscalationStatus, SLAStatus
from .business_hours import (
    BusinessHoursConfig,
    calculate_business_minutes,
    calculate_sla_due_at,
    calculate_sla_warning_at,
)

logger = logging.getLogger(__name__)


@dataclass
class SLAState:
    """
    Snapshot of a ticket's SLA status at a point in time.
    Used both for storage updates and API responses.
    """
    sla_status: SLAStatus
    sla_hours: float
    sla_started_at: str               # ISO format
    sla_warning_at: str               # ISO format — threshold timestamp
    sla_due_at: str                   # ISO format — deadline
    business_minutes_consumed: int
    business_minutes_remaining: int
    percentage_consumed: float
    escalation_status: EscalationStatus
    escalated_at: Optional[str] = None
    escalation_reason: Optional[str] = None


def initialise_sla(
    sla_hours: float,
    business_cfg: BusinessHoursConfig,
    warning_threshold: float = 0.75,
    now: Optional[datetime] = None,
) -> dict:
    """
    Compute the SLA timestamps for a newly created ticket.

    Returns a dict of ticket field updates:
        sla_hours, sla_started_at, sla_warning_at, sla_due_at,
        sla_status (= IN_PROGRESS), escalation_status (= NOT_ESCALATED)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    due_at = calculate_sla_due_at(now, sla_hours, business_cfg)
    warning_at = calculate_sla_warning_at(now, sla_hours, business_cfg, warning_threshold)

    return {
        "sla_hours": sla_hours,
        "sla_started_at": now.isoformat(),
        "sla_warning_at": warning_at.isoformat(),
        "sla_due_at": due_at.isoformat(),
        "sla_status": SLAStatus.IN_PROGRESS,
        "escalation_status": EscalationStatus.NOT_ESCALATED,
        "escalated_at": None,
        "escalation_reason": None,
    }


def check_sla(
    ticket_data: dict,
    business_cfg: BusinessHoursConfig,
    warning_threshold: float = 0.75,
    now: Optional[datetime] = None,
) -> SLAState:
    """
    Evaluate the current SLA state for a ticket.

    Reads: sla_started_at, sla_warning_at, sla_due_at, sla_hours,
           escalation_status, escalated_at, escalation_reason.

    Returns an SLAState snapshot reflecting the new state.

    Breach logic:
      - if consumed >= total → BREACHED + ESCALATED (escalated_at set once)
      - elif now >= sla_warning_at or pct >= warning_threshold * 100 → WARNING
      - else                        → IN_PROGRESS

    Already-RESOLVED tickets remain RESOLVED.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    sla_status_raw = ticket_data.get("sla_status", SLAStatus.NOT_STARTED.value)
    current_status = SLAStatus(sla_status_raw) if isinstance(sla_status_raw, str) else sla_status_raw

    sla_hours: float = float(ticket_data.get("sla_hours") or 0)
    sla_started_at_str: Optional[str] = ticket_data.get("sla_started_at")
    sla_warning_at_str: Optional[str] = ticket_data.get("sla_warning_at")
    sla_due_at_str: Optional[str] = ticket_data.get("sla_due_at")

    escalation_status_raw = ticket_data.get("escalation_status", EscalationStatus.NOT_ESCALATED.value)
    escalation_status = (
        EscalationStatus(escalation_status_raw)
        if isinstance(escalation_status_raw, str)
        else escalation_status_raw
    )
    escalated_at: Optional[str] = ticket_data.get("escalated_at")
    escalation_reason: Optional[str] = ticket_data.get("escalation_reason")

    # Short-circuit if already resolved or not started
    if current_status == SLAStatus.RESOLVED:
        return _build_state(
            sla_status=SLAStatus.RESOLVED,
            sla_hours=sla_hours,
            sla_started_at=sla_started_at_str or now.isoformat(),
            sla_warning_at=sla_warning_at_str or now.isoformat(),
            sla_due_at=sla_due_at_str or now.isoformat(),
            consumed=0,
            total_mins=int(sla_hours * 60),
            escalation_status=escalation_status,
            escalated_at=escalated_at,
            escalation_reason=escalation_reason,
        )

    if current_status == SLAStatus.NOT_STARTED or not sla_started_at_str:
        return SLAState(
            sla_status=SLAStatus.NOT_STARTED,
            sla_hours=sla_hours,
            sla_started_at=sla_started_at_str or "",
            sla_warning_at=sla_warning_at_str or "",
            sla_due_at=sla_due_at_str or "",
            business_minutes_consumed=0,
            business_minutes_remaining=int(sla_hours * 60),
            percentage_consumed=0.0,
            escalation_status=escalation_status,
            escalated_at=escalated_at,
            escalation_reason=escalation_reason,
        )

    # Parse timestamps
    started = _parse_iso(sla_started_at_str)
    due = _parse_iso(sla_due_at_str) if sla_due_at_str else None
    warning = _parse_iso(sla_warning_at_str) if sla_warning_at_str else None

    # Normalise timezone
    started, now = _align_tz(started, now)

    consumed_mins = calculate_business_minutes(started, now, business_cfg)
    total_mins = int(sla_hours * 60)
    pct = (consumed_mins / total_mins * 100.0) if total_mins > 0 else 0.0

    # Determine new status
    if consumed_mins >= total_mins:
        new_status = SLAStatus.BREACHED
        # Trigger escalation once
        if escalation_status != EscalationStatus.ESCALATED:
            escalation_status = EscalationStatus.ESCALATED
            escalated_at = now.isoformat()
            escalation_reason = (
                f"SLA breach: {sla_hours:.0f}-business-hour SLA exceeded."
            )
    elif pct >= (warning_threshold * 100.0) or (warning and now >= warning):
        new_status = SLAStatus.WARNING
    else:
        new_status = SLAStatus.IN_PROGRESS

    return _build_state(
        sla_status=new_status,
        sla_hours=sla_hours,
        sla_started_at=sla_started_at_str,
        sla_warning_at=sla_warning_at_str or "",
        sla_due_at=sla_due_at_str or "",
        consumed=consumed_mins,
        total_mins=total_mins,
        escalation_status=escalation_status,
        escalated_at=escalated_at,
        escalation_reason=escalation_reason,
    )


def resolve_sla(ticket_data: dict) -> dict:
    """Mark the SLA as resolved. Returns field-update dict."""
    return {"sla_status": SLAStatus.RESOLVED}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _align_tz(a: datetime, b: datetime) -> tuple[datetime, datetime]:
    """Ensure both datetimes share the same timezone (UTC)."""
    if a.tzinfo is None:
        a = a.replace(tzinfo=timezone.utc)
    if b.tzinfo is None:
        b = b.replace(tzinfo=timezone.utc)
    return a, b


def _build_state(
    *,
    sla_status: SLAStatus,
    sla_hours: float,
    sla_started_at: str,
    sla_warning_at: str,
    sla_due_at: str,
    consumed: int,
    total_mins: int,
    escalation_status: EscalationStatus,
    escalated_at: Optional[str],
    escalation_reason: Optional[str],
) -> SLAState:
    pct = (consumed / total_mins * 100) if total_mins > 0 else 0.0
    remaining = max(0, total_mins - consumed)
    return SLAState(
        sla_status=sla_status,
        sla_hours=sla_hours,
        sla_started_at=sla_started_at,
        sla_warning_at=sla_warning_at,
        sla_due_at=sla_due_at,
        business_minutes_consumed=consumed,
        business_minutes_remaining=remaining,
        percentage_consumed=round(pct, 2),
        escalation_status=escalation_status,
        escalated_at=escalated_at,
        escalation_reason=escalation_reason,
    )
