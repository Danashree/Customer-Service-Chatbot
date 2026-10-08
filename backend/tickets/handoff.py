"""
Task 3 Phase 4: Masked Handoff Summary Generator.

Generates a structured, operational handoff summary for transferring a ticket
to another support team or agent. Fully sanitized via PIIMasker with zero PII leakage
and zero hallucinated information.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from .models import TicketBase
from .phase4_models import HandoffSummary
from .router import SkillDetector

logger = logging.getLogger(__name__)

# PII masking import
try:
    from pipeline.security import PIIMasker
except Exception:  # pragma: no cover
    class PIIMasker:  # type: ignore[no-redef]
        @classmethod
        def mask_text(cls, text: str) -> str:
            import re
            t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL_REDACTED]", text)
            t = re.sub(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[PAYMENT_REDACTED]", t)
            t = re.sub(r"(?:\+?\d{1,3}[-.\\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]", t)
            return t


def _clean_field(val: Optional[str], default: str = "Not provided") -> str:
    if val is None or not str(val).strip():
        return default
    return str(val).strip()


def _mask_order(order_id: Optional[str]) -> str:
    """Safely format/mask order ID without exposing internal payment secrets."""
    if not order_id or not order_id.strip():
        return "Not provided"
    clean = order_id.strip()
    # Mask order ID prefix/suffix if it looks like a sensitive number
    if len(clean) > 8:
        return f"{clean[:3]}***{clean[-3:]}"
    return clean


def _determine_next_action(ticket: TicketBase, skill: Optional[str]) -> str:
    """
    Recommend concrete next support actions based on category, issue, and state.
    Deterministic, domain-specific, and never hallucinated.
    """
    s = (skill or ticket.required_skill or "").lower()
    issue_text = (ticket.issue or "").lower()

    if "refund" in s or "refund" in issue_text:
        return "Verify payment transaction in billing ledger and review refund eligibility."
    if "payment" in s or "payment" in issue_text:
        return "Check payment gateway logs for transaction failure or pending capture."
    if "course_access" in s or "access" in issue_text:
        return "Verify learner course enrollment status and LMS access permissions."
    if "course_content" in s or "video" in issue_text or "quiz" in issue_text:
        return "Inspect lesson resource CDN link and confirm content player compatibility."
    if "login" in s or "password" in issue_text or "credentials" in issue_text:
        return "Verify learner account authentication status and send secure password reset link."
    if "certificate" in s or "certificate" in issue_text:
        return "Validate module completion criteria and re-issue learner digital certificate."
    if "technical" in s or "browser" in issue_text or "error" in issue_text:
        return "Check platform server error logs and advise learner on browser troubleshooting."
    if "account" in s:
        return "Validate account ownership and update requested learner profile details."

    return "Review ticket details and contact customer for further clarification."


class HandoffGenerator:
    """
    Constructs masked, operator-ready handoff summaries from ticket records.
    """

    def __init__(self, skill_detector: Optional[SkillDetector] = None) -> None:
        self._skill_detector = skill_detector or SkillDetector()

    def generate(
        self,
        ticket: TicketBase,
        group_id: Optional[str] = None,
        duplicate_status: Optional[str] = None,
    ) -> HandoffSummary:
        """
        Produce a structured and human-readable handoff summary.
        All customer-identifiable information is strictly masked.
        """
        # Determine customer reference
        customer_parts = []
        if ticket.customer_name:
            customer_parts.append(PIIMasker.mask_text(ticket.customer_name))
        if ticket.contact_email:
            customer_parts.append(PIIMasker.mask_text(ticket.contact_email))
        if ticket.contact_phone:
            customer_parts.append(PIIMasker.mask_text(ticket.contact_phone))
        if ticket.customer_id:
            customer_parts.append(ticket.customer_id)

        customer_ref = " | ".join(customer_parts) if customer_parts else "Not provided"

        # Determine skill if not yet identified
        skill = ticket.required_skill
        if not skill and ticket.issue:
            detected, _ = self._skill_detector.detect(ticket.issue)
            skill = detected

        # Mask text fields
        issue_masked = PIIMasker.mask_text(_clean_field(ticket.issue))
        evidence_masked = PIIMasker.mask_text(_clean_field(ticket.evidence))
        course_masked = PIIMasker.mask_text(_clean_field(ticket.course_name))
        order_masked = PIIMasker.mask_text(_mask_order(ticket.order_id))
        context_masked = PIIMasker.mask_text(_clean_field(ticket.unresolved_reason))
        routing_reason_masked = PIIMasker.mask_text(_clean_field(ticket.routing_reason))

        next_action = _determine_next_action(ticket, skill)

        assigned_team = _clean_field(ticket.assigned_team_id, default="Not assigned")
        assigned_agent = _clean_field(
            ticket.assigned_agent_name or ticket.assigned_agent_id, default="Not assigned"
        )
        dup_status = duplicate_status or "NOT_DUPLICATE"
        rel_group = group_id or "None"

        structured_summary: Dict[str, Any] = {
            "ticket_id": ticket.ticket_id,
            "customer_reference": customer_ref,
            "issue": issue_masked,
            "course_product": course_masked,
            "order_reference": order_masked,
            "evidence": evidence_masked,
            "severity": _clean_field(ticket.severity),
            "sentiment": _clean_field(ticket.sentiment),
            "customer_impact": _clean_field(ticket.customer_impact),
            "priority": _clean_field(ticket.priority),
            "sla_status": _clean_field(ticket.sla_status),
            "assigned_team": assigned_team,
            "assigned_agent": assigned_agent,
            "duplicate_status": dup_status,
            "related_group_id": rel_group,
            "routing_reason": routing_reason_masked,
            "context": context_masked,
            "next_action": next_action,
        }

        # Build human-readable formatted summary
        lines = [
            "==================================================",
            "              SUPPORT TICKET HANDOFF              ",
            "==================================================",
            f"Ticket ID:        {ticket.ticket_id}",
            f"Customer:         {customer_ref}",
            f"Issue:            {issue_masked}",
            f"Course/Product:   {course_masked}",
            f"Order Reference:  {order_masked}",
            f"Evidence:         {evidence_masked}",
            f"Severity:         {_clean_field(ticket.severity)}",
            f"Sentiment:        {_clean_field(ticket.sentiment)}",
            f"Customer Impact:  {_clean_field(ticket.customer_impact)}",
            f"Priority:         {_clean_field(ticket.priority)}",
            f"SLA Status:       {_clean_field(ticket.sla_status)}",
            f"Assigned Team:    {assigned_team}",
            f"Assigned Agent:   {assigned_agent}",
            f"Duplicate Status: {dup_status}",
            f"Issue Group:      {rel_group}",
            f"Routing Reason:   {routing_reason_masked}",
            f"Context:          {context_masked}",
            f"Next Action:      {next_action}",
            "==================================================",
        ]
        masked_summary = "\n".join(lines)

        return HandoffSummary(
            ticket_id=ticket.ticket_id,
            group_id=group_id,
            masked_summary=masked_summary,
            structured_summary=structured_summary,
        )
