"""
Task 3 Support Ticket Service (Phase 1 + Phase 2).
Orchestrates ticket creation, validation, severity/sentiment/impact classification,
priority calculation, and SLA lifecycle management.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .models import (
    TicketBase,
    TicketCreateRequest,
    TicketResponse,
    TicketStatus,
    PriorityResponse,
    SLAResponse,
)
from .priority_enums import (
    Severity,
    Sentiment,
    CustomerImpact,
    Priority,
    SLAStatus,
    EscalationStatus,
)
from .routing_models import (
    RoutingResult,
    RoutingStatus,
)
from .routing_config import RoutingConfig, get_routing_config
from .router import TicketRouter
from .extractor import ConversationExtractor
from .validator import MandatoryFieldValidator
from .storage import TicketStorage
from .sentiment import detect_sentiment
from .severity import classify_severity, classify_customer_impact
from .priority_calculator import calculate_priority, PriorityBreakdown
from .sla_config import get_sla_config, SLAConfig
from .sla_manager import initialise_sla, check_sla, resolve_sla, SLAState
from .phase4_models import (
    DuplicateCheckResponse,
    RelationshipResponse,
    IssueGroup,
    HandoffSummary,
)
from .duplicate_detector import DuplicateDetector
from .issue_grouper import IssueGrouper
from .handoff import HandoffGenerator

logger = logging.getLogger(__name__)

# Optional PII masking for safe logging — reuse pipeline.security if available.
try:
    from pipeline.security import PIIMasker as _PIIMasker

    def _mask(text: str) -> str:
        return _PIIMasker.mask_text(text)

except Exception:  # pragma: no cover — graceful degradation
    def _mask(text: str) -> str:  # type: ignore[misc]
        return "[CONVERSATION MASKED]"


class TicketService:
    """
    High-level orchestrator for Task 3 Support Ticket lifecycle.

    Phase 1:
      - create_from_conversation(request)
    Phase 2:
      - calculate_and_update_priority(ticket_id, now)
      - get_priority_breakdown(ticket_id, now)
      - get_sla_info(ticket_id, now)
      - check_and_update_sla(ticket_id, now)
    Phase 3:
      - route_ticket(ticket_id, now, ignore_business_hours)
      - get_ticket_routing(ticket_id)
    Phase 4:
      - check_ticket_duplicate(ticket_id)
      - get_ticket_relationships(ticket_id)
      - group_ticket(ticket_id, group_id, group_topic)
      - get_issue_group(group_id)
      - list_issue_groups()
      - remove_ticket_from_group(ticket_id, group_id)
      - generate_ticket_handoff(ticket_id)
    """

    def __init__(
        self,
        storage: Optional[TicketStorage] = None,
        extractor: Optional[ConversationExtractor] = None,
        validator: Optional[MandatoryFieldValidator] = None,
        sla_config: Optional[SLAConfig] = None,
        routing_config: Optional[RoutingConfig] = None,
        router: Optional[TicketRouter] = None,
        duplicate_detector: Optional[DuplicateDetector] = None,
        issue_grouper: Optional[IssueGrouper] = None,
        handoff_generator: Optional[HandoffGenerator] = None,
    ) -> None:
        self._storage = storage or TicketStorage()
        self._extractor = extractor or ConversationExtractor()
        self._validator = validator or MandatoryFieldValidator()
        self._sla_config = sla_config
        self._routing_config = routing_config
        self._router = router or TicketRouter(
            routing_config=routing_config,
            business_hours_cfg=self._get_sla_cfg().business_hours,
        )
        self._duplicate_detector = duplicate_detector or DuplicateDetector()
        self._issue_grouper = issue_grouper or IssueGrouper(duplicate_detector=self._duplicate_detector)
        self._handoff_generator = handoff_generator or HandoffGenerator()

    def _get_sla_cfg(self) -> SLAConfig:
        return self._sla_config if self._sla_config is not None else get_sla_config()

    # ─── create ──────────────────────────────────────────────────────────────

    def create_from_conversation(
        self,
        request: TicketCreateRequest,
        now: Optional[datetime] = None,
    ) -> TicketResponse:
        """
        Convert an unresolved conversation into a structured support ticket.

        Steps:
        1. Extract structured fields from conversation text (regex + optional LLM).
        2. Build TicketBase from extracted fields + request metadata.
        3. Validate mandatory fields; mark missing ones.
        4. Set ticket status: READY_FOR_ROUTING if complete, else WAITING_FOR_CUSTOMER.
        5. Phase 2: Classify sentiment, severity, customer impact.
        6. Phase 2: Compute initial priority and initialise SLA.
        7. Persist and return TicketResponse.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Log safely — never log raw conversation text
        logger.info(
            "Creating ticket from conversation (id=%s): %s",
            request.conversation_id or "n/a",
            _mask(request.conversation[:80]),
        )

        # Step 1 — Extract
        extracted = self._extractor.extract(request.conversation)

        # Step 2 — Build TicketBase
        ticket = TicketBase(
            conversation_id=request.conversation_id,
            customer_id=request.customer_id,
            unresolved_reason=request.unresolved_reason,
            created_at=now.isoformat(),
            # Extracted fields
            customer_name=extracted.get("customer_name"),
            contact_email=extracted.get("contact_email"),
            contact_phone=extracted.get("contact_phone"),
            course_name=extracted.get("course_name"),
            order_id=extracted.get("order_id"),
            issue=extracted.get("issue"),
            evidence=extracted.get("evidence"),
        )

        # Step 3 — Validate
        validation = self._validator.validate(ticket)

        # Step 4 — Set status from validation result
        ticket.missing_fields = validation.missing_fields
        if validation.is_complete:
            ticket.status = TicketStatus.READY_FOR_ROUTING
        else:
            ticket.status = TicketStatus.WAITING_FOR_CUSTOMER

        # Step 5 — Phase 2: Sentiment, Severity, Customer Impact
        sentiment = detect_sentiment(request.conversation)
        severity = classify_severity(ticket.issue, request.conversation)
        customer_impact = classify_customer_impact(ticket.issue, request.conversation)

        ticket.sentiment = sentiment.value
        ticket.severity = severity.value
        ticket.customer_impact = customer_impact.value

        # Step 6 — Phase 2: Priority & SLA Initialisation
        sla_cfg = self._get_sla_cfg()
        priority, priority_score, breakdown = calculate_priority(
            severity=severity,
            sentiment=sentiment,
            customer_impact=customer_impact,
            sla_status=SLAStatus.IN_PROGRESS,
            sla_hours=None,
            sla_started_at=None,
            now=now,
            business_hours_cfg=sla_cfg.business_hours,
        )
        ticket.priority = priority.value
        ticket.priority_score = priority_score

        sla_hours = sla_cfg.get_sla_hours(priority.value)
        sla_info = initialise_sla(
            sla_hours=sla_hours,
            business_cfg=sla_cfg.business_hours,
            warning_threshold=sla_cfg.warning_threshold,
            now=now,
        )
        ticket.sla_hours = sla_info["sla_hours"]
        ticket.sla_started_at = sla_info["sla_started_at"]
        ticket.sla_warning_at = sla_info["sla_warning_at"]
        ticket.sla_due_at = sla_info["sla_due_at"]
        ticket.sla_status = sla_info["sla_status"].value
        ticket.escalation_status = sla_info["escalation_status"].value
        ticket.escalated_at = sla_info["escalated_at"]
        ticket.escalation_reason = sla_info["escalation_reason"]
        ticket.waiting_time = 0.0

        # Step 7 — Persist
        self._storage.create(ticket)
        logger.info(
            "Ticket %s stored with status %s, priority=%s (missing=%s)",
            ticket.ticket_id,
            ticket.status,
            ticket.priority,
            ticket.missing_fields,
        )

        return TicketResponse(
            ticket=ticket,
            is_complete=validation.is_complete,
            missing_fields=validation.missing_fields,
            clarification_request=validation.clarification_request,
        )

    # ─── Priority operations ──────────────────────────────────────────────────

    def calculate_and_update_priority(
        self,
        ticket_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[PriorityResponse]:
        """
        Calculate and update the priority and breakdown for an existing ticket.
        Dynamically accounts for current SLA consumption, waiting time, and urgency.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        ticket = self._storage.get(ticket_id)
        if not ticket:
            return None

        sla_cfg = self._get_sla_cfg()

        # Resolve classification enums safely
        sentiment = Sentiment(ticket.sentiment) if ticket.sentiment else Sentiment.NEUTRAL
        severity = Severity(ticket.severity) if ticket.severity else Severity.LOW
        customer_impact = CustomerImpact(ticket.customer_impact) if ticket.customer_impact else CustomerImpact.LOW

        # Evaluate SLA state first
        sla_state = check_sla(
            ticket_data=ticket.model_dump(),
            business_cfg=sla_cfg.business_hours,
            warning_threshold=sla_cfg.warning_threshold,
            now=now,
        )

        # Recalculate priority with current waiting time & SLA state
        priority, priority_score, breakdown = calculate_priority(
            severity=severity,
            sentiment=sentiment,
            customer_impact=customer_impact,
            sla_status=sla_state.sla_status,
            sla_hours=ticket.sla_hours,
            sla_started_at=ticket.sla_started_at,
            now=now,
            business_hours_cfg=sla_cfg.business_hours,
        )

        # Update ticket in storage
        ticket.priority = priority.value
        ticket.priority_score = priority_score
        ticket.sla_status = sla_state.sla_status.value
        ticket.escalation_status = sla_state.escalation_status.value
        ticket.escalated_at = sla_state.escalated_at
        ticket.escalation_reason = sla_state.escalation_reason
        ticket.waiting_time = float(sla_state.business_minutes_consumed)

        self._storage.update(ticket)

        return PriorityResponse(
            ticket_id=ticket.ticket_id,
            priority=priority.value,
            priority_score=priority_score,
            severity=severity.value,
            sentiment=sentiment.value,
            customer_impact=customer_impact.value,
            sla_status=sla_state.sla_status.value,
            breakdown=breakdown.as_dict(),
        )

    def get_priority_breakdown(
        self,
        ticket_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[PriorityResponse]:
        """
        Return the explainable priority breakdown for a ticket.
        If priority is not yet calculated, calculates it.
        """
        ticket = self._storage.get(ticket_id)
        if not ticket:
            return None

        if ticket.priority is None or ticket.priority_score is None:
            return self.calculate_and_update_priority(ticket_id, now)

        return self.calculate_and_update_priority(ticket_id, now)

    # ─── SLA operations ───────────────────────────────────────────────────────

    def get_sla_info(
        self,
        ticket_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[SLAResponse]:
        """
        Return the current SLA status, consumed/remaining minutes, and escalation info.
        """
        return self.check_and_update_sla(ticket_id, now)

    def check_and_update_sla(
        self,
        ticket_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[SLAResponse]:
        """
        Evaluate warning, breach, and escalation states for a ticket.
        Persists status transitions (e.g. IN_PROGRESS -> WARNING -> BREACHED).
        """
        if now is None:
            now = datetime.now(timezone.utc)

        ticket = self._storage.get(ticket_id)
        if not ticket:
            return None

        sla_cfg = self._get_sla_cfg()

        sla_state = check_sla(
            ticket_data=ticket.model_dump(),
            business_cfg=sla_cfg.business_hours,
            warning_threshold=sla_cfg.warning_threshold,
            now=now,
        )

        # Persist updated status
        ticket.sla_status = sla_state.sla_status.value
        ticket.escalation_status = sla_state.escalation_status.value
        ticket.escalated_at = sla_state.escalated_at
        ticket.escalation_reason = sla_state.escalation_reason
        ticket.waiting_time = float(sla_state.business_minutes_consumed)
        self._storage.update(ticket)

        return SLAResponse(
            ticket_id=ticket.ticket_id,
            sla_status=sla_state.sla_status.value,
            sla_hours=sla_state.sla_hours,
            sla_started_at=sla_state.sla_started_at or None,
            sla_warning_at=sla_state.sla_warning_at or None,
            sla_due_at=sla_state.sla_due_at or None,
            business_minutes_consumed=sla_state.business_minutes_consumed,
            business_minutes_remaining=sla_state.business_minutes_remaining,
            percentage_consumed=sla_state.percentage_consumed,
            escalation_status=sla_state.escalation_status.value,
            escalated_at=sla_state.escalated_at,
            escalation_reason=sla_state.escalation_reason,
        )

    # ─── Routing operations (Phase 3) ─────────────────────────────────────────

    def route_ticket(
        self,
        ticket_id: str,
        now: Optional[datetime] = None,
        ignore_business_hours: bool = False,
        update_agent_workload: bool = True,
    ) -> Optional[RoutingResult]:
        """
        Route an existing ticket to an eligible team and agent based on skill,
        availability, workload, and business hours.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        ticket = self._storage.get(ticket_id)
        if not ticket:
            return None

        result = self._router.route(
            ticket,
            now=now,
            ignore_business_hours=ignore_business_hours,
        )

        ticket.routing_status = result.routing_status.value
        ticket.routed_at = result.routed_at
        ticket.required_skill = result.required_skill
        ticket.routing_reason = result.reason

        if result.routing_status == RoutingStatus.ASSIGNED:
            ticket.assigned_team_id = result.team_id
            ticket.assigned_agent_id = result.agent_id
            ticket.assigned_agent_name = result.agent_name
            if update_agent_workload and result.agent_id:
                agent = self._router.config.get_agent(result.agent_id)
                if agent:
                    self._router.config.update_agent(
                        result.agent_id,
                        current_workload=agent.current_workload + 1,
                    )
        else:
            # If not assigned, reset any previous assignment fields
            ticket.assigned_team_id = result.team_id
            ticket.assigned_agent_id = None
            ticket.assigned_agent_name = None

        self._storage.update(ticket)
        return result

    def get_ticket_routing(self, ticket_id: str) -> Optional[RoutingResult]:
        """
        Return the stored routing/assignment result for a ticket.
        """
        ticket = self._storage.get(ticket_id)
        if not ticket:
            return None

        if not ticket.routing_status:
            # If not routed yet, trigger routing
            return self.route_ticket(ticket_id)

        status_enum = RoutingStatus(ticket.routing_status)
        agent = (
            self._router.config.get_agent(ticket.assigned_agent_id)
            if ticket.assigned_agent_id
            else None
        )

        return RoutingResult(
            ticket_id=ticket.ticket_id,
            routing_status=status_enum,
            team_id=ticket.assigned_team_id,
            agent_id=ticket.assigned_agent_id,
            agent_name=ticket.assigned_agent_name,
            required_skill=ticket.required_skill,
            reason=ticket.routing_reason or "Routing information retrieved.",
            workload=agent.current_workload if agent else None,
            max_workload=agent.max_workload if agent else None,
            routed_at=ticket.routed_at,
        )

    # ─── Phase 4: Duplicate, Grouping & Handoff ───────────────────────────────

    def check_ticket_duplicate(self, ticket_id: str) -> Optional[DuplicateCheckResponse]:
        """
        Compare ticket against all existing tickets to detect duplicates.
        """
        ticket = self._storage.get(ticket_id)
        if not ticket:
            return None
        existing = self._storage.list_all()
        return self._duplicate_detector.check_ticket(ticket, existing)

    def get_ticket_relationships(self, ticket_id: str) -> Optional[RelationshipResponse]:
        """
        Identify related, duplicate, and unrelated relationships for a ticket.
        """
        ticket = self._storage.get(ticket_id)
        if not ticket:
            return None
        existing = self._storage.list_all()
        relationships = self._issue_grouper.find_relationships(ticket, existing)
        return RelationshipResponse(ticket_id=ticket_id, relationships=relationships)

    def group_ticket(
        self,
        ticket_id: str,
        group_id: Optional[str] = None,
        group_topic: Optional[str] = None,
    ) -> Optional[IssueGroup]:
        """
        Add a ticket to an existing issue group, or create a new issue group.
        """
        ticket = self._storage.get(ticket_id)
        if not ticket:
            return None

        if group_id:
            existing_group = self._issue_grouper.get_group(group_id)
            if existing_group:
                return self._issue_grouper.add_ticket_to_group(group_id, ticket_id)
            # Group ID supplied does not exist: create it with that ID
            return self._issue_grouper.create_group([ticket], group_id=group_id, topic=group_topic)

        # No group_id provided: create a new group
        return self._issue_grouper.create_group([ticket], topic=group_topic)

    def get_issue_group(self, group_id: str) -> Optional[IssueGroup]:
        """
        Retrieve issue group details by ID.
        """
        return self._issue_grouper.get_group(group_id)

    def list_issue_groups(self) -> List[IssueGroup]:
        """
        List all active issue groups.
        """
        return self._issue_grouper.list_groups()

    def remove_ticket_from_group(self, ticket_id: str, group_id: str) -> Optional[IssueGroup]:
        """
        Remove a ticket from an issue group.
        """
        return self._issue_grouper.remove_ticket_from_group(group_id, ticket_id)

    def generate_ticket_handoff(self, ticket_id: str) -> Optional[HandoffSummary]:
        """
        Generate a masked operational handoff summary for transferring a ticket.
        """
        ticket = self._storage.get(ticket_id)
        if not ticket:
            return None

        # Check group membership
        groups = self._issue_grouper.storage.find_groups_for_ticket(ticket_id)
        group_id = groups[0].group_id if groups else None

        # Check duplicate status
        dup_check = self.check_ticket_duplicate(ticket_id)
        dup_status = (
            dup_check.duplicate_status.value
            if dup_check
            else "NOT_DUPLICATE"
        )

        return self._handoff_generator.generate(
            ticket,
            group_id=group_id,
            duplicate_status=dup_status,
        )
