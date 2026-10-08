"""
Task 3 Phase 3: Ticket Router, Skill Detector, and Workload Balancer.
Routes tickets deterministically based on skills, team status, agent availability,
workload balancing, and business hours.
"""

from __future__ import annotations

import re
import logging
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict, Any

from .routing_models import (
    RoutingStatus,
    RoutingResult,
    Team,
    Agent,
)
from .routing_config import RoutingConfig, get_routing_config
from .business_hours import (
    BusinessHoursConfig,
    is_business_time,
    next_business_moment,
)
from .sla_config import get_sla_config
from .models import TicketBase

logger = logging.getLogger(__name__)

# Optional PII masking for safe logging — reuse pipeline.security if available.
try:
    from pipeline.security import PIIMasker as _PIIMasker

    def _mask(text: str) -> str:
        return _PIIMasker.mask_text(text)

except Exception:  # pragma: no cover
    def _mask(text: str) -> str:  # type: ignore[misc]
        return "[MASKED]"


# ──────────────────────────────────────────────────────────────────────────────
# Skill & Team Detector
# ──────────────────────────────────────────────────────────────────────────────

class SkillDetector:
    """
    Deterministic rule-based skill and team detector for e-learning support.
    Maps customer issue text to required skills and appropriate teams.
    """

    # (skill, team_id, [keywords/phrases])
    # Evaluated in order: specific issues first
    SKILL_RULES: List[Tuple[str, str, List[str]]] = [
        (
            "refund",
            "payment_support",
            [
                "refund", "money back", "cancel enrollment and refund",
                "return payment", "cancel subscription and refund",
                "claim refund", "refund policy",
            ],
        ),
        (
            "payment",
            "payment_support",
            [
                "charged", "double charged", "charged twice", "payment",
                "invoice", "receipt", "deducted", "billing", "fee",
                "transaction failed", "payment dispute", "payment not processed",
                "bank slip", "transaction id",
            ],
        ),
        (
            "course_access",
            "course_support",
            [
                "cannot access", "can't access", "no access", "access issue",
                "course access", "login to course", "portal access",
                "course not loading", "course link", "enrolled but cannot see",
                "not received access", "access not received", "blocked from course",
                "access module", "materials not opening",
            ],
        ),
        (
            "course_content",
            "course_support",
            [
                "syllabus", "curriculum", "module", "video", "quiz",
                "assignment", "materials", "exercise", "lecture",
                "homework", "lesson", "question about topic", "course duration",
            ],
        ),
        (
            "login",
            "technical_support",
            [
                "login", "log in", "password", "sign in", "credential",
                "reset password", "auth", "otp", "cannot login", "forgot password",
            ],
        ),
        (
            "technical",
            "technical_support",
            [
                "technical", "bug", "crash", "browser", "error code",
                "platform down", "system requirement", "virtualbox", "windows",
                "install", "python environment", "code error", "jupyter",
            ],
        ),
        (
            "certificate",
            "account_support",
            [
                "certificate", "certification", "completion letter",
                "diploma", "badge", "certificate not issued", "download certificate",
            ],
        ),
        (
            "account",
            "account_support",
            [
                "account", "profile", "change email", "username",
                "name on account", "delete account", "update details",
            ],
        ),
    ]

    def detect(self, text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Analyze issue text and return (required_skill, team_id).
        Returns (None, None) if no matching rule is found.
        """
        if not text or not str(text).strip():
            return None, None

        text_lower = text.lower()
        for skill, team_id, keywords in self.SKILL_RULES:
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                    logger.debug("SkillDetector matched '%s' -> skill: %s, team: %s", kw, skill, team_id)
                    return skill, team_id

        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Ticket Router Engine
# ──────────────────────────────────────────────────────────────────────────────

class TicketRouter:
    """
    Core routing engine for support tickets.
    Evaluates:
      1. Issue skill matching
      2. Team active status
      3. Business hours (using Phase 2 configuration)
      4. Agent active & availability status
      5. Workload balancing and deterministic tie-breaking
    """

    def __init__(
        self,
        routing_config: Optional[RoutingConfig] = None,
        skill_detector: Optional[SkillDetector] = None,
        business_hours_cfg: Optional[BusinessHoursConfig] = None,
    ) -> None:
        self._routing_config = routing_config
        self.skill_detector = skill_detector or SkillDetector()
        self._business_hours_cfg = business_hours_cfg

    @property
    def config(self) -> RoutingConfig:
        return self._routing_config if self._routing_config is not None else get_routing_config()

    @property
    def business_cfg(self) -> BusinessHoursConfig:
        if self._business_hours_cfg is not None:
            return self._business_hours_cfg
        return get_sla_config().business_hours

    def route(
        self,
        ticket: TicketBase,
        now: Optional[datetime] = None,
        ignore_business_hours: bool = False,
    ) -> RoutingResult:
        """
        Route a ticket deterministically to an eligible team and agent.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # 1. Validate ticket has an issue or text to analyze
        analysis_text = f"{ticket.issue or ''} {ticket.unresolved_reason or ''}".strip()
        if not analysis_text:
            return RoutingResult(
                ticket_id=ticket.ticket_id,
                routing_status=RoutingStatus.INVALID_TICKET,
                reason="Ticket has no issue description to determine routing.",
                routed_at=now.isoformat(),
            )

        logger.info(
            "Routing ticket %s: %s",
            ticket.ticket_id,
            _mask(analysis_text[:60]),
        )

        # 2. Skill Detection
        skill, target_team_id = self.skill_detector.detect(analysis_text)
        if not skill or not target_team_id:
            logger.warning("No matching skill found for ticket %s", ticket.ticket_id)
            return RoutingResult(
                ticket_id=ticket.ticket_id,
                routing_status=RoutingStatus.NO_MATCHING_SKILL,
                reason="No matching skill or support category could be determined from the ticket issue.",
                routed_at=now.isoformat(),
            )

        # 3. Check Team Availability
        team = self.config.get_team(target_team_id)
        if not team or not team.active:
            logger.warning("Team '%s' is unavailable for ticket %s", target_team_id, ticket.ticket_id)
            return RoutingResult(
                ticket_id=ticket.ticket_id,
                routing_status=RoutingStatus.TEAM_UNAVAILABLE,
                team_id=target_team_id,
                required_skill=skill,
                reason=f"The required support team '{target_team_id}' is currently inactive or unavailable.",
                routed_at=now.isoformat(),
            )

        # 4. Business Hours Check
        if team.business_hours_supported and not ignore_business_hours:
            if not is_business_time(now, self.business_cfg):
                next_bt = next_business_moment(now, self.business_cfg)
                logger.info(
                    "Ticket %s arrived after hours for team %s. Next business window: %s",
                    ticket.ticket_id,
                    team.team_id,
                    next_bt.isoformat(),
                )
                return RoutingResult(
                    ticket_id=ticket.ticket_id,
                    routing_status=RoutingStatus.AFTER_HOURS,
                    team_id=team.team_id,
                    required_skill=skill,
                    reason="Request arrived outside configured business hours. It will be routed during the next business period.",
                    next_business_time=next_bt.isoformat(),
                    routed_at=now.isoformat(),
                )

        # 5. Agent Filtering & Eligibility
        team_agents = self.config.get_agents_by_team(team.team_id)
        eligible_agents: List[Agent] = []
        skipped_agents: List[Dict[str, str]] = []

        for agent in team_agents:
            if not agent.active:
                skipped_agents.append({"agent_id": agent.agent_id, "reason": "agent inactive"})
                continue
            if not agent.available:
                skipped_agents.append({"agent_id": agent.agent_id, "reason": "agent unavailable"})
                continue
            if skill not in agent.skills:
                skipped_agents.append({"agent_id": agent.agent_id, "reason": f"lacks required skill '{skill}'"})
                continue
            if agent.current_workload >= agent.max_workload:
                skipped_agents.append({
                    "agent_id": agent.agent_id,
                    "reason": f"at capacity ({agent.current_workload}/{agent.max_workload})",
                })
                continue

            eligible_agents.append(agent)

        if not eligible_agents:
            logger.warning(
                "No agent available for ticket %s with skill '%s' in team '%s'",
                ticket.ticket_id,
                skill,
                team.team_id,
            )
            return RoutingResult(
                ticket_id=ticket.ticket_id,
                routing_status=RoutingStatus.NO_AGENT_AVAILABLE,
                team_id=team.team_id,
                required_skill=skill,
                reason="All suitable agents with the required skill are currently unavailable or at maximum workload capacity.",
                skipped_agents=skipped_agents,
                routed_at=now.isoformat(),
            )

        # 6. Workload Balancing & Deterministic Tie-breaking
        # Preference: lowest current_workload -> lowest workload ratio -> agent_id alphabetical
        eligible_agents.sort(
            key=lambda a: (
                a.current_workload,
                (a.current_workload / a.max_workload) if a.max_workload > 0 else 1.0,
                a.agent_id,
            )
        )
        selected_agent = eligible_agents[0]

        logger.info(
            "Ticket %s routed to agent %s (%s) with workload %d/%d",
            ticket.ticket_id,
            selected_agent.agent_id,
            selected_agent.agent_name,
            selected_agent.current_workload,
            selected_agent.max_workload,
        )

        return RoutingResult(
            ticket_id=ticket.ticket_id,
            routing_status=RoutingStatus.ASSIGNED,
            team_id=team.team_id,
            agent_id=selected_agent.agent_id,
            agent_name=selected_agent.agent_name,
            required_skill=skill,
            workload=selected_agent.current_workload,
            max_workload=selected_agent.max_workload,
            routed_at=now.isoformat(),
            eligible_agents_considered=[a.agent_id for a in eligible_agents],
            skipped_agents=skipped_agents,
            reason=f"Agent has required skill '{skill}', is available, and has the lowest workload ({selected_agent.current_workload}/{selected_agent.max_workload}).",
        )
