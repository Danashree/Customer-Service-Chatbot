"""
Task 3 Phase 3: Runtime-configurable Routing Configuration.
Manages agents and teams in memory with thread safety and runtime mutability.
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import Dict, List, Optional, Any

from .routing_models import Team, Agent

logger = logging.getLogger(__name__)


def _create_default_teams() -> Dict[str, Team]:
    return {
        "course_support": Team(
            team_id="course_support",
            team_name="Course Support",
            skills=["course_access", "course_content"],
            active=True,
            business_hours_supported=True,
            description="Assists students with accessing courses and understanding course content.",
        ),
        "payment_support": Team(
            team_id="payment_support",
            team_name="Payment Support",
            skills=["payment", "refund"],
            active=True,
            business_hours_supported=True,
            description="Handles payment discrepancies, double charges, and refund requests.",
        ),
        "technical_support": Team(
            team_id="technical_support",
            team_name="Technical Support",
            skills=["technical", "login"],
            active=True,
            business_hours_supported=True,
            description="Resolves LMS platform issues, login credentials, and browser playback bugs.",
        ),
        "account_support": Team(
            team_id="account_support",
            team_name="Account Support",
            skills=["account", "certificate"],
            active=True,
            business_hours_supported=True,
            description="Assists with profile details, completion certificates, and credentials.",
        ),
    }


def _create_default_agents() -> Dict[str, Agent]:
    return {
        "AGENT-01": Agent(
            agent_id="AGENT-01",
            agent_name="Alice Chen",
            team_id="course_support",
            skills=["course_access", "course_content"],
            available=True,
            current_workload=2,
            max_workload=10,
            active=True,
        ),
        "AGENT-02": Agent(
            agent_id="AGENT-02",
            agent_name="Bob Smith",
            team_id="course_support",
            skills=["course_access"],
            available=True,
            current_workload=5,
            max_workload=10,
            active=True,
        ),
        "AGENT-03": Agent(
            agent_id="AGENT-03",
            agent_name="Carol Danvers",
            team_id="payment_support",
            skills=["payment", "refund"],
            available=True,
            current_workload=3,
            max_workload=10,
            active=True,
        ),
        "AGENT-04": Agent(
            agent_id="AGENT-04",
            agent_name="David Lee",
            team_id="payment_support",
            skills=["payment"],
            available=True,
            current_workload=7,
            max_workload=10,
            active=True,
        ),
        "AGENT-05": Agent(
            agent_id="AGENT-05",
            agent_name="Elena Rostova",
            team_id="technical_support",
            skills=["technical", "login"],
            available=True,
            current_workload=1,
            max_workload=10,
            active=True,
        ),
        "AGENT-06": Agent(
            agent_id="AGENT-06",
            agent_name="Frank Miller",
            team_id="technical_support",
            skills=["technical"],
            available=True,
            current_workload=4,
            max_workload=10,
            active=True,
        ),
        "AGENT-07": Agent(
            agent_id="AGENT-07",
            agent_name="Grace Hopper",
            team_id="account_support",
            skills=["account", "certificate"],
            available=True,
            current_workload=2,
            max_workload=10,
            active=True,
        ),
    }


class RoutingConfig:
    """
    Thread-safe registry for Teams and Agents.
    Supports dynamic runtime modifications without application restart.
    """

    def __init__(
        self,
        teams: Optional[Dict[str, Team]] = None,
        agents: Optional[Dict[str, Agent]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._teams: Dict[str, Team] = teams if teams is not None else _create_default_teams()
        self._agents: Dict[str, Agent] = agents if agents is not None else _create_default_agents()

    # ── Teams ──────────────────────────────────────────────────────────────

    def get_teams(self) -> Dict[str, Team]:
        with self._lock:
            return {k: copy.deepcopy(v) for k, v in self._teams.items()}

    def get_team(self, team_id: str) -> Optional[Team]:
        with self._lock:
            team = self._teams.get(team_id)
            return copy.deepcopy(team) if team else None

    def add_team(self, team: Team) -> None:
        with self._lock:
            self._teams[team.team_id] = copy.deepcopy(team)
            logger.info("Added team: %s", team.team_id)

    def update_team(self, team_id: str, **fields: Any) -> Optional[Team]:
        with self._lock:
            team = self._teams.get(team_id)
            if not team:
                return None
            data = team.model_dump()
            data.update({k: v for k, v in fields.items() if v is not None})
            updated = Team(**data)
            self._teams[team_id] = updated
            logger.info("Updated team %s: %s", team_id, fields)
            return copy.deepcopy(updated)

    # ── Agents ─────────────────────────────────────────────────────────────

    def get_agents(self) -> Dict[str, Agent]:
        with self._lock:
            return {k: copy.deepcopy(v) for k, v in self._agents.items()}

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        with self._lock:
            agent = self._agents.get(agent_id)
            return copy.deepcopy(agent) if agent else None

    def add_agent(self, agent: Agent) -> None:
        with self._lock:
            self._agents[agent.agent_id] = copy.deepcopy(agent)
            logger.info("Added agent: %s", agent.agent_id)

    def update_agent(self, agent_id: str, **fields: Any) -> Optional[Agent]:
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return None
            data = agent.model_dump()
            data.update({k: v for k, v in fields.items() if v is not None})
            updated = Agent(**data)
            self._agents[agent_id] = updated
            logger.info("Updated agent %s: %s", agent_id, fields)
            return copy.deepcopy(updated)

    def get_agents_by_team(self, team_id: str) -> List[Agent]:
        with self._lock:
            return [
                copy.deepcopy(a)
                for a in self._agents.values()
                if a.team_id == team_id
            ]

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset_to_defaults(self) -> None:
        with self._lock:
            self._teams = _create_default_teams()
            self._agents = _create_default_agents()
            logger.info("RoutingConfig reset to defaults.")


# ── Global Singleton ─────────────────────────────────────────────────────────

_routing_lock = threading.Lock()
_routing_config: RoutingConfig = RoutingConfig()


def get_routing_config() -> RoutingConfig:
    with _routing_lock:
        return _routing_config


def update_routing_config(
    teams: Optional[List[Team]] = None,
    agents: Optional[List[Agent]] = None,
) -> RoutingConfig:
    with _routing_lock:
        if teams:
            for t in teams:
                _routing_config.add_team(t)
        if agents:
            for a in agents:
                _routing_config.add_agent(a)
        return _routing_config


def reset_routing_config() -> RoutingConfig:
    with _routing_lock:
        _routing_config.reset_to_defaults()
        return _routing_config
