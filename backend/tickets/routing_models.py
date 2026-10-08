"""
Task 3 Phase 3: Agent, Team, and Routing Models and Enums.
Defines schemas for teams, agents, routing results, and runtime configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class RoutingStatus(str, Enum):
    ASSIGNED = "ASSIGNED"
    NO_AGENT_AVAILABLE = "NO_AGENT_AVAILABLE"
    TEAM_UNAVAILABLE = "TEAM_UNAVAILABLE"
    AFTER_HOURS = "AFTER_HOURS"
    NO_MATCHING_SKILL = "NO_MATCHING_SKILL"
    INVALID_TICKET = "INVALID_TICKET"


class Team(BaseModel):
    team_id: str
    team_name: str
    skills: List[str] = Field(default_factory=list)
    active: bool = True
    business_hours_supported: bool = True
    description: Optional[str] = None


class Agent(BaseModel):
    agent_id: str
    agent_name: str
    team_id: str
    skills: List[str] = Field(default_factory=list)
    available: bool = True
    current_workload: int = 0
    max_workload: int = 10
    active: bool = True


class RoutingResult(BaseModel):
    ticket_id: str
    routing_status: RoutingStatus
    team_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    required_skill: Optional[str] = None
    reason: str
    workload: Optional[int] = None
    max_workload: Optional[int] = None
    routed_at: Optional[str] = None
    next_business_time: Optional[str] = None
    eligible_agents_considered: List[str] = Field(default_factory=list)
    skipped_agents: List[Dict[str, str]] = Field(default_factory=list)


class RoutingConfigUpdate(BaseModel):
    teams: Optional[List[Team]] = None
    agents: Optional[List[Agent]] = None
