"""
Task 3 Phase 4: Pydantic Data Models for Duplicate Detection,
Issue Grouping, Relationships, and Masked Handoff.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class DuplicateStatus(str, Enum):
    DUPLICATE = "DUPLICATE"
    POSSIBLE_DUPLICATE = "POSSIBLE_DUPLICATE"
    NOT_DUPLICATE = "NOT_DUPLICATE"


class RelationshipType(str, Enum):
    DUPLICATE = "DUPLICATE"
    RELATED = "RELATED"
    UNRELATED = "UNRELATED"


# ──────────────────────────────────────────────────────────────────────────────
# Duplicate Models
# ──────────────────────────────────────────────────────────────────────────────

class DuplicateComparison(BaseModel):
    compared_ticket_id: str
    similarity_score: float
    matching_fields: List[str] = Field(default_factory=list)
    conflicting_fields: List[str] = Field(default_factory=list)
    duplicate_status: DuplicateStatus
    explanation: str


class DuplicateCheckResponse(BaseModel):
    ticket_id: str
    duplicate_status: DuplicateStatus
    similar_tickets: List[DuplicateComparison] = Field(default_factory=list)
    similarity_score: float = 0.0
    matching_fields: List[str] = Field(default_factory=list)
    conflicting_fields: List[str] = Field(default_factory=list)
    explanation: str


# ──────────────────────────────────────────────────────────────────────────────
# Relationship Models
# ──────────────────────────────────────────────────────────────────────────────

class TicketRelationship(BaseModel):
    ticket_id: str
    relationship: RelationshipType
    similarity_score: float
    reason: str


class RelationshipResponse(BaseModel):
    ticket_id: str
    relationships: List[TicketRelationship] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Issue Group Models
# ──────────────────────────────────────────────────────────────────────────────

class IssueGroup(BaseModel):
    group_id: str = Field(default_factory=lambda: f"GRP-{uuid.uuid4().hex[:6].upper()}")
    ticket_ids: List[str] = Field(default_factory=list)
    group_topic: str
    relationship: str = "RELATED"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GroupCreateRequest(BaseModel):
    group_id: Optional[str] = None
    group_topic: Optional[str] = None


class GroupResponse(BaseModel):
    group: IssueGroup


class GroupListResponse(BaseModel):
    groups: List[IssueGroup]
    total: int


# ──────────────────────────────────────────────────────────────────────────────
# Handoff Models
# ──────────────────────────────────────────────────────────────────────────────

class HandoffSummary(BaseModel):
    ticket_id: str
    group_id: Optional[str] = None
    masked_summary: str
    structured_summary: Dict[str, Any]
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
