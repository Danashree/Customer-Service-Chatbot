"""
Task 3 Phase 4: Issue Grouping and Unrelated Issue Separation.

Groups tickets addressing related problems (e.g. same course access/content),
separates unrelated issues, and provides thread-safe group persistence.
"""

from __future__ import annotations

import os
import json
import logging
import threading
from datetime import datetime, timezone
from typing import List, Dict, Optional

from .models import TicketBase
from .phase4_models import (
    IssueGroup,
    RelationshipType,
    TicketRelationship,
)
from .duplicate_detector import DuplicateDetector, compute_text_similarity, _normalize_course
from .duplicate_config import get_duplicate_config, DuplicateConfig
from .router import SkillDetector

logger = logging.getLogger(__name__)


class GroupStorage:
    """
    JSON-based thread-safe atomic persistence for Issue Groups.
    Patterned identically to TicketStorage.
    """

    def __init__(self, store_path: Optional[str] = None) -> None:
        if store_path is None:
            store_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "issue_groups_store.json"
            )
        self._store_path = store_path
        self._lock = threading.Lock()
        self._ensure_store()

    def _ensure_store(self) -> None:
        if not os.path.exists(self._store_path):
            self._write({})

    def _read(self) -> Dict[str, dict]:
        try:
            with open(self._store_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: Dict[str, dict]) -> None:
        tmp_path = self._store_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self._store_path)

    def create(self, group: IssueGroup) -> IssueGroup:
        with self._lock:
            data = self._read()
            if group.group_id in data:
                raise ValueError(f"Group {group.group_id!r} already exists.")
            data[group.group_id] = group.model_dump()
            self._write(data)
            logger.info("IssueGroup created: %s", group.group_id)
        return group

    def get(self, group_id: str) -> Optional[IssueGroup]:
        with self._lock:
            data = self._read()
            raw = data.get(group_id)
        if raw is None:
            return None
        return IssueGroup(**raw)

    def update(self, group: IssueGroup) -> IssueGroup:
        with self._lock:
            data = self._read()
            group.updated_at = datetime.now(timezone.utc).isoformat()
            data[group.group_id] = group.model_dump()
            self._write(data)
            logger.info("IssueGroup %s updated", group.group_id)
        return group

    def delete(self, group_id: str) -> bool:
        with self._lock:
            data = self._read()
            if group_id in data:
                del data[group_id]
                self._write(data)
                logger.info("IssueGroup %s deleted", group_id)
                return True
            return False

    def list_all(self) -> List[IssueGroup]:
        with self._lock:
            data = self._read()
        groups = [IssueGroup(**v) for v in data.values()]
        groups.sort(key=lambda g: g.created_at, reverse=True)
        return groups

    def find_groups_for_ticket(self, ticket_id: str) -> List[IssueGroup]:
        with self._lock:
            data = self._read()
        return [
            IssueGroup(**v)
            for v in data.values()
            if ticket_id in v.get("ticket_ids", [])
        ]


class IssueGrouper:
    """
    Manages relationship classification and grouping operations for support tickets.
    """

    def __init__(
        self,
        storage: Optional[GroupStorage] = None,
        duplicate_detector: Optional[DuplicateDetector] = None,
        skill_detector: Optional[SkillDetector] = None,
        config: Optional[DuplicateConfig] = None,
    ) -> None:
        self._storage = storage or GroupStorage()
        self._duplicate_detector = duplicate_detector or DuplicateDetector(config=config)
        self._skill_detector = skill_detector or SkillDetector()
        self._config = config

    @property
    def config(self) -> DuplicateConfig:
        return self._config if self._config is not None else get_duplicate_config()

    @property
    def storage(self) -> GroupStorage:
        return self._storage

    def _get_skill(self, ticket: TicketBase) -> Optional[str]:
        if ticket.required_skill:
            return ticket.required_skill
        text = f"{ticket.issue or ''} {ticket.evidence or ''}".strip()
        if text:
            skill, _ = self._skill_detector.detect(text)
            return skill
        return None

    def classify_relationship(
        self,
        ticket: TicketBase,
        other: TicketBase,
    ) -> TicketRelationship:
        """
        Determine if two tickets are RELATED, DUPLICATE, or UNRELATED.
        Applies multi-signal evaluation: category/skill, course, text similarity.
        """
        cfg = self.config

        # Check duplicate first
        cmp = self._duplicate_detector.compare_tickets(ticket, other)
        if cmp.duplicate_status.value == "DUPLICATE":
            return TicketRelationship(
                ticket_id=other.ticket_id,
                relationship=RelationshipType.DUPLICATE,
                similarity_score=cmp.similarity_score,
                reason=f"Duplicate issue detected ({cmp.explanation})",
            )

        skill_a = self._get_skill(ticket)
        skill_b = self._get_skill(other)
        course_a = _normalize_course(ticket.course_name)
        course_b = _normalize_course(other.course_name)

        same_course = bool(
            course_a and course_b and (course_a == course_b or course_a in course_b or course_b in course_a)
        )
        same_skill = bool(skill_a and skill_b and skill_a == skill_b)

        # Explicit unrelated check: Same customer + completely different issue & category
        cust_a = ticket.customer_id or ticket.contact_email
        cust_b = other.customer_id or other.contact_email
        same_customer = bool(cust_a and cust_b and cust_a.strip().lower() == cust_b.strip().lower())

        if not same_course and not same_skill and cmp.similarity_score < cfg.related_threshold:
            reason = (
                f"Issues are in different domains ({skill_a or 'unknown'} vs {skill_b or 'unknown'}) "
                f"with low similarity ({cmp.similarity_score:.2f})."
            )
            if same_customer:
                reason += " Same customer with separate unrelated requests."
            return TicketRelationship(
                ticket_id=other.ticket_id,
                relationship=RelationshipType.UNRELATED,
                similarity_score=cmp.similarity_score,
                reason=reason,
            )

        # Related conditions:
        # 1. Same course + related category (e.g. course_access and course_content)
        course_skills = {"course_access", "course_content"}
        both_course_related = bool(
            same_course and (skill_a in course_skills or skill_b in course_skills)
        )

        # 2. Same category with moderate text similarity
        category_related = bool(same_skill and cmp.similarity_score >= 0.35)

        # 3. Overall similarity >= related_threshold
        score_related = cmp.similarity_score >= cfg.related_threshold

        if both_course_related or category_related or score_related:
            topic_desc = []
            if same_course:
                topic_desc.append(f"course '{ticket.course_name}'")
            if same_skill:
                topic_desc.append(f"skill '{skill_a}'")
            
            # Related issues have a combined context similarity
            rel_score = max(cmp.similarity_score, 0.40 if same_course else 0.35)
            reason_str = (
                f"Related issues sharing {', '.join(topic_desc) if topic_desc else 'similar context'} "
                f"(similarity: {rel_score:.2f})."
            )
            return TicketRelationship(
                ticket_id=other.ticket_id,
                relationship=RelationshipType.RELATED,
                similarity_score=rel_score,
                reason=reason_str,
            )

        # Default fallback is UNRELATED
        return TicketRelationship(
            ticket_id=other.ticket_id,
            relationship=RelationshipType.UNRELATED,
            similarity_score=cmp.similarity_score,
            reason=f"Distinct, unrelated issues (similarity: {cmp.similarity_score:.2f}).",
        )

    def find_relationships(
        self,
        ticket: TicketBase,
        existing_tickets: List[TicketBase],
    ) -> List[TicketRelationship]:
        """
        Evaluate relationships of ticket against all other existing tickets.
        """
        relationships: List[TicketRelationship] = []
        for other in existing_tickets:
            if other.ticket_id == ticket.ticket_id:
                continue
            rel = self.classify_relationship(ticket, other)
            relationships.append(rel)
        relationships.sort(key=lambda r: r.similarity_score, reverse=True)
        return relationships

    def create_group(
        self,
        tickets: List[TicketBase],
        group_id: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> IssueGroup:
        """
        Create a new IssueGroup for a set of related tickets.
        """
        if not tickets:
            raise ValueError("Cannot create an issue group with empty tickets.")

        tids = [t.ticket_id for t in tickets]
        if not topic:
            course = next((t.course_name for t in tickets if t.course_name), "General")
            skill = next((self._get_skill(t) for t in tickets if self._get_skill(t)), "Support")
            topic = f"{course} - {skill.replace('_', ' ').title()}"

        group_kwargs: dict = {
            "ticket_ids": tids,
            "group_topic": topic,
            "relationship": "RELATED",
        }
        if group_id:
            group_kwargs["group_id"] = group_id

        group = IssueGroup(**group_kwargs)
        return self._storage.create(group)

    def add_ticket_to_group(self, group_id: str, ticket_id: str) -> Optional[IssueGroup]:
        """
        Add a ticket to an existing issue group.
        """
        group = self._storage.get(group_id)
        if not group:
            return None
        if ticket_id not in group.ticket_ids:
            group.ticket_ids.append(ticket_id)
            self._storage.update(group)
        return group

    def remove_ticket_from_group(self, group_id: str, ticket_id: str) -> Optional[IssueGroup]:
        """
        Remove a ticket from an issue group.
        """
        group = self._storage.get(group_id)
        if not group:
            return None
        if ticket_id in group.ticket_ids:
            group.ticket_ids.remove(ticket_id)
            self._storage.update(group)
        return group

    def get_group(self, group_id: str) -> Optional[IssueGroup]:
        return self._storage.get(group_id)

    def list_groups(self) -> List[IssueGroup]:
        return self._storage.list_all()
