"""
Task 3 Phase 4: Deterministic Duplicate Ticket Detector.

Hybrid approach:
1. Exact / near-exact text match
2. Structured field matching (course, order_id, issue category / skill, customer)
3. Deterministic content token stemming, cosine & Jaccard text similarity

Pure Python, offline, deterministic, zero external ML dependencies.
"""

from __future__ import annotations

import re
import math
import logging
from collections import Counter
from typing import List, Dict, Tuple, Optional, Set

from .models import TicketBase
from .phase4_models import (
    DuplicateStatus,
    DuplicateComparison,
    DuplicateCheckResponse,
)
from .duplicate_config import get_duplicate_config, DuplicateConfig
from .router import SkillDetector

logger = logging.getLogger(__name__)

STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "it", "in", "on", "for", "of", "to", "my", "i", "me",
    "we", "our", "you", "your", "he", "she", "they", "this", "that", "these",
    "those", "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "but", "and", "or", "as", "if", "by", "with",
    "at", "from", "after", "so", "though", "also", "just", "still"
}

STEM_MAP: Dict[str, str] = {
    "paid": "pay", "payment": "pay", "paying": "pay", "payments": "pay",
    "charge": "charg", "charged": "charg", "charges": "charg", "charging": "charg",
    "enrolled": "enroll", "enrollment": "enroll", "enrolling": "enroll", "enrolls": "enroll",
    "purchased": "purchas", "purchasing": "purchas", "purchase": "purchas", "purchases": "purchas",
    "videos": "video", "lectures": "lecture", "lessons": "lesson", "chapters": "chapter",
    "modules": "module", "quizzes": "quiz", "errors": "error", "errored": "error",
    "crashes": "crash", "crashing": "crash", "crashed": "crash",
    "submitting": "submit", "submission": "submit", "submitted": "submit", "submits": "submit",
    "failed": "fail", "failing": "fail", "failure": "fail", "fails": "fail",
    "working": "work", "works": "work", "worked": "work",
    "refunds": "refund", "refunding": "refund", "refunded": "refund",
    "buffering": "buffer", "buffered": "buffer",
    "accessing": "access", "accessible": "access",
}


def _stem(word: str) -> str:
    if word in STEM_MAP:
        return STEM_MAP[word]
    for suf in ["ing", "ed", "es", "s"]:
        if len(word) > len(suf) + 3 and word.endswith(suf):
            return word[:-len(suf)]
    return word


def _tokenize(text: str) -> List[str]:
    """Tokenize words, strip punctuation, remove stopwords, and apply light stemming."""
    if not text:
        return []
    clean = re.sub(r"[^\w\s-]", " ", text.lower())
    raw_words = [t.strip("-_") for t in clean.split() if len(t.strip("-_")) >= 1]
    return [_stem(w) for w in raw_words if w not in STOPWORDS and len(w) > 0]


def _cosine_similarity(counts_a: Counter[str], counts_b: Counter[str]) -> float:
    """Compute cosine similarity on sublinear term frequencies."""
    if not counts_a or not counts_b:
        return 0.0

    all_keys = set(counts_a.keys()) | set(counts_b.keys())
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for k in all_keys:
        val_a = 1.0 + math.log(counts_a[k]) if counts_a[k] > 0 else 0.0
        val_b = 1.0 + math.log(counts_b[k]) if counts_b[k] > 0 else 0.0
        dot += val_a * val_b
        norm_a += val_a * val_a
        norm_b += val_b * val_b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute token Jaccard similarity."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_text_similarity(text_a: Optional[str], text_b: Optional[str]) -> float:
    """
    Compute hybrid text similarity between two issue strings.
    Returns a float in [0.0, 1.0].
    """
    if not text_a or not text_b:
        return 0.0

    str_a = text_a.strip().lower()
    str_b = text_b.strip().lower()

    if not str_a or not str_b:
        return 0.0
    if str_a == str_b:
        return 1.0

    tokens_a = _tokenize(str_a)
    tokens_b = _tokenize(str_b)

    if not tokens_a or not tokens_b:
        return 0.0

    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)

    cos_sim = _cosine_similarity(counts_a, counts_b)
    jac_sim = _jaccard_similarity(tokens_a, tokens_b)

    return round(0.6 * cos_sim + 0.4 * jac_sim, 4)


def _normalize_course(course: Optional[str]) -> Optional[str]:
    if not course:
        return None
    c = course.strip().lower()
    for w in ["course", "the", "online", "bootcamp", "training"]:
        c = re.sub(rf"\b{w}\b", "", c)
    return re.sub(r"\s+", " ", c).strip()


class DuplicateDetector:
    """
    Evaluates duplicate likelihood between support tickets.
    Integrates structured matching, skill detection, and hybrid text similarity.
    """

    def __init__(
        self,
        config: Optional[DuplicateConfig] = None,
        skill_detector: Optional[SkillDetector] = None,
    ) -> None:
        self._config = config
        self._skill_detector = skill_detector or SkillDetector()

    @property
    def config(self) -> DuplicateConfig:
        return self._config if self._config is not None else get_duplicate_config()

    def _get_skill(self, ticket: TicketBase) -> Optional[str]:
        if ticket.required_skill:
            return ticket.required_skill
        text = f"{ticket.issue or ''} {ticket.evidence or ''}".strip()
        if text:
            skill, _ = self._skill_detector.detect(text)
            return skill
        return None

    def compare_tickets(self, ticket: TicketBase, other: TicketBase) -> DuplicateComparison:
        """
        Compare two tickets and determine duplicate status and similarity score.
        """
        cfg = self.config
        matching_fields: List[str] = []
        conflicting_fields: List[str] = []

        # 1. Base text similarity of the actual issue
        text_a = ticket.issue or ""
        text_b = other.issue or ""
        raw_text_sim = compute_text_similarity(text_a, text_b)

        # 2. Compare structured fields (Only if non-empty; never infer missing data)

        # 2a. Order ID
        order_a = ticket.order_id.strip() if ticket.order_id else None
        order_b = other.order_id.strip() if other.order_id else None
        if order_a and order_b:
            if order_a.lower() == order_b.lower():
                matching_fields.append("order_id")
            else:
                conflicting_fields.append("order_id")

        # 2b. Course name
        norm_course_a = _normalize_course(ticket.course_name)
        norm_course_b = _normalize_course(other.course_name)
        if norm_course_a and norm_course_b:
            if (
                norm_course_a == norm_course_b
                or norm_course_a in norm_course_b
                or norm_course_b in norm_course_a
            ):
                matching_fields.append("course")
            else:
                conflicting_fields.append("course")

        # 2c. Issue Category / Skill
        skill_a = self._get_skill(ticket)
        skill_b = self._get_skill(other)
        if skill_a and skill_b:
            if skill_a == skill_b:
                matching_fields.append("issue_category")
            else:
                # Related pairs are not conflicting (e.g. course_access and payment for access issues)
                compatible_pairs = {
                    ("course_access", "payment"),
                    ("payment", "course_access"),
                    ("course_access", "course_content"),
                    ("course_content", "course_access"),
                    ("payment", "refund"),
                    ("refund", "payment"),
                    ("technical", "login"),
                    ("login", "technical"),
                }
                if (skill_a, skill_b) not in compatible_pairs:
                    conflicting_fields.append("issue_category")

        # 2d. Customer match (informative only; customer alone never makes duplicate)
        cust_a = ticket.customer_id or ticket.contact_email
        cust_b = other.customer_id or other.contact_email
        if cust_a and cust_b and cust_a.strip().lower() == cust_b.strip().lower():
            matching_fields.append("customer")

        # 3. Score synthesis and safety adjustments
        score = raw_text_sim

        # Conflicting field penalties
        if "order_id" in conflicting_fields:
            # Different explicit order numbers strongly imply distinct purchases/issues
            score = min(score, 0.35)
        elif "course" in conflicting_fields:
            # Different explicit courses cannot be duplicates
            score = min(score, 0.35)
        elif "issue_category" in conflicting_fields and raw_text_sim < 0.80:
            # Fundamentally different issue category (e.g. refund vs account)
            score = min(score, 0.25)
        else:
            # Matching field boosts (only applied if text is already meaningfully similar)
            if raw_text_sim >= 0.40:
                if "order_id" in matching_fields:
                    # Same order ID + similar issue is a strong duplicate signal
                    score = min(1.0, score + 0.35)
                if "course" in matching_fields:
                    score = min(1.0, score + 0.20)
                if "issue_category" in matching_fields:
                    score = min(1.0, score + 0.10)

        score = round(score, 4)

        # 4. Determine Duplicate Status
        if score >= cfg.duplicate_threshold:
            status = DuplicateStatus.DUPLICATE
            explanation = (
                f"Issues are substantially similar (score: {score:.2f}) "
                f"with matching {', '.join(matching_fields) if matching_fields else 'context'}."
            )
        elif score >= cfg.possible_duplicate_threshold:
            status = DuplicateStatus.POSSIBLE_DUPLICATE
            explanation = (
                f"Moderate issue similarity (score: {score:.2f}). "
                f"Matches: {matching_fields or 'none'}, Conflicts: {conflicting_fields or 'none'}."
            )
        else:
            status = DuplicateStatus.NOT_DUPLICATE
            if conflicting_fields:
                explanation = (
                    f"Low issue similarity or conflicting fields: {', '.join(conflicting_fields)} "
                    f"(score: {score:.2f})."
                )
            else:
                explanation = f"Insufficient similarity between issue descriptions (score: {score:.2f})."

        return DuplicateComparison(
            compared_ticket_id=other.ticket_id,
            similarity_score=score,
            matching_fields=matching_fields,
            conflicting_fields=conflicting_fields,
            duplicate_status=status,
            explanation=explanation,
        )

    def check_ticket(
        self,
        ticket: TicketBase,
        existing_tickets: List[TicketBase],
    ) -> DuplicateCheckResponse:
        """
        Compare ticket against all existing tickets, returning the best match
        and detailed comparisons.
        """
        comparisons: List[DuplicateComparison] = []

        for other in existing_tickets:
            # Do not compare against self
            if other.ticket_id == ticket.ticket_id:
                continue
            cmp = self.compare_tickets(ticket, other)
            comparisons.append(cmp)

        # Sort descending by similarity score
        comparisons.sort(key=lambda c: c.similarity_score, reverse=True)

        if not comparisons:
            return DuplicateCheckResponse(
                ticket_id=ticket.ticket_id,
                duplicate_status=DuplicateStatus.NOT_DUPLICATE,
                similar_tickets=[],
                similarity_score=0.0,
                matching_fields=[],
                conflicting_fields=[],
                explanation="No existing tickets to compare against.",
            )

        best = comparisons[0]
        return DuplicateCheckResponse(
            ticket_id=ticket.ticket_id,
            duplicate_status=best.duplicate_status,
            similar_tickets=comparisons,
            similarity_score=best.similarity_score,
            matching_fields=best.matching_fields,
            conflicting_fields=best.conflicting_fields,
            explanation=best.explanation,
        )
