"""
Task 6 Phase 1 — Entity Preserver
==================================
Identifies, protects, and restores sensitive and business-critical entities:
  - Names
  - Order IDs
  - Dates
  - Product codes
  - Email addresses, Phone numbers, URLs

Guarantees:
  - Entities are shielded with non-translatable placeholders before any
    spelling normalization or translation occurs.
  - Entities are restored with 100% exact character fidelity.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from .models import EntityPreservationResult, EntityType, PreservedEntity


ORDER_ID_PATTERN = re.compile(
    r"\b(?:ORD(?:ER)?(?:[-_][0-9A-Z]+)+|ORD(?:ER)?[0-9A-Z]{3,12})\b",
    re.IGNORECASE,
)


class EntityPreserver:
    """Detects and shields business entities to prevent alteration during processing."""

    # Pre-compiled patterns ordered by specificity
    PATTERNS: List[Tuple[EntityType, re.Pattern]] = [
        # URLs
        (
            EntityType.URL,
            re.compile(r"\bhttps?://[^\s/$.?#].[^\s]*\b", re.IGNORECASE),
        ),
        # Emails
        (
            EntityType.EMAIL,
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ),
        # Order IDs: e.g. ORD12345, ORD-2026-001, ORDER-9988, ORD_5544
        (
            EntityType.ORDER_ID,
            ORDER_ID_PATTERN,
        ),
        # Product codes: e.g. PROD-AX21, ABC-PROD-01, COURSE-PY-101
        (
            EntityType.PRODUCT_CODE,
            re.compile(
                r"\b(?:[A-Z0-9]{2,5}[-_])?PROD[-_][0-9A-Z]{2,8}\b|\bCOURSE[-_][A-Z0-9]{2,8}(?:[-_]\d{1,4})?\b",
                re.IGNORECASE,
            ),
        ),
        # Dates: e.g. 15-09-2026, 2026/09/15, 15/09/2026, September 15, 2026, 15 Sept 2026
        (
            EntityType.DATE,
            re.compile(
                r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|"
                r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
                r"(?:\s+\d{1,2}(?:st|nd|rd|th)?)?,?\s+\d{4}\b",
                re.IGNORECASE,
            ),
        ),
        # Phone numbers: standard formatted or 10-digit standalone
        (
            EntityType.PHONE,
            re.compile(r"(?:\+?\d{1,3}[-.\\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b[6-9]\d{9}\b"),
        ),
    ]

    # Greeting-based name pattern: "Hi Danashree", "Hello Rahul"
    NAME_GREETING_PATTERN = re.compile(
        r"(?i)\b(?:hi|hello|hey|dear)\s+([A-Z][a-z]+)\b"
    )

    # Known customer service domain names
    KNOWN_NAMES = {
        "danashree", "rahul", "priya", "arun", "ananya", "vijay", "deepa",
        "karthik", "suresh", "ramesh", "sneha", "rohit", "pooja", "aditya"
    }

    @classmethod
    def extract_entities(cls, text: str) -> List[Tuple[EntityType, str, int, int]]:
        """
        Extracts entities from text without overlapping spans.
        Returns list of (entity_type, value, start, end) sorted by start index.
        """
        if not text:
            return []

        spans: List[Tuple[EntityType, str, int, int]] = []
        occupied_indices = set()

        def _is_overlapping(s: int, e: int) -> bool:
            return any(i in occupied_indices for i in range(s, e))

        def _mark_occupied(s: int, e: int) -> None:
            occupied_indices.update(range(s, e))

        # 1. Specific standard patterns
        for entity_type, pattern in cls.PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span()
                if not _is_overlapping(start, end):
                    val = match.group(0)
                    spans.append((entity_type, val, start, end))
                    _mark_occupied(start, end)

        # 2. Greeting-based names
        for match in cls.NAME_GREETING_PATTERN.finditer(text):
            # Capture group 1 is the name
            name_val = match.group(1)
            start = match.start(1)
            end = match.end(1)
            if not _is_overlapping(start, end):
                spans.append((EntityType.NAME, name_val, start, end))
                _mark_occupied(start, end)

        # 3. Known domain names as standalone words
        for match in re.finditer(r"\b([A-Za-z]+)\b", text):
            word = match.group(1)
            if word.lower() in cls.KNOWN_NAMES:
                start, end = match.span(1)
                if not _is_overlapping(start, end):
                    spans.append((EntityType.NAME, word, start, end))
                    _mark_occupied(start, end)

        # Sort spans by start index
        spans.sort(key=lambda x: x[2])
        return spans

    @classmethod
    def protect(cls, text: str) -> EntityPreservationResult:
        """
        Replaces detected entities with protective placeholders.
        Returns EntityPreservationResult containing protected text and entity metadata.
        """
        if not text:
            return EntityPreservationResult(original_text="", protected_text="", entities=[])

        raw_entities = cls.extract_entities(text)
        if not raw_entities:
            return EntityPreservationResult(original_text=text, protected_text=text, entities=[])

        preserved_entities: List[PreservedEntity] = []
        protected_text = text

        # Replace from end to start so indices remain valid
        for i, (etype, val, start, end) in enumerate(reversed(raw_entities)):
            placeholder = f"__PRESERVED_{etype.value.upper()}_{len(raw_entities) - 1 - i}__"
            protected_text = protected_text[:start] + placeholder + protected_text[end:]
            preserved_entities.append(
                PreservedEntity(
                    entity_type=etype,
                    value=val,
                    start=start,
                    end=end,
                    placeholder=placeholder,
                )
            )

        # Reverse back to original order
        preserved_entities.reverse()

        return EntityPreservationResult(
            original_text=text,
            protected_text=protected_text,
            entities=preserved_entities,
        )

    @classmethod
    def restore(cls, text: str, entities: List[PreservedEntity]) -> str:
        """
        Restores preserved entity values from placeholders.
        """
        if not text or not entities:
            return text

        restored = text
        for entity in entities:
            restored = restored.replace(entity.placeholder, entity.value)
        return restored
