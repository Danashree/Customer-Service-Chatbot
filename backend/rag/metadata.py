"""
Task 4 Phase 1 — Metadata Normalization & Helpers
===================================================
Utility functions for extracting and normalising document metadata
before it is stored in the knowledge base.
"""
from __future__ import annotations

import os
import re
from datetime import date
from typing import Any, Dict, Optional

from .models import AccessLevel, DocumentMetadata, DocumentType


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def parse_iso_date(value: Any) -> Optional[date]:
    """
    Convert a value to a date object.

    Accepted forms:
      - ``date`` objects → returned as-is
      - ISO-8601 string ``"YYYY-MM-DD"``
      - ``None`` → returns ``None``
    """
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        value = value.strip()
        return date.fromisoformat(value)
    raise TypeError(f"Cannot parse date from {type(value).__name__}: {value!r}")


def is_date_active(
    effective_date: date,
    expiry_date: Optional[date],
    *,
    reference_date: Optional[date] = None,
) -> bool:
    """
    Return True if the document is currently active (i.e. effective_date <= today < expiry_date).

    Parameters
    ----------
    effective_date : date
        The date from which the document version is effective.
    expiry_date : date | None
        The date after which the document expires.  ``None`` means no expiry.
    reference_date : date | None
        The date to treat as "today".  Defaults to ``date.today()``.
    """
    today = reference_date or date.today()
    if today < effective_date:
        return False  # not yet effective (future document)
    if expiry_date is not None and today >= expiry_date:
        return False  # expired
    return True


# ---------------------------------------------------------------------------
# Access-level helpers
# ---------------------------------------------------------------------------

def normalize_access_level(value: Any) -> AccessLevel:
    """
    Coerce various representations to an ``AccessLevel``.

    Accepted forms:
      - ``AccessLevel`` enum member
      - Integer ``0–3``
      - Case-insensitive string ``"public" | "customer" | "agent" | "admin"``
    """
    if isinstance(value, AccessLevel):
        return value
    if isinstance(value, int):
        return AccessLevel(value)
    if isinstance(value, str):
        return AccessLevel.from_str(value)
    raise TypeError(f"Cannot coerce {type(value).__name__} to AccessLevel.")


def caller_can_access(
    required_level: AccessLevel,
    caller_level: AccessLevel,
) -> bool:
    """
    Return True if ``caller_level`` is sufficient to read a document
    that requires ``required_level``.

    The hierarchy is: public(0) < customer(1) < agent(2) < admin(3).
    A caller at level N may access documents at levels 0 … N.
    """
    return int(caller_level) >= int(required_level)


# ---------------------------------------------------------------------------
# Source-file helpers
# ---------------------------------------------------------------------------

def infer_document_type(source_file: str) -> str:
    """
    Heuristically infer a document type from the filename.

    Mapping:
      - ``faq`` in name  → ``DocumentType.FAQ``
      - ``policy``       → ``DocumentType.POLICY``
      - ``how_to``       → ``DocumentType.HOW_TO``
      - ``troubleshoot`` → ``DocumentType.TROUBLESHOOTING``
      - otherwise        → ``DocumentType.GENERAL``
    """
    name = os.path.basename(source_file).lower()
    if "faq" in name:
        return DocumentType.FAQ
    if "policy" in name or "policies" in name:
        return DocumentType.POLICY
    if "how_to" in name or "howto" in name or "how-to" in name or "guide" in name:
        return DocumentType.HOW_TO
    if "troubleshoot" in name:
        return DocumentType.TROUBLESHOOTING
    return DocumentType.GENERAL


def normalize_product(product: str) -> str:
    """Lowercase and underscore-normalize a product string."""
    return product.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_region(region: str) -> str:
    """Lowercase and strip a region string."""
    return region.strip().lower()


# ---------------------------------------------------------------------------
# Metadata dict helpers (for FAISS round-trip)
# ---------------------------------------------------------------------------

def metadata_to_flat_dict(meta: DocumentMetadata) -> Dict[str, Any]:
    """Convert ``DocumentMetadata`` to a flat JSON-serializable dict."""
    return meta.to_faiss_metadata()


def metadata_from_flat_dict(flat: Dict[str, Any]) -> DocumentMetadata:
    """Reconstruct ``DocumentMetadata`` from a flat FAISS metadata dict."""
    return DocumentMetadata.from_faiss_metadata(flat)


def inherit_metadata(
    parent_meta: DocumentMetadata,
    *,
    chunk_id: Optional[str] = None,
    source_page: Optional[int] = None,
) -> DocumentMetadata:
    """
    Derive child chunk metadata from a parent ``DocumentMetadata``.

    Copies all fields; optionally overrides ``chunk_id`` and ``source_page``.
    Used by the chunker to stamp each chunk with a fresh ``chunk_id`` while
    preserving all other metadata fields intact.
    """
    import uuid as _uuid
    flat = parent_meta.to_faiss_metadata()
    flat["chunk_id"] = chunk_id or str(_uuid.uuid4())
    if source_page is not None:
        flat["source_page"] = source_page
    return DocumentMetadata.from_faiss_metadata(flat)
