"""
Task 4 Phase 2 — Date-Aware Document Filtering
================================================
Provides deterministic, testable filtering of document chunks based on
effective and expiry dates.

Design rules (per Phase 2 spec)
--------------------------------
* Inclusive boundary: effective_date <= reference_date <= expiry_date.
* null expiry: document never expires; condition is just effective_date <= ref.
* Future document (effective_date > reference_date): EXCLUDED.
* Expired document (expiry_date < reference_date): EXCLUDED.
* Currently active document: INCLUDED.
* All date comparisons are deterministic; callers supply the reference_date.
  This module never calls date.today() internally.

Query semantics
---------------
* current question  → caller supplies reference_date = today's date
* historical question → caller supplies reference_date = the historical date
  The filtering logic is identical; only the supplied date differs.
"""
from __future__ import annotations

from datetime import date
from typing import Any, List, Optional, Union

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Low-level predicate
# ---------------------------------------------------------------------------

def is_document_active_on(
    meta: dict,
    reference_date: date,
) -> bool:
    """
    Return True if the document described by *meta* is active on *reference_date*.

    Applies inclusive boundaries:
        effective_date <= reference_date  AND
        (expiry_date is None  OR  reference_date <= expiry_date)

    Parameters
    ----------
    meta:
        Flat metadata dict (as stored in LangChain ``Document.metadata``).
        Must contain ``"effective_date"`` (ISO-8601 string).
        May contain ``"expiry_date"`` (ISO-8601 string or ``None``).
    reference_date:
        The date to evaluate against.

    Returns
    -------
    bool
    """
    effective_raw = meta.get("effective_date")
    if effective_raw is None:
        # No date metadata — not a RAG-managed doc; caller decides what to do.
        return True

    effective = _parse_date(effective_raw)

    # Future document?
    if reference_date < effective:
        return False

    # Expired?
    expiry_raw = meta.get("expiry_date")
    if expiry_raw is not None:
        expiry = _parse_date(expiry_raw)
        if reference_date > expiry:
            return False

    return True


def _parse_date(value: Any) -> date:
    """Parse ISO-8601 string or date object to date."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


# ---------------------------------------------------------------------------
# Document-list filter
# ---------------------------------------------------------------------------

def filter_documents_by_date(
    docs: List[Document],
    reference_date: date,
) -> List[Document]:
    """
    Filter a list of LangChain ``Document`` objects to those active on
    *reference_date*.

    Documents that do NOT carry ``effective_date`` metadata (i.e. non-RAG Task 1
    dataset entries) are **excluded** — they are outside the RAG namespace.

    Parameters
    ----------
    docs:
        Source list (typically the output of access/product/region filtering).
    reference_date:
        Reference date for the filter.

    Returns
    -------
    List[Document]
        Subset of *docs* that are active on *reference_date*.
    """
    result: List[Document] = []
    for doc in docs:
        meta = doc.metadata
        if "effective_date" not in meta:
            # Skip non-RAG documents entirely at this stage
            continue
        if is_document_active_on(meta, reference_date):
            result.append(doc)
    return result


# ---------------------------------------------------------------------------
# Convenience: coerce reference_date from various input types
# ---------------------------------------------------------------------------

def resolve_reference_date(
    reference_date: Optional[Union[str, date]],
    *,
    default_today: bool = True,
) -> Optional[date]:
    """
    Normalise *reference_date* to a ``date`` object.

    * ``date`` → returned as-is.
    * ISO-8601 ``str`` → parsed.
    * ``None`` with ``default_today=True`` → ``date.today()`` (current question).
    * ``None`` with ``default_today=False`` → ``None`` (caller handles it).
    """
    if reference_date is None:
        return date.today() if default_today else None
    if isinstance(reference_date, date):
        return reference_date
    return date.fromisoformat(str(reference_date))
