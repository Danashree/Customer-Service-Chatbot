"""
Task 4 Phase 2 — Policy Conflict Resolution
=============================================
Resolves conflicts when multiple versions of the same policy are retrieved
for the same product + region + applicable date.

Design rules (per Phase 2 spec)
--------------------------------
Conflict definition:
    Two chunks are "conflicting" when they share the same:
      - document_type == "policy"
      - product (normalised)
      - region  (normalised)
    and represent different document versions / document IDs.

Resolution strategy:
    1. Group policy chunks by (product, region).
    2. Within each group, identify distinct documents (by document_id).
    3. Select the document with the LATEST effective_date among those already
       date-filtered (i.e. all of them are applicable for the reference date).
    4. If two documents share the same effective_date, use document_version as
       a deterministic tie-breaker (parsed as semver tuples; higher wins).
    5. Retain ALL chunks from the winning document_id; discard the rest.

Non-policy documents (FAQ, how_to, troubleshooting, general):
    These are additive evidence, not conflicting policies.
    ALL such chunks are always preserved unchanged.

Selection reason:
    Each surviving chunk is annotated with a human-readable ``selection_reason``
    explaining why it was kept.  Policy-winner chunks receive
    "Latest applicable policy (vX) for reference date YYYY-MM-DD".
    Non-policy chunks receive "Relevant evidence (non-policy)".
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_version(version_str: str) -> Tuple:
    """
    Parse a version string into a comparable tuple.

    Examples:
      "1.0"   → (1, 0)
      "2.3.1" → (2, 3, 1)
      "3"     → (3,)
      "abc"   → (0,)   ← fallback for non-numeric strings
    """
    try:
        parts = tuple(int(x) for x in str(version_str).split(".") if x.strip().isdigit())
        return parts if parts else (0,)
    except (ValueError, AttributeError):
        return (0,)


def _effective_date_from_meta(meta: dict) -> date:
    """Extract and parse effective_date from a flat metadata dict."""
    raw = meta.get("effective_date", "1970-01-01")
    if isinstance(raw, date):
        return raw
    return date.fromisoformat(str(raw))


def _group_key(meta: dict) -> Tuple[str, str]:
    """Grouping key for policy conflict detection: (product, region)."""
    return (
        str(meta.get("product", "global")).lower(),
        str(meta.get("region", "global")).lower(),
    )


def _is_policy(meta: dict) -> bool:
    """True if this document is of type 'policy'."""
    return str(meta.get("document_type", "")).lower() == "policy"


# ---------------------------------------------------------------------------
# Winning-document selector
# ---------------------------------------------------------------------------

def _select_winning_document_id(
    doc_id_to_meta: Dict[str, dict],
) -> str:
    """
    Given a mapping of {document_id → representative_metadata},
    return the document_id of the "best" (latest applicable) policy.

    Ordering (highest priority first):
      1. Latest effective_date  (higher date wins)
      2. Highest document_version  (as semver tuple, higher wins)
      3. document_id lexicographic (stable final tie-break)
    """
    def sort_key(item: Tuple[str, dict]) -> Tuple:
        doc_id, meta = item
        eff = _effective_date_from_meta(meta)
        ver = _parse_version(str(meta.get("document_version", "0")))
        return (eff, ver, doc_id)

    winner_id, _ = max(doc_id_to_meta.items(), key=sort_key)
    return winner_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ResolvedResult:
    """
    Lightweight container pairing a LangChain Document with its
    selection reason (for debugging / test assertions).
    """
    __slots__ = ("doc", "selection_reason")

    def __init__(self, doc: Document, selection_reason: str) -> None:
        self.doc = doc
        self.selection_reason = selection_reason


def resolve_policy_conflicts(
    docs: List[Document],
    reference_date: Optional[date] = None,
) -> List[ResolvedResult]:
    """
    Apply policy conflict resolution to a filtered list of documents.

    Steps
    -----
    1. Separate policy docs from non-policy docs.
    2. For policy docs, group by (product, region).
    3. Within each group, pick the winning document_id.
    4. Retain all chunks from the winner; discard the rest within that group.
    5. Annotate every surviving item with a ``selection_reason``.

    Parameters
    ----------
    docs:
        Documents that have already passed access / product / region / date
        filtering.  All are assumed to be applicable for the reference date.
    reference_date:
        Used only for the selection_reason string; does not affect logic
        (filtering is already done before this function is called).

    Returns
    -------
    List[ResolvedResult]
        Documents that survived conflict resolution, each tagged with a reason.
    """
    ref_str = reference_date.isoformat() if reference_date else "current date"

    policy_docs: List[Document] = []
    non_policy_docs: List[Document] = []

    for doc in docs:
        if _is_policy(doc.metadata):
            policy_docs.append(doc)
        else:
            non_policy_docs.append(doc)

    # ------------------------------------------------------------------
    # Resolve policy conflicts group by group
    # ------------------------------------------------------------------
    # Build groups: (product, region) → { document_id → representative_meta }
    groups: Dict[Tuple[str, str], Dict[str, dict]] = {}
    group_chunks: Dict[Tuple[str, str], List[Document]] = {}

    for doc in policy_docs:
        key = _group_key(doc.metadata)
        doc_id = str(doc.metadata.get("document_id", ""))

        if key not in groups:
            groups[key] = {}
            group_chunks[key] = []

        # Use the first chunk of each document_id as representative metadata
        if doc_id not in groups[key]:
            groups[key][doc_id] = doc.metadata

        group_chunks[key].append(doc)

    resolved: List[ResolvedResult] = []

    for key, id_to_meta in groups.items():
        if len(id_to_meta) == 1:
            # Only one document in this group — no conflict
            winner_id = next(iter(id_to_meta))
            winner_meta = id_to_meta[winner_id]
        else:
            # Multiple documents — pick the latest applicable
            winner_id = _select_winning_document_id(id_to_meta)
            winner_meta = id_to_meta[winner_id]

        winner_version = winner_meta.get("document_version", "?")
        reason = (
            f"Latest applicable policy (v{winner_version}) "
            f"for reference date {ref_str}"
        )

        for doc in group_chunks[key]:
            if str(doc.metadata.get("document_id", "")) == winner_id:
                resolved.append(ResolvedResult(doc=doc, selection_reason=reason))
            # chunks from losing documents are silently dropped

    # ------------------------------------------------------------------
    # Non-policy documents: keep all
    # ------------------------------------------------------------------
    for doc in non_policy_docs:
        resolved.append(
            ResolvedResult(
                doc=doc,
                selection_reason="Relevant evidence (non-policy)",
            )
        )

    return resolved
