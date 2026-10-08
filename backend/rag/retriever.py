"""
Task 4 Phase 1 & 2 — Authorization-Aware, Date-Aware RAG Retriever
====================================================================
Retrieves document chunks from the active Task 1 FAISS index, applying:

  Phase 1  ─ access-level, product, and region filters
  Phase 2  ─ effective/expiry date filtering + policy conflict resolution

Key design constraints
-----------------------
1. MUST call ``langchain_helper.get_active_faiss_path()`` — no hardcoded path.
2. Authorization filtering MUST happen BEFORE returning results.
3. No silent fallback to any other index path.
4. ``AccessLevel`` hierarchy: public(0) < customer(1) < agent(2) < admin(3).
5. Returns ``List[RetrievedChunk]`` with full provenance + selection_reason.

Full retrieval pipeline (Phase 2)
----------------------------------
Query
 ↓
Access authorization
 ↓
Product filter
 ↓
Region filter
 ↓
Effective/expiry date filter   ← Phase 2
 ↓
Policy conflict resolution     ← Phase 2
 ↓
Relevant evidence (top_k)

Date semantics
--------------
* ``reference_date=None``   → current question: uses date.today().
* ``reference_date=<date>`` → historical or simulated question.
  Tests MUST supply an explicit date to remain deterministic.

Filtering semantics (Phase 1, preserved)
-----------------------------------------
* ``user_access_level``  — required.
* ``product``            — optional; None / "*" = no filter; "global" docs always pass.
* ``region``             — optional; None / "*" = no filter; "global" docs always pass.
* ``top_k``              — maximum number of chunks returned after all filtering.
"""
from __future__ import annotations

import os
from datetime import date
from typing import List, Optional, Union

from langchain_core.documents import Document

from .models import AccessLevel, DocumentMetadata, DocumentType, RetrievedChunk
from .metadata import (
    caller_can_access,
    metadata_from_flat_dict,
    normalize_product,
    normalize_region,
)
from .provenance import build_provenance
from .date_filter import filter_documents_by_date, resolve_reference_date
from .conflict_resolver import resolve_policy_conflicts


# ---------------------------------------------------------------------------
# Lazy embeddings / active path loaders
# ---------------------------------------------------------------------------

def _get_embeddings():
    """Return the shared HuggingFaceInstructEmbeddings from langchain_helper."""
    import sys
    _backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _backend_dir not in sys.path:
        sys.path.insert(0, _backend_dir)
    from langchain_helper import instructor_embeddings  # noqa: PLC0415
    return instructor_embeddings


def _get_active_path() -> str:
    """Call get_active_faiss_path() from langchain_helper. No silent fallback."""
    import sys
    _backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _backend_dir not in sys.path:
        sys.path.insert(0, _backend_dir)
    from langchain_helper import get_active_faiss_path  # noqa: PLC0415
    return get_active_faiss_path()


# ---------------------------------------------------------------------------
# Phase 1 filter helpers (preserved verbatim)
# ---------------------------------------------------------------------------

def _access_level_passes(doc_level: int, caller_level: AccessLevel) -> bool:
    """True if the caller has sufficient access."""
    return int(caller_level) >= doc_level


def _product_passes(doc_product: str, filter_product: Optional[str]) -> bool:
    """True if the document matches the product filter."""
    if filter_product is None or filter_product == "*":
        return True
    fp = normalize_product(filter_product)
    dp = normalize_product(doc_product)
    return dp == "global" or dp == fp


def _region_passes(doc_region: str, filter_region: Optional[str]) -> bool:
    """True if the document matches the region filter."""
    if filter_region is None or filter_region == "*":
        return True
    fr = normalize_region(filter_region)
    dr = normalize_region(doc_region)
    return dr == "global" or dr == fr


def _normalize_doc_metadata(doc: Document) -> Optional[Document]:
    """
    Ensures doc.metadata has full Task 4 RAG metadata.
    If already populated with access_level and effective_date, returns doc as-is.
    If from Task 1 pipeline or bootstrap dataset, maps to canonical metadata.
    """
    meta = dict(doc.metadata)
    if "access_level" in meta and "effective_date" in meta:
        return doc

    # Only adapt if it's a known Task 1 document or has identifiable source/row
    if not any(k in meta for k in ("source", "source_file", "row", "doc_type")):
        return None

    source = str(meta.get("source", meta.get("source_file", ""))).lower()
    page = meta.get("page", meta.get("source_page", 1))

    if "policy" in source:
        spec = {
            "source_file": "knowledge_base/course_policies.pdf",
            "document_version": "1.0",
            "document_type": DocumentType.POLICY,
            "product": "general",
            "region": "global",
            "access_level": AccessLevel.customer,
            "effective_date": "2024-01-01",
            "expiry_date": None,
            "source_page": page,
        }
    elif "faq" in source:
        spec = {
            "source_file": "knowledge_base/course_faqs.pdf",
            "document_version": "1.0",
            "document_type": DocumentType.FAQ,
            "product": "general",
            "region": "global",
            "access_level": AccessLevel.public,
            "effective_date": "2024-01-01",
            "expiry_date": None,
            "source_page": page,
        }
    elif "how_to" in source or "guide" in source:
        spec = {
            "source_file": "knowledge_base/course_how_to_guides.pdf",
            "document_version": "1.0",
            "document_type": DocumentType.HOW_TO,
            "product": "general",
            "region": "global",
            "access_level": AccessLevel.customer,
            "effective_date": "2024-01-01",
            "expiry_date": None,
            "source_page": page,
        }
    elif "row" in meta or "prompt" in meta or "dataset" in source:
        spec = {
            "source_file": "dataset/dataset.csv",
            "document_version": "1.0",
            "document_type": DocumentType.FAQ,
            "product": "general",
            "region": "global",
            "access_level": AccessLevel.public,
            "effective_date": "2024-01-01",
            "expiry_date": None,
            "source_page": None,
        }
    else:
        spec = {
            "source_file": source or "knowledge_base/general.pdf",
            "document_version": "1.0",
            "document_type": DocumentType.GENERAL,
            "product": "general",
            "region": "global",
            "access_level": AccessLevel.public,
            "effective_date": "2024-01-01",
            "expiry_date": None,
            "source_page": page,
        }

    enriched_meta = DocumentMetadata(**spec).to_faiss_metadata()
    return Document(page_content=doc.page_content, metadata=enriched_meta)


def _apply_filters(
    docs: List[Document],
    caller_level: AccessLevel,
    filter_product: Optional[str],
    filter_region: Optional[str],
) -> List[Document]:
    """
    Return only the docs that pass access_level, product, and region filters.
    Non-RAG documents (missing recognizable metadata) are skipped silently.
    """
    passed: List[Document] = []
    for doc in docs:
        norm_doc = _normalize_doc_metadata(doc)
        if norm_doc is None:
            continue

        meta_dict = norm_doc.metadata

        doc_level = meta_dict.get("access_level", 0)
        if not _access_level_passes(doc_level, caller_level):
            continue

        doc_product = meta_dict.get("product", "global")
        if not _product_passes(doc_product, filter_product):
            continue

        doc_region = meta_dict.get("region", "global")
        if not _region_passes(doc_region, filter_region):
            continue

        passed.append(norm_doc)
    return passed


# ---------------------------------------------------------------------------
# Public retriever  (Phase 1 + Phase 2)
# ---------------------------------------------------------------------------

def retrieve_knowledge(
    query: str,
    user_access_level: AccessLevel,
    product: Optional[str] = None,
    region: Optional[str] = None,
    top_k: int = 5,
    reference_date: Optional[Union[str, date]] = None,
    *,
    _faiss_store=None,  # test-only injection; bypasses real FAISS load
) -> List[RetrievedChunk]:
    """
    Retrieve the most relevant document chunks for *query*.

    Phase 1 filters (access, product, region) and Phase 2 filters (date,
    conflict resolution) are all applied before returning results.

    Parameters
    ----------
    query:
        Natural-language query from the caller.
    user_access_level:
        The caller's access level.
    product:
        Optional product filter.  ``None`` / ``"*"`` = no filter.
    region:
        Optional region filter.  ``None`` / ``"*"`` = no filter.
    top_k:
        Maximum number of chunks to return after all filtering.
    reference_date:
        Date for temporal filtering.
        * ``None``       → current question; uses ``date.today()`` internally.
        * ISO string or ``date`` object → historical / simulated question.
          Pass an explicit date in tests to keep results deterministic.
    _faiss_store:
        **Test-only** injection.  Pass a pre-built FAISS store to bypass the
        real ``get_active_faiss_path()`` + ``FAISS.load_local()`` call.

    Returns
    -------
    List[RetrievedChunk]
        Retrieved and fully filtered chunks (access + product + region + date +
        conflict resolution), ordered by relevance, limited to *top_k*.
        Each chunk carries ``metadata``, ``provenance``, and ``selection_reason``.

    Raises
    ------
    FileNotFoundError
        If the active FAISS index cannot be located.
    ValueError
        If the active version file is invalid.
    """
    if not query or not query.strip():
        return []

    # ------------------------------------------------------------------
    # Resolve reference date
    # ------------------------------------------------------------------
    ref_date: date = resolve_reference_date(reference_date, default_today=True)

    # ------------------------------------------------------------------
    # Load FAISS store
    # ------------------------------------------------------------------
    if _faiss_store is not None:
        vectordb = _faiss_store
    else:
        from langchain_community.vectorstores import FAISS  # noqa: PLC0415
        active_path = _get_active_path()
        embeddings = _get_embeddings()
        vectordb = FAISS.load_local(
            active_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    # ------------------------------------------------------------------
    # Similarity search — fetch more than top_k to account for all filters
    # ------------------------------------------------------------------
    fetch_k = max(top_k * 6, 30)
    try:
        docs_and_scores = vectordb.similarity_search_with_score(query, k=fetch_k)
    except Exception:
        raw_docs = vectordb.similarity_search(query, k=fetch_k)
        docs_and_scores = [(d, None) for d in raw_docs]

    all_docs = [d for d, _ in docs_and_scores]
    score_map = {id(d): s for d, s in docs_and_scores}

    # ------------------------------------------------------------------
    # Phase 1: access + product + region filters
    # ------------------------------------------------------------------
    filtered = _apply_filters(all_docs, user_access_level, product, region)

    # ------------------------------------------------------------------
    # Phase 2a: date filter (effective / expiry)
    # ------------------------------------------------------------------
    date_filtered = filter_documents_by_date(filtered, ref_date)

    # ------------------------------------------------------------------
    # Phase 2b: policy conflict resolution
    # ------------------------------------------------------------------
    resolved = resolve_policy_conflicts(date_filtered, reference_date=ref_date)

    if not resolved:
        # No applicable evidence found
        return []

    # ------------------------------------------------------------------
    # Build RetrievedChunk objects (up to top_k)
    # ------------------------------------------------------------------
    results: List[RetrievedChunk] = []
    for rr in resolved[:top_k]:
        doc = rr.doc
        try:
            meta = metadata_from_flat_dict(doc.metadata)
        except Exception:
            continue  # skip malformed metadata

        provenance = build_provenance(meta)
        score = score_map.get(id(doc))

        results.append(
            RetrievedChunk(
                content=doc.page_content,
                score=float(score) if score is not None else None,
                metadata=meta,
                provenance=provenance,
                selection_reason=rr.selection_reason,
            )
        )

    return results
