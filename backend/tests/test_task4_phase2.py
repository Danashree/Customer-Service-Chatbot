"""
Task 4 Phase 2 — Date-Aware Retrieval & Policy Conflict Resolution Tests
==========================================================================
Covers all 24 required test cases from the Phase 2 specification, plus
additional boundary and regression tests.

Test fixtures (per spec section 13)
-------------------------------------
A. future_policy       — effective 2027-01-01, no expiry       (future)
B. expired_policy      — effective 2025-01-01, expiry 2025-12-31 (expired as of today)
C. active_policy_v2    — effective 2026-07-01, no expiry       (currently active)
D. historical_v1       — effective 2026-01-01, expiry 2026-06-30 (historical)
E. historical_v2       — effective 2026-07-01, no expiry       (replaced v1 on 2026-07-01)
F. future_v3           — effective 2027-01-01, no expiry       (future higher version)
G. conflict_old + conflict_new — same product/region, different versions
H. same_policy_us / same_policy_eu — same product, different regions
I. policy_productA / policy_productB — different products, same region

Reference dates used (all explicit — no date.today() dependency in tests)
---------------------------------------------------------------------------
2026-03-15  → historical: v1 active, v2 future
2026-09-28  → current:    v2 active, v1 expired, v3 future
2027-02-01  → future:     v3 active
"""
from __future__ import annotations

import hashlib
import os
import sys
import uuid
from datetime import date
from typing import List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BACKEND_DIR, ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from rag.models import AccessLevel, DocumentMetadata, DocumentType, RetrievedChunk
from rag.metadata import metadata_to_flat_dict
from rag.date_filter import (
    is_document_active_on,
    filter_documents_by_date,
    resolve_reference_date,
)
from rag.conflict_resolver import (
    ResolvedResult,
    resolve_policy_conflicts,
    _parse_version,
    _select_winning_document_id,
)
from rag.retriever import (
    _apply_filters,
    retrieve_knowledge,
)
from rag.provenance import build_provenance


# ===========================================================================
# Shared fixture helpers
# ===========================================================================

def _make_meta(
    document_type: str = DocumentType.POLICY,
    product: str = "python_bootcamp",
    region: str = "global",
    access_level: AccessLevel = AccessLevel.public,
    document_version: str = "1.0",
    effective_date: str = "2026-01-01",
    expiry_date: str | None = None,
    document_id: str | None = None,
    source_file: str = "knowledge_base/course_policies.pdf",
) -> DocumentMetadata:
    kwargs = dict(
        document_type=document_type,
        product=product,
        region=region,
        access_level=access_level,
        document_version=document_version,
        effective_date=effective_date,
        expiry_date=expiry_date,
        source_file=source_file,
    )
    if document_id:
        kwargs["document_id"] = document_id
    return DocumentMetadata(**kwargs)


def _make_doc(text: str, meta: DocumentMetadata):
    """Wrap text + DocumentMetadata into a LangChain Document."""
    from langchain_core.documents import Document
    return Document(page_content=text, metadata=metadata_to_flat_dict(meta))


def _build_mock_faiss(pairs: list):
    """
    Build a lightweight mock FAISS store from (text, DocumentMetadata) pairs.
    similarity_search_with_score returns them in order with dummy scores.
    """
    from langchain_core.documents import Document
    docs = [
        Document(page_content=text, metadata=metadata_to_flat_dict(meta))
        for text, meta in pairs
    ]
    store = MagicMock()
    store.similarity_search_with_score.return_value = [
        (doc, 0.95 - i * 0.05) for i, doc in enumerate(docs)
    ]
    return store


# ---------------------------------------------------------------------------
# Canonical test fixtures (defined once, reused across tests)
# ---------------------------------------------------------------------------

FIXED_DOC_ID_V1 = str(uuid.uuid4())
FIXED_DOC_ID_V2 = str(uuid.uuid4())
FIXED_DOC_ID_V3 = str(uuid.uuid4())

# A  — future policy
META_FUTURE = _make_meta(
    document_version="3.0",
    effective_date="2027-01-01",
    expiry_date=None,
    document_id=FIXED_DOC_ID_V3,
)

# B  — expired policy (expired 2025-12-31)
META_EXPIRED = _make_meta(
    document_version="0.9",
    effective_date="2025-01-01",
    expiry_date="2025-12-31",
)

# C  — currently active policy v2
META_ACTIVE_V2 = _make_meta(
    document_version="2.0",
    effective_date="2026-07-01",
    expiry_date=None,
    document_id=FIXED_DOC_ID_V2,
)

# D  — historical v1 (expired 2026-06-30)
META_HIST_V1 = _make_meta(
    document_version="1.0",
    effective_date="2026-01-01",
    expiry_date="2026-06-30",
    document_id=FIXED_DOC_ID_V1,
)

# E  — historical v2 (same as C for retrieval tests)
META_HIST_V2 = META_ACTIVE_V2

# F  — future higher-version policy (same as A)
META_FUTURE_V3 = META_FUTURE

# G  — conflicting policy versions (same product/region)
META_CONFLICT_OLD = _make_meta(
    document_version="1.0",
    effective_date="2026-01-01",
    expiry_date="2026-06-30",
    product="data_science_pro",
    document_id=str(uuid.uuid4()),
)
META_CONFLICT_NEW = _make_meta(
    document_version="2.0",
    effective_date="2026-07-01",
    expiry_date=None,
    product="data_science_pro",
    document_id=str(uuid.uuid4()),
)

# H  — same policy topic, different regions
META_US = _make_meta(region="us", document_version="1.0", effective_date="2026-01-01")
META_EU = _make_meta(region="eu", document_version="1.0", effective_date="2026-01-01")

# I  — different products
META_PRODUCT_A = _make_meta(product="product_a", document_version="1.0", effective_date="2026-01-01")
META_PRODUCT_B = _make_meta(product="product_b", document_version="1.0", effective_date="2026-01-01")

# FAQ / how_to — non-policy docs (should always be preserved)
META_FAQ = _make_meta(
    document_type=DocumentType.FAQ,
    document_version="1.0",
    effective_date="2026-01-01",
)
META_HOWTO = _make_meta(
    document_type=DocumentType.HOW_TO,
    document_version="1.0",
    effective_date="2026-01-01",
)

# Reference dates (all explicit)
DATE_HISTORICAL = date(2026, 3, 15)
DATE_CURRENT = date(2026, 9, 28)
DATE_FUTURE = date(2027, 2, 1)


# ===========================================================================
# T01 — current-date retrieval: only active docs returned
# ===========================================================================
def test_t01_current_date_retrieval_returns_active():
    """retrieve_knowledge with current reference_date returns only active docs."""
    store = _build_mock_faiss([
        ("active v2 content", META_ACTIVE_V2),
        ("future v3 content", META_FUTURE),
        ("expired content", META_EXPIRED),
    ])
    results = retrieve_knowledge(
        "refund policy",
        AccessLevel.admin,
        reference_date=DATE_CURRENT,
        top_k=10,
        _faiss_store=store,
    )
    contents = {r.content for r in results}
    assert "active v2 content" in contents
    assert "future v3 content" not in contents
    assert "expired content" not in contents


# ===========================================================================
# T02 — future document explicitly excluded
# ===========================================================================
def test_t02_future_document_excluded():
    """A document with effective_date > reference_date must be excluded."""
    future_meta = _make_meta(effective_date="2027-01-01", expiry_date=None)
    assert not is_document_active_on(
        metadata_to_flat_dict(future_meta), date(2026, 9, 28)
    )


# ===========================================================================
# T03 — expired document explicitly excluded
# ===========================================================================
def test_t03_expired_document_excluded():
    """A document with expiry_date < reference_date must be excluded."""
    expired_meta = _make_meta(effective_date="2025-01-01", expiry_date="2025-12-31")
    assert not is_document_active_on(
        metadata_to_flat_dict(expired_meta), date(2026, 9, 28)
    )


# ===========================================================================
# T04 — active document included
# ===========================================================================
def test_t04_active_document_included():
    """A document within [effective, expiry] must be included."""
    active_meta = _make_meta(effective_date="2026-01-01", expiry_date="2026-12-31")
    assert is_document_active_on(
        metadata_to_flat_dict(active_meta), date(2026, 9, 28)
    )


# ===========================================================================
# T05 — effective-date boundary: effective_date == reference_date → active
# ===========================================================================
def test_t05_effective_date_boundary_inclusive():
    """Boundary: effective_date == reference_date → document IS active."""
    meta = _make_meta(effective_date="2026-07-01", expiry_date=None)
    assert is_document_active_on(
        metadata_to_flat_dict(meta), date(2026, 7, 1)
    )


# ===========================================================================
# T06 — expiry-date boundary: expiry_date == reference_date → still active
# ===========================================================================
def test_t06_expiry_date_boundary_inclusive():
    """Boundary: expiry_date == reference_date → document IS still active."""
    meta = _make_meta(effective_date="2026-01-01", expiry_date="2026-06-30")
    assert is_document_active_on(
        metadata_to_flat_dict(meta), date(2026, 6, 30)
    )


# ===========================================================================
# T07 — null expiry: document never expires
# ===========================================================================
def test_t07_null_expiry_document_never_expires():
    """A document with expiry_date=None should be active for any future date."""
    meta = _make_meta(effective_date="2026-01-01", expiry_date=None)
    flat = metadata_to_flat_dict(meta)
    assert is_document_active_on(flat, date(2030, 1, 1))
    assert is_document_active_on(flat, date(2099, 12, 31))


# ===========================================================================
# T08 — historical-date retrieval: only v1 active on 2026-03-15
# ===========================================================================
def test_t08_historical_date_retrieval_selects_v1():
    """Historical query on 2026-03-15: v1 should be active, v2 should not."""
    store = _build_mock_faiss([
        ("v1 refund policy", META_HIST_V1),
        ("v2 refund policy", META_HIST_V2),
    ])
    results = retrieve_knowledge(
        "refund policy",
        AccessLevel.admin,
        reference_date=DATE_HISTORICAL,
        top_k=10,
        _faiss_store=store,
    )
    contents = {r.content for r in results}
    assert "v1 refund policy" in contents
    assert "v2 refund policy" not in contents


# ===========================================================================
# T09 — historical policy v1 selected (via date_filter directly)
# ===========================================================================
def test_t09_historical_v1_active_on_2026_03_15():
    """is_document_active_on: v1 (eff 2026-01-01, exp 2026-06-30) is active on 2026-03-15."""
    flat = metadata_to_flat_dict(META_HIST_V1)
    assert is_document_active_on(flat, DATE_HISTORICAL)


# ===========================================================================
# T10 — later policy v2 selected after its effective date
# ===========================================================================
def test_t10_v2_active_after_effective_date():
    """is_document_active_on: v2 (eff 2026-07-01, no expiry) is active on 2026-09-28."""
    flat = metadata_to_flat_dict(META_HIST_V2)
    assert is_document_active_on(flat, DATE_CURRENT)


# ===========================================================================
# T11 — future higher-version policy ignored for current date
# ===========================================================================
def test_t11_future_higher_version_ignored():
    """v3 (eff 2027-01-01) must NOT be retrieved on reference_date=2026-09-28."""
    store = _build_mock_faiss([
        ("v2 content", META_ACTIVE_V2),
        ("v3 future content", META_FUTURE_V3),
    ])
    results = retrieve_knowledge(
        "refund policy",
        AccessLevel.admin,
        reference_date=DATE_CURRENT,
        top_k=10,
        _faiss_store=store,
    )
    contents = {r.content for r in results}
    assert "v2 content" in contents
    assert "v3 future content" not in contents


# ===========================================================================
# T12 — latest applicable policy selection (v1, v2, v3 — only v2 for 2026-09-28)
# ===========================================================================
def test_t12_latest_applicable_policy_selection():
    """Given v1 (expired), v2 (active), v3 (future): on 2026-09-28 → v2 wins."""
    store = _build_mock_faiss([
        ("v1 expired", META_HIST_V1),
        ("v2 active", META_ACTIVE_V2),
        ("v3 future", META_FUTURE_V3),
    ])
    results = retrieve_knowledge(
        "refund policy",
        AccessLevel.admin,
        reference_date=DATE_CURRENT,
        top_k=10,
        _faiss_store=store,
    )
    contents = {r.content for r in results}
    assert "v2 active" in contents
    assert "v1 expired" not in contents
    assert "v3 future" not in contents


# ===========================================================================
# T13 — same effective date: deterministic tie-break via document_version
# ===========================================================================
def test_t13_same_effective_date_version_tiebreak():
    """When two docs share the same effective_date, higher version wins."""
    meta_low = _make_meta(
        document_version="1.0",
        effective_date="2026-07-01",
        expiry_date=None,
        document_id=str(uuid.uuid4()),
    )
    meta_high = _make_meta(
        document_version="2.0",
        effective_date="2026-07-01",
        expiry_date=None,
        document_id=str(uuid.uuid4()),
    )

    id_to_meta = {
        meta_low.document_id: metadata_to_flat_dict(meta_low),
        meta_high.document_id: metadata_to_flat_dict(meta_high),
    }
    winner = _select_winning_document_id(id_to_meta)
    assert winner == meta_high.document_id


# ===========================================================================
# T14 — product filtering preserved in Phase 2
# ===========================================================================
def test_t14_product_filtering_preserved():
    """Product filter must still work after Phase 2 date/conflict integration."""
    meta_py = _make_meta(product="python_bootcamp", effective_date="2026-01-01")
    meta_ds = _make_meta(product="data_science_pro", effective_date="2026-01-01")

    store = _build_mock_faiss([
        ("python content", meta_py),
        ("data science content", meta_ds),
    ])
    results = retrieve_knowledge(
        "course policy",
        AccessLevel.admin,
        product="python_bootcamp",
        reference_date=DATE_CURRENT,
        top_k=10,
        _faiss_store=store,
    )
    contents = {r.content for r in results}
    assert "python content" in contents
    assert "data science content" not in contents


# ===========================================================================
# T15 — region filtering preserved in Phase 2
# ===========================================================================
def test_t15_region_filtering_preserved():
    """Region filter must still work after Phase 2 date/conflict integration."""
    meta_us = _make_meta(region="us", effective_date="2026-01-01")
    meta_eu = _make_meta(region="eu", effective_date="2026-01-01")

    store = _build_mock_faiss([
        ("US content", meta_us),
        ("EU content", meta_eu),
    ])
    results = retrieve_knowledge(
        "refund policy",
        AccessLevel.admin,
        region="us",
        reference_date=DATE_CURRENT,
        top_k=10,
        _faiss_store=store,
    )
    contents = {r.content for r in results}
    assert "US content" in contents
    assert "EU content" not in contents


# ===========================================================================
# T16 — access control preserved in Phase 2
# ===========================================================================
def test_t16_access_control_preserved():
    """Admin-only doc must not be returned to a public-level caller."""
    meta_pub = _make_meta(access_level=AccessLevel.public, effective_date="2026-01-01")
    meta_admin = _make_meta(access_level=AccessLevel.admin, effective_date="2026-01-01")

    store = _build_mock_faiss([
        ("public content", meta_pub),
        ("admin-only content", meta_admin),
    ])
    results = retrieve_knowledge(
        "policy",
        AccessLevel.public,
        reference_date=DATE_CURRENT,
        top_k=10,
        _faiss_store=store,
    )
    contents = {r.content for r in results}
    assert "public content" in contents
    assert "admin-only content" not in contents


# ===========================================================================
# T17 — no applicable evidence handled safely (empty list, no exception)
# ===========================================================================
def test_t17_no_applicable_evidence_returns_empty():
    """When all docs are future or expired, retrieve_knowledge returns []."""
    store = _build_mock_faiss([
        ("future doc", META_FUTURE),
        ("expired doc", META_EXPIRED),
    ])
    results = retrieve_knowledge(
        "refund policy",
        AccessLevel.admin,
        reference_date=DATE_CURRENT,
        top_k=10,
        _faiss_store=store,
    )
    assert results == []


# ===========================================================================
# T18 — multiple non-conflicting documents preserved (FAQ + policy + how_to)
# ===========================================================================
def test_t18_multiple_non_conflicting_docs_preserved():
    """FAQ + how_to + policy (all different types) should all be returned."""
    meta_faq = _make_meta(document_type=DocumentType.FAQ, effective_date="2026-01-01")
    meta_howto = _make_meta(document_type=DocumentType.HOW_TO, effective_date="2026-01-01")
    meta_policy = _make_meta(document_type=DocumentType.POLICY, effective_date="2026-01-01")

    store = _build_mock_faiss([
        ("faq content", meta_faq),
        ("how to content", meta_howto),
        ("policy content", meta_policy),
    ])
    results = retrieve_knowledge(
        "course help",
        AccessLevel.admin,
        reference_date=DATE_CURRENT,
        top_k=10,
        _faiss_store=store,
    )
    contents = {r.content for r in results}
    assert "faq content" in contents
    assert "how to content" in contents
    assert "policy content" in contents


# ===========================================================================
# T19 — conflicting policy versions resolved correctly
# ===========================================================================
def test_t19_conflicting_policy_versions_resolved():
    """When old (expired) and new (active) policy versions coexist, only new wins."""
    store = _build_mock_faiss([
        ("conflict old content", META_CONFLICT_OLD),
        ("conflict new content", META_CONFLICT_NEW),
    ])
    results = retrieve_knowledge(
        "data science refund policy",
        AccessLevel.admin,
        product="data_science_pro",
        reference_date=DATE_CURRENT,
        top_k=10,
        _faiss_store=store,
    )
    # OLD was effective 2026-01-01 → 2026-06-30; expired as of 2026-09-28
    # NEW was effective 2026-07-01; still active
    contents = {r.content for r in results}
    assert "conflict new content" in contents
    assert "conflict old content" not in contents


# ===========================================================================
# T20 — simulated date works without touching system clock
# ===========================================================================
def test_t20_simulated_date_deterministic():
    """
    Passing reference_date="2026-03-15" as a string must produce deterministic
    results without any mock of date.today() or system clock.
    """
    store = _build_mock_faiss([
        ("v1 content", META_HIST_V1),
        ("v2 content", META_HIST_V2),
    ])
    results = retrieve_knowledge(
        "refund policy",
        AccessLevel.admin,
        reference_date="2026-03-15",   # ← string, not date object
        top_k=10,
        _faiss_store=store,
    )
    contents = {r.content for r in results}
    assert "v1 content" in contents
    assert "v2 content" not in contents


# ===========================================================================
# T21 — metadata remains attached to every returned chunk
# ===========================================================================
def test_t21_metadata_attached_to_results():
    """Every RetrievedChunk must carry a populated DocumentMetadata."""
    meta = _make_meta(effective_date="2026-01-01")
    store = _build_mock_faiss([("some content", meta)])
    results = retrieve_knowledge(
        "policy",
        AccessLevel.admin,
        reference_date=DATE_CURRENT,
        top_k=5,
        _faiss_store=store,
    )
    assert len(results) == 1
    chunk = results[0]
    assert chunk.metadata is not None
    assert chunk.metadata.document_version == meta.document_version
    assert chunk.metadata.effective_date == date(2026, 1, 1)


# ===========================================================================
# T22 — provenance remains attached to every returned chunk
# ===========================================================================
def test_t22_provenance_attached_to_results():
    """Every RetrievedChunk must carry a populated ProvenanceRecord with citation."""
    meta = DocumentMetadata(
        document_type=DocumentType.POLICY,
        product="python_bootcamp",
        region="global",
        access_level=AccessLevel.public,
        document_version="1.0",
        effective_date="2026-01-01",
        expiry_date=None,
        source_file="knowledge_base/course_policies.pdf",
        source_page=3,
    )
    store = _build_mock_faiss([("content", meta)])
    results = retrieve_knowledge(
        "policy",
        AccessLevel.admin,
        reference_date=DATE_CURRENT,
        top_k=5,
        _faiss_store=store,
    )
    assert len(results) == 1
    chunk = results[0]
    assert chunk.provenance is not None
    assert chunk.provenance.source_page == 3
    citation = chunk.citation()
    assert "POLICY" in citation.upper()
    assert "course_policies.pdf" in citation


# ===========================================================================
# T23 — active Task 1 FAISS integration preserved
# ===========================================================================
def test_t23_active_faiss_integration_preserved():
    """
    Verify get_active_faiss_path() still resolves to a valid v1 directory.
    Does NOT load embeddings model — only checks file system state.
    """
    import json
    versions_dir = os.path.join(_BACKEND_DIR, "versions")
    active_file = os.path.join(versions_dir, "active_version.json")
    assert os.path.exists(active_file)

    with open(active_file) as f:
        active_data = json.load(f)
    active_version = active_data.get("active_version")
    assert active_version

    target_dir = os.path.join(versions_dir, active_version)
    assert os.path.exists(os.path.join(target_dir, "index.faiss"))
    assert os.path.exists(os.path.join(target_dir, "index.pkl"))


# ===========================================================================
# T24 — Phase 1 tests remain passing (smoke: models, metadata round-trip)
# ===========================================================================
def test_t24_phase1_smoke_still_passes():
    """Quick Phase 1 smoke: DocumentMetadata creation and round-trip."""
    from rag.metadata import metadata_from_flat_dict

    meta = _make_meta()
    flat = metadata_to_flat_dict(meta)
    restored = metadata_from_flat_dict(flat)
    assert restored.document_version == meta.document_version
    assert restored.access_level == meta.access_level
    assert restored.product == meta.product


# ===========================================================================
# T25 — selection_reason populated after conflict resolution
# ===========================================================================
def test_t25_selection_reason_populated():
    """RetrievedChunk.selection_reason must be set (not None) after Phase 2."""
    meta = _make_meta(effective_date="2026-01-01")
    store = _build_mock_faiss([("content with reason", meta)])
    results = retrieve_knowledge(
        "policy",
        AccessLevel.admin,
        reference_date=DATE_CURRENT,
        top_k=5,
        _faiss_store=store,
    )
    assert len(results) == 1
    assert results[0].selection_reason is not None
    assert len(results[0].selection_reason) > 0


# ===========================================================================
# T26 — filter_documents_by_date filters a mixed list correctly
# ===========================================================================
def test_t26_filter_documents_by_date_mixed():
    """filter_documents_by_date should keep only date-active docs."""
    docs = [
        _make_doc("active", _make_meta(effective_date="2026-01-01", expiry_date=None)),
        _make_doc("future", _make_meta(effective_date="2027-01-01", expiry_date=None)),
        _make_doc("expired", _make_meta(effective_date="2025-01-01", expiry_date="2025-12-31")),
        _make_doc("boundary_start", _make_meta(effective_date="2026-09-28", expiry_date=None)),
    ]
    filtered = filter_documents_by_date(docs, date(2026, 9, 28))
    contents = {d.page_content for d in filtered}
    assert "active" in contents
    assert "boundary_start" in contents
    assert "future" not in contents
    assert "expired" not in contents


# ===========================================================================
# T27 — resolve_reference_date coercion
# ===========================================================================
def test_t27_resolve_reference_date_coercion():
    """resolve_reference_date must accept str, date, and None."""
    assert resolve_reference_date("2026-03-15") == date(2026, 3, 15)
    assert resolve_reference_date(date(2026, 3, 15)) == date(2026, 3, 15)

    today_result = resolve_reference_date(None, default_today=True)
    assert isinstance(today_result, date)

    none_result = resolve_reference_date(None, default_today=False)
    assert none_result is None


# ===========================================================================
# T28 — _parse_version helper for tie-breaking
# ===========================================================================
def test_t28_parse_version_helper():
    """_parse_version should produce comparable tuples for semver strings."""
    assert _parse_version("2.0") > _parse_version("1.0")
    assert _parse_version("1.3") > _parse_version("1.2")
    assert _parse_version("10.0") > _parse_version("9.0")
    assert _parse_version("2.0") == _parse_version("2.0")
    assert _parse_version("invalid") == (0,)


# ===========================================================================
# T29 — resolve_policy_conflicts: two conflicting policy groups
# ===========================================================================
def test_t29_conflict_resolution_picks_latest():
    """
    resolve_policy_conflicts: when two policy docs share product+region,
    the one with the later effective_date wins.
    """
    old_meta = _make_meta(
        document_version="1.0",
        effective_date="2026-01-01",
        expiry_date="2026-06-30",
        document_id=str(uuid.uuid4()),
    )
    new_meta = _make_meta(
        document_version="2.0",
        effective_date="2026-07-01",
        expiry_date=None,
        document_id=str(uuid.uuid4()),
    )
    docs = [
        _make_doc("old policy content", old_meta),
        _make_doc("new policy content", new_meta),
    ]
    resolved = resolve_policy_conflicts(docs, reference_date=DATE_CURRENT)
    contents = {r.doc.page_content for r in resolved}
    assert "new policy content" in contents
    assert "old policy content" not in contents


# ===========================================================================
# T30 — different-region policies NOT merged into one conflict group
# ===========================================================================
def test_t30_different_region_policies_not_merged():
    """
    Policies with the same product but different regions are independent.
    Both should survive conflict resolution.
    """
    meta_us = _make_meta(
        region="us",
        product="python_bootcamp",
        effective_date="2026-01-01",
        document_id=str(uuid.uuid4()),
    )
    meta_eu = _make_meta(
        region="eu",
        product="python_bootcamp",
        effective_date="2026-01-01",
        document_id=str(uuid.uuid4()),
    )
    docs = [
        _make_doc("US policy", meta_us),
        _make_doc("EU policy", meta_eu),
    ]
    resolved = resolve_policy_conflicts(docs, reference_date=DATE_CURRENT)
    contents = {r.doc.page_content for r in resolved}
    assert "US policy" in contents
    assert "EU policy" in contents


# ===========================================================================
# T31 — non-policy docs always preserved through conflict resolution
# ===========================================================================
def test_t31_non_policy_docs_always_preserved():
    """FAQ and how_to docs must pass through resolve_policy_conflicts untouched."""
    from langchain_core.documents import Document

    faq_doc = _make_doc("faq content", META_FAQ)
    howto_doc = _make_doc("howto content", META_HOWTO)
    policy_doc = _make_doc("policy content", _make_meta(effective_date="2026-01-01"))

    resolved = resolve_policy_conflicts(
        [faq_doc, howto_doc, policy_doc],
        reference_date=DATE_CURRENT,
    )
    contents = {r.doc.page_content for r in resolved}
    assert "faq content" in contents
    assert "howto content" in contents
    assert "policy content" in contents


# ===========================================================================
# T32 — dataset SHA-256 unchanged
# ===========================================================================
EXPECTED_DATASET_SHA256 = "930649d927881235ecbd3b53b29de92ddf8dbeca084657774335630eb9361b9a"


def test_t32_dataset_csv_unchanged():
    dataset_path = os.path.join(_PROJECT_ROOT, "dataset", "dataset.csv")
    assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"
    sha = hashlib.sha256()
    with open(dataset_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    actual = sha.hexdigest()
    assert actual == EXPECTED_DATASET_SHA256, (
        f"dataset.csv was modified!\n"
        f"Expected: {EXPECTED_DATASET_SHA256}\n"
        f"Actual:   {actual}"
    )
