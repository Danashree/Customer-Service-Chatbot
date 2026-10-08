"""
Task 4 Phase 1 — Provenance Builder
=====================================
Builds ``ProvenanceRecord`` objects from ``DocumentMetadata``.
"""
from __future__ import annotations

from .models import DocumentMetadata, ProvenanceRecord


def build_provenance(meta: DocumentMetadata) -> ProvenanceRecord:
    """
    Construct a :class:`ProvenanceRecord` from a :class:`DocumentMetadata` instance.

    Every field is copied directly; no information is lost or fabricated.
    """
    return ProvenanceRecord(
        chunk_id=meta.chunk_id,
        document_id=meta.document_id,
        document_version=meta.document_version,
        source_file=meta.source_file,
        source_page=meta.source_page,
        document_type=meta.document_type,
        product=meta.product,
        region=meta.region,
        access_level=meta.access_level,
        effective_date=meta.effective_date,
        expiry_date=meta.expiry_date,
    )
