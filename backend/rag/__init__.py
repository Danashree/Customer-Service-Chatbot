"""
Task 4 — RAG Knowledge Assistant (Phase 1, 2, 3)
=================================================
Package exposing the core RAG components.

Phase 1
-------
  models          — Pydantic data models (DocumentMetadata, AccessLevel, …)
  metadata        — Metadata normalization & ISO date helpers
  document_loader — Loads KB PDFs and attaches rich metadata
  chunker         — Metadata-preserving text splitter
  retriever       — Authorization-aware + date-aware FAISS retriever
  provenance      — ProvenanceRecord builder

Phase 2 additions
-----------------
  date_filter       — Effective/expiry date filtering helpers
  conflict_resolver — Policy conflict resolution (latest applicable policy)

Phase 3 additions
-----------------
  injection_detector — Document-level prompt-injection detection & sanitization
  citations          — Mandatory citation formatting and validation
  grounding          — Unsupported claim detection, hallucinated value checks & evidence sufficiency
  answer_generator   — End-to-end evidence-grounded answer generation with safety guardrails
"""

from .models import (
    AccessLevel,
    DocumentType,
    DocumentMetadata,
    ProvenanceRecord,
    RetrievedChunk,
    EvidenceSufficiency,
    ClaimStatus,
    InjectionScanResult,
    ClaimVerificationResult,
    CitationValidationResult,
    RAGAnswer,
)
from .retriever import retrieve_knowledge
from .provenance import build_provenance
from .date_filter import (
    is_document_active_on,
    filter_documents_by_date,
    resolve_reference_date,
)
from .conflict_resolver import resolve_policy_conflicts, ResolvedResult
from .injection_detector import scan_document_injection, normalize_text
from .citations import (
    format_citation,
    format_citations,
    parse_citations,
    validate_citations,
)
from .grounding import (
    evaluate_evidence_sufficiency,
    extract_claims,
    extract_factual_values,
    verify_claims,
)
from .answer_generator import generate_rag_answer

__all__ = [
    # Phase 1
    "AccessLevel",
    "DocumentType",
    "DocumentMetadata",
    "ProvenanceRecord",
    "RetrievedChunk",
    "retrieve_knowledge",
    "build_provenance",
    # Phase 2
    "is_document_active_on",
    "filter_documents_by_date",
    "resolve_reference_date",
    "resolve_policy_conflicts",
    "ResolvedResult",
    # Phase 3
    "EvidenceSufficiency",
    "ClaimStatus",
    "InjectionScanResult",
    "ClaimVerificationResult",
    "CitationValidationResult",
    "RAGAnswer",
    "scan_document_injection",
    "normalize_text",
    "format_citation",
    "format_citations",
    "parse_citations",
    "validate_citations",
    "evaluate_evidence_sufficiency",
    "extract_claims",
    "extract_factual_values",
    "verify_claims",
    "generate_rag_answer",
]
