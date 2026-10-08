"""
Task 4 Phase 1 — RAG Data Models
=================================
Pydantic models for document metadata, access control, and retrieved chunks.
"""
from __future__ import annotations

import uuid
from datetime import date
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Access-level hierarchy  (integer comparison is intentional: N ≥ required)
# ---------------------------------------------------------------------------

class AccessLevel(IntEnum):
    """
    Tiered access hierarchy.
    A caller at level N may retrieve documents whose access_level ≤ N.
    """
    public = 0
    customer = 1
    agent = 2
    admin = 3

    @classmethod
    def from_str(cls, value: str) -> "AccessLevel":
        """Case-insensitive string → AccessLevel."""
        try:
            return cls[value.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown access level '{value}'. "
                f"Valid values: {[m.name for m in cls]}"
            )


# ---------------------------------------------------------------------------
# Document type
# ---------------------------------------------------------------------------

class DocumentType(str):
    """
    Allowed document type labels (open string with well-known constants).
    Use the class attributes as canonical values.
    """
    FAQ = "faq"
    POLICY = "policy"
    HOW_TO = "how_to"
    TROUBLESHOOTING = "troubleshooting"
    GENERAL = "general"

    _KNOWN = {FAQ, POLICY, HOW_TO, TROUBLESHOOTING, GENERAL}

    @classmethod
    def normalize(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return normalized  # accept any string; warn only if unknown


# ---------------------------------------------------------------------------
# Core metadata model
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """
    Rich metadata attached to every document chunk in the knowledge base.
    All fields survive chunking and are persisted in the FAISS metadata dict.
    """

    # Identity
    document_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the source document.",
    )
    document_version: str = Field(
        ...,
        description="Semantic version of the document (e.g. '1.0', '2.3').",
    )

    # Classification
    document_type: str = Field(
        default=DocumentType.GENERAL,
        description="Type of document: faq, policy, how_to, troubleshooting, general.",
    )
    product: str = Field(
        ...,
        description="Product or course the document covers (e.g. 'python_bootcamp').",
    )
    region: str = Field(
        default="global",
        description="Geographic / regulatory region (e.g. 'global', 'us', 'eu').",
    )

    # Authorization
    access_level: AccessLevel = Field(
        ...,
        description="Minimum caller access level required to read this document.",
    )

    # Temporal validity
    effective_date: date = Field(
        ...,
        description="Date from which this document version is active (ISO 8601).",
    )
    expiry_date: Optional[date] = Field(
        default=None,
        description="Date after which this document version expires. None = no expiry.",
    )

    # Provenance / chunk tracking
    source_file: str = Field(
        ...,
        description="Relative path to the originating file (e.g. 'knowledge_base/course_faqs.pdf').",
    )
    source_page: Optional[int] = Field(
        default=None,
        description="Page number within the source PDF (1-indexed). None for non-paged sources.",
    )
    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this specific chunk (set during chunking).",
    )

    # ---------------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------------

    @field_validator("effective_date", "expiry_date", mode="before")
    @classmethod
    def parse_iso_date(cls, v: Any) -> Optional[date]:
        """Accept ISO-8601 strings like '2024-01-15' or date objects."""
        if v is None:
            return None
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                raise ValueError(
                    f"Invalid ISO-8601 date '{v}'. Expected format: YYYY-MM-DD."
                )
        raise TypeError(f"Date must be a str or date, got {type(v).__name__}.")

    @field_validator("document_version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("document_version must not be empty.")
        return v

    @field_validator("product")
    @classmethod
    def validate_product(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("product must not be empty.")
        return v.lower()

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("document_type", mode="before")
    @classmethod
    def normalize_doc_type(cls, v: str) -> str:
        return DocumentType.normalize(str(v))

    @field_validator("access_level", mode="before")
    @classmethod
    def coerce_access_level(cls, v: Any) -> AccessLevel:
        if isinstance(v, AccessLevel):
            return v
        if isinstance(v, int):
            return AccessLevel(v)
        if isinstance(v, str):
            return AccessLevel.from_str(v)
        raise TypeError(f"Cannot coerce {type(v)} to AccessLevel.")

    @model_validator(mode="after")
    def validate_date_range(self) -> "DocumentMetadata":
        if self.expiry_date is not None and self.expiry_date < self.effective_date:
            raise ValueError(
                f"expiry_date ({self.expiry_date}) must not be before "
                f"effective_date ({self.effective_date})."
            )
        return self

    # ---------------------------------------------------------------------------
    # Serialization helpers
    # ---------------------------------------------------------------------------

    def to_faiss_metadata(self) -> Dict[str, Any]:
        """
        Flatten metadata into a dict of JSON-serializable primitives
        for storage as LangChain Document.metadata.
        """
        return {
            "document_id": self.document_id,
            "document_version": self.document_version,
            "document_type": self.document_type,
            "product": self.product,
            "region": self.region,
            "access_level": int(self.access_level),
            "effective_date": self.effective_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "source_file": self.source_file,
            "source_page": self.source_page,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_faiss_metadata(cls, meta: Dict[str, Any]) -> "DocumentMetadata":
        """Reconstruct a DocumentMetadata from a FAISS metadata dict."""
        return cls(**meta)


# ---------------------------------------------------------------------------
# Provenance record
# ---------------------------------------------------------------------------

class ProvenanceRecord(BaseModel):
    """Tracks exactly where a retrieved chunk came from."""

    chunk_id: str
    document_id: str
    document_version: str
    source_file: str
    source_page: Optional[int]
    document_type: str
    product: str
    region: str
    access_level: AccessLevel
    effective_date: date
    expiry_date: Optional[date]

    @field_validator("effective_date", "expiry_date", mode="before")
    @classmethod
    def parse_date(cls, v: Any) -> Optional[date]:
        if v is None:
            return None
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v))

    @field_validator("access_level", mode="before")
    @classmethod
    def coerce_level(cls, v: Any) -> AccessLevel:
        if isinstance(v, AccessLevel):
            return v
        if isinstance(v, int):
            return AccessLevel(v)
        return AccessLevel.from_str(str(v))

    def citation(self) -> str:
        """Human-readable citation string."""
        page_part = f", page {self.source_page}" if self.source_page else ""
        return (
            f"[{self.document_type.upper()}] {self.source_file}{page_part} "
            f"(v{self.document_version}, {self.access_level.name})"
        )


# ---------------------------------------------------------------------------
# Retrieved chunk (returned by retriever)
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """A document chunk returned by the RAG retriever with full provenance."""

    content: str = Field(..., description="Text content of the chunk.")
    score: Optional[float] = Field(
        default=None,
        description="Similarity score from FAISS (higher = more similar). None if unavailable.",
    )
    metadata: DocumentMetadata = Field(..., description="Full structured metadata.")
    provenance: ProvenanceRecord = Field(..., description="Provenance / citation record.")
    selection_reason: Optional[str] = Field(
        default=None,
        description=(
            "Phase 2: human-readable explanation of why this chunk was selected / "
            "survived conflict resolution.  None when conflict resolution was not applied."
        ),
    )

    def citation(self) -> str:
        """Delegates to provenance.citation()."""
        return self.provenance.citation()


# ---------------------------------------------------------------------------
# Phase 3 Models: Evidence Sufficiency, Claims, Injections & Answer
# ---------------------------------------------------------------------------

class EvidenceSufficiency(str, Enum):
    SUPPORTED = "supported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class InjectionScanResult(BaseModel):
    """Machine-readable document prompt-injection scan result."""
    is_injection: bool = Field(..., description="Whether prompt injection patterns were detected.")
    reason: Optional[str] = Field(default=None, description="Category of the detected injection.")
    risk_level: str = Field(default="none", description="Risk level: none, medium, high.")
    matched_patterns: List[str] = Field(default_factory=list, description="Specific matched directive text.")
    categories: List[str] = Field(default_factory=list, description="Categories matched.")
    sanitized_content: Optional[str] = Field(default=None, description="Clean non-malicious content if salvageable.")


class ClaimVerificationResult(BaseModel):
    """Grounding evaluation result for factual claims against retrieved evidence."""
    status: ClaimStatus = Field(..., description="Overall claim verification status.")
    is_grounded: bool = Field(..., description="True if all factual claims are supported by evidence.")
    claims: List[str] = Field(default_factory=list, description="Extracted factual claims from the answer.")
    supported_claims: List[str] = Field(default_factory=list, description="Claims substantiated by retrieved evidence.")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims without evidence support.")
    hallucinated_values: List[str] = Field(default_factory=list, description="Extracted values (prices, dates, deadlines) absent from evidence.")


class CitationValidationResult(BaseModel):
    """Result of validating citations in an answer against authorized retrieved chunks."""
    is_valid: bool = Field(..., description="True if citations exist and all map to authorized evidence.")
    cited_sources: List[str] = Field(default_factory=list, description="Sources cited in the answer.")
    valid_sources: List[str] = Field(default_factory=list, description="Cited sources that match authorized evidence.")
    invalid_sources: List[str] = Field(default_factory=list, description="Cited sources not present in authorized evidence.")
    errors: List[str] = Field(default_factory=list, description="Descriptions of validation failures.")


class RAGAnswer(BaseModel):
    """Complete answer object from RAG knowledge assistant."""
    query: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    sufficiency: EvidenceSufficiency
    is_refusal: bool = False
    refusal_reason: Optional[str] = None
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    claim_verification: Optional[ClaimVerificationResult] = None
    citation_validation: Optional[CitationValidationResult] = None
    injection_scans: List[InjectionScanResult] = Field(default_factory=list)


