"""
Task 4 Phase 3 — Evidence-Grounded Answer Generation & Guardrails
==================================================================
Coordinates the end-to-end safe answer generation pipeline:
  1. Authorization & access-level filtering (Phase 1)
  2. Date-aware validity filtering (Phase 2)
  3. Policy conflict resolution (Phase 2)
  4. Document prompt-injection scan & sanitization (Phase 3)
  5. Evidence sufficiency evaluation (Phase 3)
  6. Answer synthesis strictly from clean evidence (Phase 3)
  7. Factual grounding & unsupported claim verification (Phase 3)
  8. Mandatory source citation validation (Phase 3)

Guarantees:
- If evidence is missing or ambiguous, returns safe refusal / clarification.
- Malicious instructions inside documents are never followed or executed.
- Factual claims without evidence or with hallucinated values are rejected.
- Every factual answer carries validated citations matching retrieved chunks.
"""
from __future__ import annotations

from datetime import date
from typing import Callable, List, Optional, Union

from .models import (
    AccessLevel,
    EvidenceSufficiency,
    InjectionScanResult,
    RAGAnswer,
    RetrievedChunk,
)
from .retriever import retrieve_knowledge
from .injection_detector import scan_document_injection
from .citations import format_citations, validate_citations
from .grounding import evaluate_evidence_sufficiency, verify_claims


def _default_synthesize_answer(query: str, clean_chunks: List[RetrievedChunk]) -> str:
    """
    Deterministic answer synthesizer for testability and non-LLM execution.
    Combines clean evidence text.
    """
    if not clean_chunks:
        return ""
    # Use primary relevant chunk content as factual answer body
    primary_text = clean_chunks[0].content.strip()
    return primary_text


def generate_rag_answer(
    query: str,
    user_access_level: AccessLevel,
    product: Optional[str] = None,
    region: Optional[str] = None,
    reference_date: Optional[Union[str, date]] = None,
    top_k: int = 5,
    *,
    _faiss_store=None,
    _llm_generator: Optional[Callable[[str, List[RetrievedChunk]], str]] = None,
) -> RAGAnswer:
    """
    Executes the guarded RAG answer generation workflow.

    Parameters
    ----------
    query:
        The customer's question.
    user_access_level:
        Caller's authorization level.
    product:
        Optional product filter.
    region:
        Optional geographic region filter.
    reference_date:
        Reference date for current/historical evaluation.
    top_k:
        Maximum number of chunks to evaluate.
    _faiss_store:
        Test-only vector store injection.
    _llm_generator:
        Optional callable (query, clean_chunks) -> str to customize answer body.
    """
    # ------------------------------------------------------------------
    # 1. Retrieval (enforces Access Level, Product, Region, Date, Conflict)
    # ------------------------------------------------------------------
    raw_chunks = retrieve_knowledge(
        query=query,
        user_access_level=user_access_level,
        product=product,
        region=region,
        top_k=top_k,
        reference_date=reference_date,
        _faiss_store=_faiss_store,
    )

    if not raw_chunks:
        return RAGAnswer(
            query=query,
            answer="I don't have enough information in the available knowledge base to answer that accurately.",
            citations=[],
            sufficiency=EvidenceSufficiency.INSUFFICIENT_EVIDENCE,
            is_refusal=True,
            refusal_reason="No relevant authorized evidence found.",
            retrieved_chunks=[],
        )

    # ------------------------------------------------------------------
    # 2. Document-Level Prompt-Injection Scan & Sanitization
    # ------------------------------------------------------------------
    clean_chunks: List[RetrievedChunk] = []
    injection_scans: List[InjectionScanResult] = []

    for chunk in raw_chunks:
        scan = scan_document_injection(chunk.content)
        injection_scans.append(scan)

        if scan.is_injection:
            if scan.sanitized_content and len(scan.sanitized_content.strip()) > 5:
                # Sanitized clean portion salvaged
                sanitized_chunk = chunk.model_copy(
                    update={"content": scan.sanitized_content.strip()}
                )
                clean_chunks.append(sanitized_chunk)
            else:
                # Completely malicious chunk dropped
                continue
        else:
            clean_chunks.append(chunk)

    if not clean_chunks:
        return RAGAnswer(
            query=query,
            answer="I cannot process this request because the retrieved evidence contains unauthorized instructions.",
            citations=[],
            sufficiency=EvidenceSufficiency.INSUFFICIENT_EVIDENCE,
            is_refusal=True,
            refusal_reason="All retrieved evidence contained malicious prompt injection.",
            retrieved_chunks=[],
            injection_scans=injection_scans,
        )

    # ------------------------------------------------------------------
    # 3. Evidence Sufficiency & Ambiguity Check
    # ------------------------------------------------------------------
    sufficiency, refusal_msg = evaluate_evidence_sufficiency(
        query=query,
        chunks=clean_chunks,
        product=product,
        region=region,
    )

    if sufficiency != EvidenceSufficiency.SUPPORTED:
        return RAGAnswer(
            query=query,
            answer=refusal_msg or "I don't have enough information in the available knowledge base to answer that accurately.",
            citations=[],
            sufficiency=sufficiency,
            is_refusal=True,
            refusal_reason=refusal_msg,
            retrieved_chunks=clean_chunks,
            injection_scans=injection_scans,
        )

    # ------------------------------------------------------------------
    # 4. Synthesize Candidate Answer & Formulate Citation
    # ------------------------------------------------------------------
    if _llm_generator is not None:
        candidate_body = _llm_generator(query, clean_chunks)
    else:
        candidate_body = _default_synthesize_answer(query, clean_chunks)

    # Attach citations
    citations_text = format_citations(clean_chunks)
    candidate_answer = f"{candidate_body.strip()}\n\n{citations_text}".strip()

    # ------------------------------------------------------------------
    # 5. Claim Grounding Verification (detect unsupported / hallucinated claims)
    # ------------------------------------------------------------------
    claim_ver = verify_claims(candidate_answer, clean_chunks)
    if not claim_ver.is_grounded:
        return RAGAnswer(
            query=query,
            answer="I don't have enough information in the available knowledge base to verify that answer accurately.",
            citations=[],
            sufficiency=EvidenceSufficiency.UNSUPPORTED_CLAIM,
            is_refusal=True,
            refusal_reason=f"Unsupported claims detected: {claim_ver.unsupported_claims or claim_ver.hallucinated_values}",
            retrieved_chunks=clean_chunks,
            claim_verification=claim_ver,
            injection_scans=injection_scans,
        )

    # ------------------------------------------------------------------
    # 6. Citation Validation (ensure citations match authorized evidence)
    # ------------------------------------------------------------------
    cit_val = validate_citations(candidate_answer, clean_chunks)
    if not cit_val.is_valid:
        return RAGAnswer(
            query=query,
            answer="I cannot provide an unverified answer without authorized citations.",
            citations=[],
            sufficiency=EvidenceSufficiency.UNSUPPORTED_CLAIM,
            is_refusal=True,
            refusal_reason=f"Citation validation failed: {cit_val.errors}",
            retrieved_chunks=clean_chunks,
            claim_verification=claim_ver,
            citation_validation=cit_val,
            injection_scans=injection_scans,
        )

    # ------------------------------------------------------------------
    # 7. Final Verified Answer
    # ------------------------------------------------------------------
    return RAGAnswer(
        query=query,
        answer=candidate_answer,
        citations=cit_val.valid_sources,
        sufficiency=EvidenceSufficiency.SUPPORTED,
        is_refusal=False,
        retrieved_chunks=clean_chunks,
        claim_verification=claim_ver,
        citation_validation=cit_val,
        injection_scans=injection_scans,
    )
