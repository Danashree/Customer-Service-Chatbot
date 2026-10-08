"""
Task 4 Phase 3 — Mandatory Source Citations & Validation
==========================================================
Generates and validates source citations derived strictly from authorized
retrieved evidence.

Key guarantees:
- Formats citations with source file, page (when available), and version.
- Rejects fabricated, nonexistent, or inaccessible citations.
- Enforces that factual answers MUST include valid citations matching retrieved chunks.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Set

from .models import CitationValidationResult, ProvenanceRecord, RetrievedChunk


def format_citation(prov: ProvenanceRecord) -> str:
    """
    Format a standardized citation string from a ProvenanceRecord.

    Example:
      "Source: course_policies.pdf, page 5, version v1.0"
      "Source: course_faqs.pdf, version v1.0" (when page is None)
    """
    filename = os.path.basename(prov.source_file)
    page_str = f", page {prov.source_page}" if prov.source_page is not None else ""
    version_str = f", version v{prov.document_version}" if prov.document_version else ""
    return f"Source: {filename}{page_str}{version_str}"


def format_citations(chunks: List[RetrievedChunk]) -> str:
    """
    Formats deduplicated citations for a list of retrieved chunks.
    """
    if not chunks:
        return ""
    seen: Set[str] = set()
    citations: List[str] = []
    for chunk in chunks:
        c = format_citation(chunk.provenance)
        if c not in seen:
            seen.add(c)
            citations.append(c)
    return "\n".join(citations)


def parse_citations(text: str) -> List[str]:
    """
    Extracts citation strings from answer text.
    Recognizes:
      - "Source: ..."
      - "[POLICY] ... (v...)"
      - "[FAQ] ... (v...)"
    """
    citations: List[str] = []
    # Match "Source: ..."
    for match in re.finditer(r"(?i)\bSource:\s*([^\n\r]+)", text):
        citations.append(match.group(0).strip())

    # Match bracketed format "[POLICY] course_policies.pdf, page 5 (v1.0, customer)"
    for match in re.finditer(
        r"\[(?:FAQ|POLICY|HOW_TO|TROUBLESHOOTING|GENERAL)\]\s*([^\n\r]+)",
        text,
        re.IGNORECASE,
    ):
        citations.append(match.group(0).strip())

    return citations


def validate_citations(
    answer: str,
    authorized_chunks: List[RetrievedChunk],
) -> CitationValidationResult:
    """
    Validates that the citations present in `answer` match actual authorized
    retrieved chunks.

    Checks:
    1. Answer contains at least one citation.
    2. Every cited file matches a chunk in `authorized_chunks`.
    3. Any cited page or version matches the corresponding chunk metadata.
    4. Citing an inaccessible or nonexistent document is rejected.
    """
    errors: List[str] = []
    cited_strings = parse_citations(answer)

    if not cited_strings:
        return CitationValidationResult(
            is_valid=False,
            cited_sources=[],
            valid_sources=[],
            invalid_sources=[],
            errors=["Factual answer lacks mandatory source citation."],
        )

    # Build index of authorized sources from retrieved chunks
    authorized_files = set()
    authorized_full_keys = set()  # (filename, page, version)
    for chunk in authorized_chunks:
        meta = chunk.metadata
        fname = os.path.basename(meta.source_file).lower()
        authorized_files.add(fname)
        authorized_full_keys.add(
            (fname, meta.source_page, str(meta.document_version).lower())
        )

    valid_sources: List[str] = []
    invalid_sources: List[str] = []

    for citation in cited_strings:
        cit_lower = citation.lower()
        # Find which authorized file is mentioned in this citation
        matched_file = None
        for af in authorized_files:
            if af in cit_lower:
                matched_file = af
                break

        if not matched_file:
            # Citation references a file not in authorized retrieved chunks
            invalid_sources.append(citation)
            errors.append(
                f"Citation '{citation}' references a nonexistent or unauthorized document."
            )
            continue

        # If page number is specified, verify it
        page_match = re.search(r"page\s+(\d+)", cit_lower)
        if page_match:
            cited_page = int(page_match.group(1))
            # Verify that at least one chunk for this file matches the page
            page_matched = any(
                af == matched_file and page == cited_page
                for af, page, _ in authorized_full_keys
            )
            if not page_matched:
                invalid_sources.append(citation)
                errors.append(
                    f"Citation '{citation}' references page {cited_page}, which is not present in retrieved evidence."
                )
                continue

        # If version is specified, verify it
        ver_match = re.search(r"version\s+v?([0-9\.]+)", cit_lower) or re.search(
            r"\(v([0-9\.]+)", cit_lower
        )
        if ver_match:
            cited_ver = ver_match.group(1).lower()
            ver_matched = any(
                af == matched_file and ver == cited_ver
                for af, _, ver in authorized_full_keys
            )
            if not ver_matched:
                invalid_sources.append(citation)
                errors.append(
                    f"Citation '{citation}' references version {cited_ver}, which is not in retrieved evidence."
                )
                continue

        valid_sources.append(citation)

    is_valid = len(invalid_sources) == 0 and len(valid_sources) > 0
    return CitationValidationResult(
        is_valid=is_valid,
        cited_sources=cited_strings,
        valid_sources=valid_sources,
        invalid_sources=invalid_sources,
        errors=errors,
    )
