"""
Task 4 Phase 1 — Document Loader
==================================
Loads knowledge-base PDFs and attaches rich ``DocumentMetadata`` to each page
as a LangChain ``Document.metadata`` dict.

Design decisions
----------------
* Uses ``pypdf`` (already a project dependency via Task 1) to extract pages.
* Each page becomes a LangChain ``Document`` whose ``metadata`` dict is the
  flattened form of a ``DocumentMetadata`` Pydantic model.
* The caller supplies the ``DocumentMetadata`` template; the loader stamps
  each page with the correct ``source_page`` and a fresh ``chunk_id``.
* This module does NOT modify the FAISS index — it only produces documents
  that can be fed to the chunker.
"""
from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.documents import Document

from .models import AccessLevel, DocumentMetadata, DocumentType
from .metadata import infer_document_type, metadata_to_flat_dict


# ---------------------------------------------------------------------------
# Knowledge-base document catalogue
# ---------------------------------------------------------------------------

# Default metadata for the three existing KB PDFs.
# These are the authoritative source of truth for the online-course domain.
_DEFAULT_KB_SPECS = [
    {
        "source_file": "knowledge_base/course_faqs.pdf",
        "document_version": "1.0",
        "document_type": DocumentType.FAQ,
        "product": "general",
        "region": "global",
        "access_level": AccessLevel.public,
        "effective_date": "2024-01-01",
        "expiry_date": None,
    },
    {
        "source_file": "knowledge_base/course_how_to_guides.pdf",
        "document_version": "1.0",
        "document_type": DocumentType.HOW_TO,
        "product": "general",
        "region": "global",
        "access_level": AccessLevel.customer,
        "effective_date": "2024-01-01",
        "expiry_date": None,
    },
    {
        "source_file": "knowledge_base/course_policies.pdf",
        "document_version": "1.0",
        "document_type": DocumentType.POLICY,
        "product": "general",
        "region": "global",
        "access_level": AccessLevel.customer,
        "effective_date": "2024-01-01",
        "expiry_date": None,
    },
]


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def _resolve_path(source_file: str, base_dir: Optional[str] = None) -> str:
    """
    Resolve *source_file* to an absolute path.

    Search order:
    1. As-is (if absolute or exists relative to cwd).
    2. Relative to *base_dir* if provided.
    3. Relative to the project root inferred from this file's location.
    """
    if os.path.isabs(source_file) and os.path.exists(source_file):
        return source_file
    if os.path.exists(source_file):
        return os.path.abspath(source_file)
    if base_dir:
        candidate = os.path.join(base_dir, source_file)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    # Infer project root: this file is at <project>/backend/rag/document_loader.py
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    candidate = os.path.join(project_root, source_file)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"Knowledge-base file not found: '{source_file}'")


def load_pdf_with_metadata(
    source_file: str,
    meta_template: DocumentMetadata,
    base_dir: Optional[str] = None,
) -> List[Document]:
    """
    Load a PDF and return one :class:`~langchain_core.documents.Document` per page,
    each carrying the full ``DocumentMetadata`` as a flat metadata dict.

    Parameters
    ----------
    source_file:
        Relative or absolute path to the PDF.
    meta_template:
        A ``DocumentMetadata`` instance describing the document.
        ``source_page`` and ``chunk_id`` are overridden per page.
    base_dir:
        Optional base directory to resolve relative paths against.

    Returns
    -------
    List[Document]
        One document per non-empty page, with ``metadata`` populated.
    """
    from pypdf import PdfReader
    from .metadata import inherit_metadata

    abs_path = _resolve_path(source_file, base_dir)
    reader = PdfReader(abs_path)
    documents: List[Document] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue  # skip blank pages

        page_meta = inherit_metadata(
            meta_template,
            source_page=page_num,
        )
        flat_meta = metadata_to_flat_dict(page_meta)

        documents.append(
            Document(page_content=text, metadata=flat_meta)
        )

    return documents


def load_documents_from_spec(
    spec: dict,
    base_dir: Optional[str] = None,
) -> List[Document]:
    """
    Build a ``DocumentMetadata`` from a raw spec dict and call
    :func:`load_pdf_with_metadata`.

    The spec dict must include at minimum: ``source_file``, ``document_version``,
    ``product``, ``access_level``, ``effective_date``.
    """
    meta = DocumentMetadata(**spec)
    return load_pdf_with_metadata(
        source_file=spec["source_file"],
        meta_template=meta,
        base_dir=base_dir,
    )


def load_default_knowledge_base(
    base_dir: Optional[str] = None,
) -> List[Document]:
    """
    Load all three default knowledge-base PDFs with their predefined metadata.

    Parameters
    ----------
    base_dir:
        Optional root directory for resolving relative paths.
        Defaults to the project root inferred from this module's location.

    Returns
    -------
    List[Document]
        Combined list of page-level documents from all KB PDFs.
    """
    all_docs: List[Document] = []
    for spec in _DEFAULT_KB_SPECS:
        try:
            docs = load_documents_from_spec(spec, base_dir=base_dir)
            all_docs.extend(docs)
        except FileNotFoundError:
            # Gracefully skip if a KB file is missing (e.g. in CI)
            pass
    return all_docs
