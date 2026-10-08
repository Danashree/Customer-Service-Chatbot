"""
Task 4 Phase 1 — Metadata-Preserving Chunker
=============================================
Wraps LangChain's ``RecursiveCharacterTextSplitter`` to guarantee that
every chunk produced from a ``Document`` inherits all metadata fields
from the parent document, with only ``chunk_id`` replaced per chunk.

Design decisions
----------------
* ``chunk_size`` defaults to 500 characters — small enough for precise
  retrieval in a customer-service context.
* ``chunk_overlap`` defaults to 50 characters to prevent context loss
  at chunk boundaries.
* The ``source_page`` is preserved from the parent document so the chunk
  can always be traced back to its origin page.
* A fresh UUID ``chunk_id`` is assigned to each produced chunk.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .metadata import metadata_from_flat_dict, metadata_to_flat_dict, inherit_metadata
from .models import DocumentMetadata


# ---------------------------------------------------------------------------
# Default chunker parameters
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of ``Document`` objects into smaller chunks.

    Each chunk:
    * Inherits **all** metadata fields from its parent document.
    * Receives a **fresh** ``chunk_id`` (UUID4).
    * Preserves the parent's ``source_page`` so provenance is never lost.

    Parameters
    ----------
    documents:
        Source documents (typically the output of :mod:`document_loader`).
        Each must have a fully populated metadata dict that can be
        deserialized into a :class:`~rag.models.DocumentMetadata`.
    chunk_size:
        Maximum number of characters per chunk.
    chunk_overlap:
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    List[Document]
        Chunked documents with metadata preserved and ``chunk_id`` updated.

    Raises
    ------
    ValueError
        If a document's metadata dict is missing required fields.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False,
    )

    chunked: List[Document] = []

    for doc in documents:
        # Deserialize parent metadata for inheritance
        try:
            parent_meta = metadata_from_flat_dict(doc.metadata)
        except Exception as exc:
            raise ValueError(
                f"Cannot deserialize metadata for document "
                f"'{doc.metadata.get('source_file', '<unknown>')}': {exc}"
            ) from exc

        # Split text
        sub_texts = splitter.split_text(doc.page_content)

        for sub_text in sub_texts:
            if not sub_text.strip():
                continue  # drop whitespace-only chunks

            child_meta = inherit_metadata(parent_meta)  # assigns fresh chunk_id
            flat_meta = metadata_to_flat_dict(child_meta)

            chunked.append(Document(page_content=sub_text, metadata=flat_meta))

    return chunked


def chunk_documents_with_custom_splitter(
    documents: List[Document],
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """
    Chunk documents using a caller-supplied splitter instance.

    Metadata inheritance rules are identical to :func:`chunk_documents`.
    Use this when you need fine-grained control over splitting parameters.
    """
    chunked: List[Document] = []

    for doc in documents:
        try:
            parent_meta = metadata_from_flat_dict(doc.metadata)
        except Exception as exc:
            raise ValueError(
                f"Cannot deserialize metadata for document "
                f"'{doc.metadata.get('source_file', '<unknown>')}': {exc}"
            ) from exc

        sub_texts = splitter.split_text(doc.page_content)

        for sub_text in sub_texts:
            if not sub_text.strip():
                continue

            child_meta = inherit_metadata(parent_meta)
            flat_meta = metadata_to_flat_dict(child_meta)

            chunked.append(Document(page_content=sub_text, metadata=flat_meta))

    return chunked
