import os
import shutil
import hashlib
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import PipelineConfig

logger = logging.getLogger("pipeline.document_processor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Pipeline] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class IngestionResult:
    new_documents: List[str] = field(default_factory=list)
    modified_documents: List[str] = field(default_factory=list)
    unchanged_documents: List[str] = field(default_factory=list)
    duplicate_documents: List[str] = field(default_factory=list)
    quarantined_documents: List[Dict] = field(default_factory=list)
    processed_documents: List[str] = field(default_factory=list)
    chunks: List[Document] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    def summary(self) -> Dict:
        return {
            "new_count": len(self.new_documents),
            "modified_count": len(self.modified_documents),
            "unchanged_count": len(self.unchanged_documents),
            "duplicate_count": len(self.duplicate_documents),
            "quarantined_count": len(self.quarantined_documents),
            "processed_count": len(self.processed_documents),
            "total_chunks": len(self.chunks),
        }


class DocumentProcessor:
    """
    Task 1: Phase 1 Document Processor
    Handles Document Discovery, SHA256 Hashing, Duplicate Detection,
    PDF Validation, Quarantine, Text Extraction, and Chunking.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    # ------------------------------------------------------------------
    # SHA-256 Calculation
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_sha256(filepath: str) -> str:
        """Calculate SHA256 hex digest for a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    # ------------------------------------------------------------------
    # Persistent Registry Management
    # ------------------------------------------------------------------
    def load_registry(self) -> Dict:
        """Load persistent document registry JSON."""
        if os.path.exists(self.config.registry_file):
            try:
                with open(self.config.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read registry file ({e}). Starting with empty registry.")
        return {"documents": {}, "hashes": {}}

    def save_registry(self, registry: Dict) -> None:
        """Save persistent document registry JSON."""
        reg_dir = os.path.dirname(self.config.registry_file)
        if reg_dir and not os.path.exists(reg_dir):
            os.makedirs(reg_dir, exist_ok=True)
        with open(self.config.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

    # ------------------------------------------------------------------
    # Quarantine Management
    # ------------------------------------------------------------------
    def _record_quarantine(self, filename: str, filepath: str, reason: str, error_details: str = "") -> Dict:
        """Move an invalid document to quarantine and record report entry."""
        os.makedirs(self.config.quarantine_dir, exist_ok=True)
        destination = os.path.join(self.config.quarantine_dir, filename)

        # Move the invalid file
        if os.path.exists(filepath):
            shutil.move(filepath, destination)

        report_entry = {
            "filename": filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "error_details": error_details,
            "quarantined_path": destination,
        }

        # Load existing quarantine reports and append
        existing_reports = []
        if os.path.exists(self.config.quarantine_report_file):
            try:
                with open(self.config.quarantine_report_file, "r", encoding="utf-8") as f:
                    existing_reports = json.load(f)
                    if not isinstance(existing_reports, list):
                        existing_reports = []
            except Exception:
                existing_reports = []

        existing_reports.append(report_entry)
        report_dir = os.path.dirname(self.config.quarantine_report_file)
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir, exist_ok=True)
        with open(self.config.quarantine_report_file, "w", encoding="utf-8") as f:
            json.dump(existing_reports, f, indent=2)

        return report_entry

    # ------------------------------------------------------------------
    # PDF Validation & Text Extraction
    # ------------------------------------------------------------------
    def validate_and_extract_pdf(self, filepath: str) -> Tuple[bool, Optional[List[Document]], str, str]:
        """
        Validates PDF readability and non-empty content.
        Extracts pages as LangChain Document objects with metadata:
        - source: filename
        - page: 1-indexed page number
        - doc_type: 'pdf'

        Returns: (is_valid, documents_list, reason, error_details)
        """
        filename = os.path.basename(filepath)
        try:
            reader = pypdf.PdfReader(filepath)
        except Exception as e:
            return False, None, "Corrupted or unreadable PDF file", str(e)

        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
                if decrypted == 0:
                    return False, None, "Encrypted PDF cannot be read without credentials", ""
            except Exception as e:
                return False, None, "Encrypted/password-protected PDF", str(e)

        if len(reader.pages) == 0:
            return False, None, "PDF has 0 pages", ""

        page_documents = []
        total_text_length = 0

        for idx, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text() or ""
            except Exception as e:
                return False, None, f"Failed to extract text from page {idx + 1}", str(e)

            cleaned_text = extracted.strip()
            total_text_length += len(cleaned_text)

            if cleaned_text:
                page_documents.append(
                    Document(
                        page_content=cleaned_text,
                        metadata={
                            "source": filename,
                            "page": idx + 1,
                            "doc_type": "pdf"
                        }
                    )
                )

        if total_text_length == 0 or len(page_documents) == 0:
            return False, None, "Empty PDF: contains no extractable text content", ""

        return True, page_documents, "Validation passed", ""

    # ------------------------------------------------------------------
    # Document Discovery & Ingestion Pipeline
    # ------------------------------------------------------------------
    def process_knowledge_base(self) -> IngestionResult:
        """
        Scans knowledge_base/, computes hashes, checks duplicates,
        validates PDFs, quarantines invalid ones, extracts and chunks text.
        """
        result = IngestionResult()

        def log_event(msg: str, level=logging.INFO):
            logger.log(level, msg)
            result.logs.append(msg)

        if not os.path.exists(self.config.kb_dir):
            log_event(f"Knowledge base directory not found: {self.config.kb_dir}", logging.WARNING)
            return result

        registry = self.load_registry()
        registered_docs = registry.setdefault("documents", {})
        registered_hashes = registry.setdefault("hashes", {})

        # 1. Document Discovery
        discovered_files = []
        quarantine_norm = os.path.normpath(self.config.quarantine_dir)

        for root, dirs, files in os.walk(self.config.kb_dir):
            # Exclude quarantine directory from scan
            if os.path.normpath(root).startswith(quarantine_norm):
                continue
            for file in files:
                if file.startswith(".") or file.startswith("~"):
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in self.config.supported_extensions:
                    discovered_files.append(os.path.join(root, file))

        log_event(f"Discovered {len(discovered_files)} supported document(s) in {self.config.kb_dir}")

        current_batch_hashes = {}  # sha256 -> filename for detecting duplicates within the same run

        for filepath in discovered_files:
            filename = os.path.basename(filepath)

            # 2. SHA256 Hashing
            try:
                current_hash = self.calculate_sha256(filepath)
            except Exception as e:
                # File access issue or corruption before reading hash
                reason = "Failed to compute SHA256 hash"
                q_entry = self._record_quarantine(filename, filepath, reason, str(e))
                result.quarantined_documents.append(q_entry)
                log_event(f"validation failed: {filename} - {reason}: {e}", logging.ERROR)
                log_event(f"document quarantined: {filename} -> {q_entry['quarantined_path']}")
                continue

            # 3. Duplicate Detection
            # Exact duplicate check 1: Same hash already claimed by a DIFFERENT file in current run
            if current_hash in current_batch_hashes:
                other_file = current_batch_hashes[current_hash]
                result.duplicate_documents.append(filename)
                log_event(f"duplicate skipped: {filename} (identical content to {other_file}, hash: {current_hash[:10]}...)")
                continue

            # Exact duplicate check 2: Same hash registered under a DIFFERENT filename previously
            if current_hash in registered_hashes and registered_hashes[current_hash] != filename:
                existing_file = registered_hashes[current_hash]
                result.duplicate_documents.append(filename)
                log_event(f"duplicate skipped: {filename} (identical content to already registered file {existing_file}, hash: {current_hash[:10]}...)")
                continue

            # 4. Check if Unchanged or Modified
            is_new = filename not in registered_docs
            is_modified = False

            if not is_new:
                old_hash = registered_docs[filename].get("sha256")
                if old_hash == current_hash:
                    result.unchanged_documents.append(filename)
                    log_event(f"unchanged document skipped: {filename} (hash unchanged: {current_hash[:10]}...)")
                    current_batch_hashes[current_hash] = filename
                    continue
                else:
                    is_modified = True
                    result.modified_documents.append(filename)
                    log_event(f"modified document detected: {filename} (old: {old_hash[:10]}..., new: {current_hash[:10]}...)")
            else:
                result.new_documents.append(filename)
                log_event(f"new document detected: {filename} (hash: {current_hash[:10]}...)")

            # 5. PDF Validation
            is_valid, page_docs, reason, error_details = self.validate_and_extract_pdf(filepath)

            if not is_valid:
                # 6. Quarantine Invalid PDF
                q_entry = self._record_quarantine(filename, filepath, reason, error_details)
                result.quarantined_documents.append(q_entry)
                log_event(f"validation failed: {filename} - {reason}")
                log_event(f"document quarantined: {filename} -> {q_entry['quarantined_path']}")
                # If it was previously registered, remove old stale hash
                if filename in registered_docs:
                    old_h = registered_docs[filename].get("sha256")
                    registered_hashes.pop(old_h, None)
                    registered_docs.pop(filename, None)
                continue

            log_event(f"validation passed: {filename} ({len(page_docs)} pages extracted)")

            # 7. Chunking with RecursiveCharacterTextSplitter
            chunks = self.text_splitter.split_documents(page_docs)
            result.chunks.extend(chunks)
            result.processed_documents.append(filename)
            log_event(f"extraction completed: {filename} ({len(page_docs)} pages, {len(chunks)} chunks created)")

            # Update Registry
            current_batch_hashes[current_hash] = filename
            registered_hashes[current_hash] = filename
            registered_docs[filename] = {
                "sha256": current_hash,
                "status": "processed",
                "last_processed": datetime.now(timezone.utc).isoformat(),
                "pages_count": len(page_docs),
                "chunks_count": len(chunks),
                "is_modified": is_modified,
            }

        # Persist registry
        self.save_registry(registry)
        log_event(f"Ingestion pipeline finished. Summary: {result.summary()}")
        return result


if __name__ == "__main__":
    processor = DocumentProcessor()
    res = processor.process_knowledge_base()
    print("\n--- PHASE 1 INGESTION SUMMARY ---")
    print(json.dumps(res.summary(), indent=2))

