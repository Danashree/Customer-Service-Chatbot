import os
import io
import uuid
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from PIL import Image
import pypdf

from .config import MultimodalConfig, MIME_EXTENSION_MAP
from .security import MultimodalSecurityGuard
from pipeline.security import PIIMasker, SecurityEventLogger

logger = logging.getLogger("multimodal.validator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [MultimodalValidator] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Signatures for dangerous disguised files
SUSPICIOUS_MAGIC_SIGNATURES = [
    (b"MZ", "Windows Executable/DLL (MZ)"),
    (b"\x7fELF", "Linux ELF Binary"),
    (b"#!", "Executable Shell Script"),
    (b"<script", "HTML/JavaScript Script"),
    (b"<?php", "PHP Executable Script"),
]


class MultimodalValidationError(Exception):
    """Raised when file validation fails."""
    pass


class MultimodalValidator:
    """
    Task 2 Phase 1: Multimodal File Upload and Validation Engine.
    Enforces strict file type, MIME, content integrity, size, and disguised executable checks.
    Safely stores validated files in a protected temporary upload directory with UUID-based filenames.
    """

    def __init__(
        self,
        config: Optional[MultimodalConfig] = None,
        security_logger: Optional[SecurityEventLogger] = None
    ):
        self.config = config or MultimodalConfig()
        self.security_logger = security_logger or SecurityEventLogger(log_file=self.config.security_log_file)

    def validate_file(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validates an uploaded file across 6 critical dimensions:
        1. Non-empty check
        2. Size limit check
        3. Extension validation
        4. MIME type validation
        5. Disguised binary/script detection
        6. Format-specific deep content integrity verification (PIL for images, pypdf for PDFs)

        Returns:
            (is_valid: bool, reason: str, metadata: Optional[Dict[str, Any]])
        """
        sanitized_filename = PIIMasker.mask_text(filename or "unknown_file")

        # 0. Path traversal / unsafe filename check
        is_safe_name, err_reason = MultimodalSecurityGuard.validate_filename(filename)
        if not is_safe_name:
            self._log_rejection("PATH_TRAVERSAL_ATTEMPT", err_reason, sanitized_filename, severity="CRITICAL")
            return False, f"Invalid filename: {err_reason}", None

        # 1. Non-empty check
        if not file_bytes or len(file_bytes) == 0:
            self._log_rejection("EMPTY_FILE", "File payload is empty (0 bytes)", sanitized_filename)
            return False, "File is empty (0 bytes).", None

        # 2. File size limit
        max_bytes = self.config.max_file_size_mb * 1024 * 1024
        file_size = len(file_bytes)
        if file_size > max_bytes:
            msg = f"File size {file_size / (1024 * 1024):.2f}MB exceeds maximum limit of {self.config.max_file_size_mb}MB."
            self._log_rejection("OVERSIZED_FILE", msg, sanitized_filename)
            return False, msg, None

        # 3. File extension check
        ext = os.path.splitext(filename or "")[1].lower()
        if not ext or ext not in self.config.allowed_extensions:
            msg = f"Unsupported file extension '{ext}'. Allowed extensions: {list(self.config.allowed_extensions)}"
            self._log_rejection("UNSUPPORTED_EXTENSION", msg, sanitized_filename)
            return False, msg, None

        # 4. MIME type check
        expected_mime = MIME_EXTENSION_MAP.get(ext)
        if content_type:
            normalized_mime = content_type.lower().split(";")[0].strip()
            if normalized_mime not in self.config.allowed_mime_types:
                msg = f"Unsupported MIME type '{normalized_mime}'. Allowed MIME types: {list(self.config.allowed_mime_types)}"
                self._log_rejection("INVALID_MIME_TYPE", msg, sanitized_filename)
                return False, msg, None

            # Verify MIME matches extension
            if ext in (".jpg", ".jpeg") and normalized_mime != "image/jpeg":
                msg = f"MIME type '{normalized_mime}' does not match extension '{ext}' (expected image/jpeg)."
                self._log_rejection("MIME_EXTENSION_MISMATCH", msg, sanitized_filename)
                return False, msg, None
            elif ext == ".png" and normalized_mime != "image/png":
                msg = f"MIME type '{normalized_mime}' does not match extension '{ext}' (expected image/png)."
                self._log_rejection("MIME_EXTENSION_MISMATCH", msg, sanitized_filename)
                return False, msg, None
            elif ext == ".webp" and normalized_mime != "image/webp":
                msg = f"MIME type '{normalized_mime}' does not match extension '{ext}' (expected image/webp)."
                self._log_rejection("MIME_EXTENSION_MISMATCH", msg, sanitized_filename)
                return False, msg, None
            elif ext == ".pdf" and normalized_mime != "application/pdf":
                msg = f"MIME type '{normalized_mime}' does not match extension '{ext}' (expected application/pdf)."
                self._log_rejection("MIME_EXTENSION_MISMATCH", msg, sanitized_filename)
                return False, msg, None

        # 5. Check for disguised executables or scripts
        header_sample = file_bytes[:1024]
        for sig, desc in SUSPICIOUS_MAGIC_SIGNATURES:
            if header_sample.startswith(sig) or (sig in (b"<script", b"<?php") and sig in header_sample.lower()):
                msg = f"Security alert: Disguised unsafe executable or script detected ({desc}). File rejected."
                self._log_rejection("DISGUISED_MALICIOUS_FILE", msg, sanitized_filename, severity="CRITICAL")
                return False, msg, None

        # 6. Deep content validation by format
        try:
            if ext in (".png", ".jpg", ".jpeg", ".webp"):
                # Validate magic headers
                if ext == ".png" and not header_sample.startswith(b"\x89PNG\r\n\x1a\n"):
                    msg = "File lacks valid PNG magic header."
                    self._log_rejection("CORRUPTED_FILE", msg, sanitized_filename)
                    return False, msg, None

                if ext in (".jpg", ".jpeg") and not header_sample.startswith(b"\xff\xd8\xff"):
                    msg = "File lacks valid JPEG magic header."
                    self._log_rejection("CORRUPTED_FILE", msg, sanitized_filename)
                    return False, msg, None

                if ext == ".webp" and (not header_sample.startswith(b"RIFF") or b"WEBP" not in header_sample[:16]):
                    msg = "File lacks valid WEBP magic header."
                    self._log_rejection("CORRUPTED_FILE", msg, sanitized_filename)
                    return False, msg, None

                # Read format and size first (verify() exhausts the stream)
                bio = io.BytesIO(file_bytes)
                with Image.open(bio) as img:
                    img.load()  # fully load before reading attributes
                    format_name = img.format
                    width, height = img.size
                # Re-open to run verify() for deep integrity check
                bio2 = io.BytesIO(file_bytes)
                with Image.open(bio2) as img2:
                    img2.verify()

                metadata = {
                    "format": format_name,
                    "dimensions": {"width": width, "height": height},
                    "file_size_bytes": file_size,
                    "extension": ext,
                    "content_type": expected_mime
                }

            elif ext == ".pdf":
                # Validate PDF magic header
                if b"%PDF-" not in header_sample[:1024]:
                    msg = "File lacks valid PDF magic header (%PDF-)."
                    self._log_rejection("CORRUPTED_FILE", msg, sanitized_filename)
                    return False, msg, None

                bio = io.BytesIO(file_bytes)
                reader = pypdf.PdfReader(bio)
                page_count = len(reader.pages)
                if page_count == 0:
                    msg = "PDF file contains 0 pages or has corrupt page tree."
                    self._log_rejection("CORRUPTED_FILE", msg, sanitized_filename)
                    return False, msg, None

                metadata = {
                    "format": "PDF",
                    "page_count": page_count,
                    "file_size_bytes": file_size,
                    "extension": ext,
                    "content_type": expected_mime
                }
            else:
                msg = f"Unsupported format: '{ext}'"
                self._log_rejection("UNSUPPORTED_FORMAT", msg, sanitized_filename)
                return False, msg, None

        except Exception as e:
            msg = f"Corrupted or unreadable {ext.upper()} content: {e}"
            self._log_rejection("CORRUPTED_FILE", msg, sanitized_filename)
            return False, msg, None

        return True, "Validation successful.", metadata

    def save_upload(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates the file and safely persists it into the configured temporary upload directory.
        Generates a secure UUIDv4-based filename so the untrusted client filename is never stored on disk.
        """
        is_valid, reason, metadata = self.validate_file(file_bytes, filename, content_type)
        if not is_valid:
            raise MultimodalValidationError(reason)

        file_id = uuid.uuid4().hex
        ext = os.path.splitext(filename)[1].lower()
        safe_filename = f"{file_id}{ext}"
        destination_path = os.path.join(self.config.upload_dir, safe_filename)
        if not MultimodalSecurityGuard.ensure_safe_upload_path(destination_path, self.config.upload_dir):
            self._log_rejection("PATH_TRAVERSAL_ATTEMPT", "Target file path escapes upload directory", sanitized_filename, severity="CRITICAL")
            raise MultimodalValidationError("Security violation: Upload destination is outside the allowed directory.")

        # Write safely
        tmp_path = f"{destination_path}.tmp_{os.getpid()}"
        try:
            with open(tmp_path, "wb") as f:
                f.write(file_bytes)
            os.replace(tmp_path, destination_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise RuntimeError(f"Failed to store uploaded file: {e}")

        sanitized_filename = PIIMasker.mask_text(filename)
        return {
            "file_id": file_id,
            "storage_path": destination_path,
            "safe_filename": safe_filename,
            "original_filename": sanitized_filename,
            "file_size_bytes": len(file_bytes),
            "content_type": metadata.get("content_type") if metadata else MIME_EXTENSION_MAP.get(ext),
            "metadata": metadata or {}
        }

    def _log_rejection(self, event_type: str, reason: str, filename: str, severity: str = "WARNING"):
        try:
            self.security_logger.log_event(
                event_type=f"MULTIMODAL_{event_type}",
                severity=severity,
                action="upload_validation",
                result="DENIED",
                resource=filename,
                details={"reason": reason}
            )
        except Exception as e:
            logger.warning(f"Could not write security log for multimodal rejection: {e}")

