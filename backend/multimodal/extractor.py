"""
Task 2 Phase 2 – Document Extractor

Core extraction engine that:
1. Runs OCR on images (PNG/JPG/JPEG/WEBP) via pytesseract (when available)
2. Extracts text from text-based PDFs via PyMuPDF (fitz)
3. Falls back to OCR for image/scanned PDF pages
4. Delegates field parsing to FieldParser
5. Delegates quality scoring to QualityAssessor
6. Returns a structured ExtractionResult — never exposing filesystem paths or raw PII

Privacy:
  - Raw OCR text is NEVER written to logs
  - PII in extracted field values is masked via PIIMasker
  - Only field-level extraction events are logged
"""

import io
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .field_parser import FieldParser
from .quality_assessor import QualityAssessor, QualityResult
from .security import MultimodalSecurityGuard
from pipeline.security import SecurityEventLogger

logger = logging.getLogger("multimodal.extractor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [DocumentExtractor] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ─── Optional heavy deps (graceful fallback if not installed) ──────────────────

try:
    import pytesseract
    from PIL import Image as PILImage
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import pymupdf as fitz  # PyMuPDF (preferred import style)
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz  # fallback for older pymupdf versions
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False

try:
    from PIL import Image as PILImage, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pipeline.security import PIIMasker
    PII_MASKER_AVAILABLE = True
except ImportError:
    PII_MASKER_AVAILABLE = False

# Minimum text chars on a PDF page before we consider it "text-based"
PDF_MIN_TEXT_CHARS = 20


# ─── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Structured result of a document extraction pass."""
    file_id: str
    document_type: str              # "image" | "pdf"
    extraction_method: str          # "ocr" | "pdf_text" | "pdf_mixed" | "unavailable"
    pages_processed: int
    extracted_fields: Dict[str, Any]   # field name → {value, page_number, ...} or None
    evidence_quality: str           # "good" | "acceptable" | "poor"
    quality_score: float
    quality_notes: List[str]
    clarification_required: bool
    user_message: Optional[str]
    processing_time_ms: float
    prompt_injection_detected: bool = False
    security_notes: List[str] = field(default_factory=list)


# ─── Extractor ────────────────────────────────────────────────────────────────

class DocumentExtractor:
    """
    Orchestrates OCR and PDF text extraction for validated multimodal uploads.
    Pluggable OCR backend — gracefully degrades when Tesseract is unavailable.
    """

    def __init__(self, security_logger: Optional[SecurityEventLogger] = None):
        self._parser = FieldParser()
        self._quality = QualityAssessor()
        self.security_logger = security_logger or SecurityEventLogger()

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def extract(
        self,
        file_bytes: bytes,
        filename: str,
        file_id: str,
        content_type: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Main extraction method. Routes to PDF or image pipeline based on extension.
        """
        start = time.monotonic()
        ext = self._ext(filename)
        is_pdf = ext == ".pdf"

        try:
            if is_pdf:
                result = self._extract_pdf(file_bytes, file_id)
            else:
                result = self._extract_image(file_bytes, file_id, ext)
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.warning("Extraction error for file_id=%s: %s", file_id, type(e).__name__)
            result = self._error_result(file_id, ext, elapsed, str(e))

        result.processing_time_ms = round((time.monotonic() - start) * 1000, 2)
        logger.info(
            "Extraction complete: file_id=%s type=%s method=%s quality=%s pages=%d injection=%s",
            file_id,
            result.document_type,
            result.extraction_method,
            result.evidence_quality,
            result.pages_processed,
            result.prompt_injection_detected,
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Image extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_image(self, file_bytes: bytes, file_id: str, ext: str) -> ExtractionResult:
        # Pre-OCR image quality check
        image_quality: QualityResult = self._quality.assess_image(file_bytes)

        # Attempt OCR
        ocr_text, ocr_available = self._run_ocr_on_bytes(file_bytes)

        prompt_injection_detected = False
        security_notes = []

        # Text quality on top of image quality
        if ocr_available and ocr_text is not None:
            # Check prompt injection on untrusted extracted text
            injection_check = MultimodalSecurityGuard.scan_prompt_injection(
                ocr_text, source="image_ocr", file_id=file_id, security_logger=self.security_logger
            )
            prompt_injection_detected = injection_check["detected"]
            if prompt_injection_detected:
                security_notes.append(
                    f"Prompt injection detected ({', '.join(injection_check['categories'])}). Content treated as untrusted data."
                )

            text_quality = self._quality.assess_text(ocr_text, "ocr")
            # Use the worse of the two assessments
            combined_score = min(image_quality.quality_score, text_quality.quality_score)
            combined_notes = image_quality.quality_notes + text_quality.quality_notes
            final_quality, clarification = QualityAssessor._tier(combined_score)
            user_msg = QualityAssessor._user_message(final_quality, combined_notes)

            text_by_page = {1: ocr_text}
            parsed = self._parser.parse(text_by_page)
            extracted = self._parser.to_dict(parsed)
            extracted = self._mask_pii_in_fields(extracted)

            method = "ocr"
        else:
            # OCR unavailable — report gracefully
            combined_score = 0.2
            combined_notes = image_quality.quality_notes + [
                "Tesseract OCR engine is not installed on this server. "
                "Text could not be extracted from the image."
            ]
            final_quality = "poor"
            clarification = True
            user_msg = (
                "OCR is currently unavailable on this server. "
                "Please upload a text-based PDF instead, or contact support."
            )
            extracted = self._empty_fields()
            method = "unavailable"

        return ExtractionResult(
            file_id=file_id,
            document_type="image",
            extraction_method=method,
            pages_processed=1,
            extracted_fields=extracted,
            evidence_quality=final_quality,
            quality_score=round(combined_score, 3),
            quality_notes=combined_notes,
            clarification_required=clarification,
            user_message=user_msg,
            processing_time_ms=0.0,  # set by caller
            prompt_injection_detected=prompt_injection_detected,
            security_notes=security_notes,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PDF extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_pdf(self, file_bytes: bytes, file_id: str) -> ExtractionResult:
        if not PYMUPDF_AVAILABLE:
            # Fall back to pypdf
            return self._extract_pdf_pypdf(file_bytes, file_id)

        text_by_page: Dict[int, str] = {}
        page_methods: List[str] = []

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(doc)
            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_num = page_idx + 1
                page_text = page.get_text("text").strip()

                if len(page_text) >= PDF_MIN_TEXT_CHARS:
                    # Text-based page
                    text_by_page[page_num] = page_text
                    page_methods.append("pdf_text")
                else:
                    # Likely scanned — attempt OCR via rendering
                    ocr_text = self._ocr_pdf_page(page, page_num)
                    text_by_page[page_num] = ocr_text or ""
                    page_methods.append("ocr" if ocr_text else "unavailable")

            doc.close()

        except Exception as e:
            logger.warning("PyMuPDF extraction error: %s", type(e).__name__)
            return self._extract_pdf_pypdf(file_bytes, file_id)

        # Determine overall extraction method
        if all(m == "pdf_text" for m in page_methods):
            method = "pdf_text"
        elif all(m == "unavailable" for m in page_methods):
            method = "unavailable"
        elif any(m == "ocr" for m in page_methods):
            method = "pdf_mixed"
        else:
            method = "pdf_text"

        # Assess quality
        full_text = "\n".join(text_by_page.values())
        text_quality = self._quality.assess_text(full_text, method)

        # Check prompt injection on PDF text
        injection_check = MultimodalSecurityGuard.scan_prompt_injection(
            full_text, source="pdf_document", file_id=file_id, security_logger=self.security_logger
        )
        prompt_injection_detected = injection_check["detected"]
        security_notes = []
        if prompt_injection_detected:
            security_notes.append(
                f"Prompt injection detected ({', '.join(injection_check['categories'])}). Content treated as untrusted data."
            )

        # Parse fields
        parsed = self._parser.parse(text_by_page)
        extracted = self._parser.to_dict(parsed)
        extracted = self._mask_pii_in_fields(extracted)

        return ExtractionResult(
            file_id=file_id,
            document_type="pdf",
            extraction_method=method,
            pages_processed=len(text_by_page),
            extracted_fields=extracted,
            evidence_quality=text_quality.quality,
            quality_score=text_quality.quality_score,
            quality_notes=text_quality.quality_notes,
            clarification_required=text_quality.clarification_required,
            user_message=text_quality.user_message,
            processing_time_ms=0.0,
            prompt_injection_detected=prompt_injection_detected,
            security_notes=security_notes,
        )

    def _extract_pdf_pypdf(self, file_bytes: bytes, file_id: str) -> ExtractionResult:
        """Fallback PDF extraction using pypdf (already installed in Phase 1)."""
        import pypdf

        text_by_page: Dict[int, str] = {}
        try:
            bio = io.BytesIO(file_bytes)
            reader = pypdf.PdfReader(bio)
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                text = page.extract_text() or ""
                text_by_page[page_num] = text.strip()
        except Exception as e:
            logger.warning("pypdf fallback error: %s", type(e).__name__)
            text_by_page = {1: ""}

        full_text = "\n".join(text_by_page.values())
        text_quality = self._quality.assess_text(full_text, "pdf_text")

        injection_check = MultimodalSecurityGuard.scan_prompt_injection(
            full_text, source="pdf_document_pypdf", file_id=file_id, security_logger=self.security_logger
        )
        prompt_injection_detected = injection_check["detected"]
        security_notes = []
        if prompt_injection_detected:
            security_notes.append(
                f"Prompt injection detected ({', '.join(injection_check['categories'])}). Content treated as untrusted data."
            )

        parsed = self._parser.parse(text_by_page)
        extracted = self._parser.to_dict(parsed)
        extracted = self._mask_pii_in_fields(extracted)

        return ExtractionResult(
            file_id=file_id,
            document_type="pdf",
            extraction_method="pdf_text",
            pages_processed=len(text_by_page),
            extracted_fields=extracted,
            evidence_quality=text_quality.quality,
            quality_score=text_quality.quality_score,
            quality_notes=text_quality.quality_notes,
            clarification_required=text_quality.clarification_required,
            user_message=text_quality.user_message,
            processing_time_ms=0.0,
            prompt_injection_detected=prompt_injection_detected,
            security_notes=security_notes,
        )

    def _ocr_pdf_page(self, page, page_num: int) -> Optional[str]:
        """Render a PDF page to an image and run OCR on it."""
        if not PIL_AVAILABLE:
            return None
        try:
            # Render at 150 DPI
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            ocr_text, available = self._run_ocr_on_bytes(img_bytes)
            return ocr_text if available else None
        except Exception as e:
            logger.debug("PDF page OCR failed for page %d: %s", page_num, type(e).__name__)
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # OCR helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _run_ocr_on_bytes(self, image_bytes: bytes):
        """
        Run pytesseract on raw image bytes.
        Returns (text_or_None, ocr_was_available).
        """
        if not TESSERACT_AVAILABLE or not PIL_AVAILABLE:
            return None, False
        try:
            # Check if tesseract binary is callable
            pytesseract.get_tesseract_version()
        except Exception:
            return None, False

        try:
            bio = io.BytesIO(image_bytes)
            img = PILImage.open(bio).convert("RGB")
            # Upscale small images for better OCR accuracy
            w, h = img.size
            if w < 300 or h < 300:
                scale = max(300 / w, 300 / h)
                img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
            # Enhance contrast
            img = ImageEnhance.Contrast(img).enhance(1.5)
            text = pytesseract.image_to_string(img, config="--psm 6")
            return text, True
        except Exception as e:
            logger.debug("pytesseract error: %s", type(e).__name__)
            return None, False

    # ──────────────────────────────────────────────────────────────────────────
    # Privacy / PII masking
    # ──────────────────────────────────────────────────────────────────────────

    def _mask_pii_in_fields(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """Mask PII, payment info, credentials in all extracted field values."""
        for key, evidence in extracted.items():
            if evidence is None:
                continue
            if isinstance(evidence, dict) and evidence.get("value"):
                evidence["value"] = MultimodalSecurityGuard.mask_sensitive_text(str(evidence["value"]))
                # Also mask context snippet
                if evidence.get("context_snippet"):
                    evidence["context_snippet"] = MultimodalSecurityGuard.mask_sensitive_text(evidence["context_snippet"])
        return extracted

    # ──────────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ext(filename: str) -> str:
        import os
        return os.path.splitext(filename or "")[1].lower()

    @staticmethod
    def _empty_fields() -> Dict[str, Any]:
        field_names = [
            "order_id", "invoice_number", "date", "amount", "currency",
            "product_name", "product_code", "quantity", "error_code",
            "delivery_info", "payment_info",
        ]
        return {k: None for k in field_names}

    @staticmethod
    def _error_result(file_id: str, ext: str, elapsed_ms: float, error: str) -> ExtractionResult:
        doc_type = "pdf" if ext == ".pdf" else "image"
        return ExtractionResult(
            file_id=file_id,
            document_type=doc_type,
            extraction_method="unavailable",
            pages_processed=0,
            extracted_fields=DocumentExtractor._empty_fields(),
            evidence_quality="poor",
            quality_score=0.0,
            quality_notes=[f"Extraction failed: {error}"],
            clarification_required=True,
            user_message="Document extraction failed. Please upload a clearer file.",
            processing_time_ms=elapsed_ms,
            prompt_injection_detected=False,
            security_notes=[],
        )
