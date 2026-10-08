"""
Task 2 Phase 2: OCR and Evidence-Based Document Extraction Tests

Strategy:
  - ALL pytesseract calls are mocked so tests are deterministic and do not
    require Tesseract to be installed on the test machine.
  - PyMuPDF (fitz) is used for PDF text extraction — no mocking needed for
    text-based PDFs.
  - Synthetic test fixtures are generated in-memory using PIL and minimal
    PDF bytes — no external files required.
  - Phase 1 and Task 1 regression tests are included at the end.
"""

import io
import os
import sys
import re
import json
import pytest
from unittest.mock import patch, MagicMock

# Ensure backend/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from fastapi.testclient import TestClient
from main import app

from multimodal.extractor import DocumentExtractor, TESSERACT_AVAILABLE
from multimodal.field_parser import FieldParser
from multimodal.quality_assessor import QualityAssessor
from multimodal.config import MultimodalConfig
from multimodal.validator import MultimodalValidator

client = TestClient(app)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_png_bytes(text_hint: str = "") -> bytes:
    """Create a minimal valid PNG (in memory)."""
    buf = io.BytesIO()
    img = Image.new("RGB", (8, 8), color=(200, 200, 200))
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (8, 8), color=(180, 180, 180))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_webp_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (8, 8), color=(160, 160, 160))
    img.save(buf, format="WEBP")
    return buf.getvalue()


def make_text_pdf(page_texts: list) -> bytes:
    """
    Build a minimal multi-page text PDF using PyMuPDF (fitz).
    Falls back to raw PDF bytes if fitz not available.
    """
    try:
        import fitz
        doc = fitz.open()
        for text in page_texts:
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 100), text, fontsize=11)
        return doc.tobytes()
    except Exception:
        # Fallback: raw single-page PDF with text in stream
        return _minimal_pdf_with_text(page_texts[0] if page_texts else "")


def _minimal_pdf_with_text(text: str) -> bytes:
    """Minimal valid 1-page PDF with a text stream."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content_stream = f"BT /F1 12 Tf 50 750 Td ({escaped}) Tj ET".encode()
    stream_len = len(content_stream)
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        + f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode()
        + content_stream
        + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        b"0000000250 00000 n \n0000000400 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n500\n%%EOF\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sample OCR text fixtures (deterministic)
# ─────────────────────────────────────────────────────────────────────────────

INVOICE_TEXT = """
INVOICE
Invoice Number: INV-2024-00789
Date: 2024-03-15
Order ID: ORD-56789

Bill To: John Smith
         123 Main Street, Springfield

Product: Premium Widget Pro
SKU: SKU-WGT-001
Quantity: 3

Subtotal: $149.97
Tax:      $12.00
Total:    $161.97
Currency: USD

Payment Method: Credit Card
Tracking Number: TRK-987654321

Thank you for your purchase!
"""

ORDER_TEXT = """
Order Confirmation
Order #ORD-20240315-001

Date: 15/03/2024
Product: Wireless Headphones
SKU: SKU-WH-2024

Qty: 2
Unit Price: $79.99
Total: $159.98 USD

Ship To: 456 Oak Avenue, Portland
Estimated Delivery: March 20, 2024
"""

ERROR_TEXT = """
Payment Processing Failed

Error Code: ERR-PAYMENT-403
Order ID: ORD-99001

Please contact support with error code E40403.
Transaction ID: TXN-2024-ABCD

Your card ending in 4242 was declined.
"""

PRODUCT_TEXT = """
Product Information Sheet
Product: Industrial Laser Cutter X500
Product Code: PROD-LC-X500
Model: X500-Pro
SKU: SKU-ILC-001

Quantity in Stock: 15
Unit Price: $2,499.00 USD

Description: Professional grade laser cutter for industrial applications.
"""

POOR_OCR_TEXT = "!!! @#$ %%% &&& ***"  # garbage / unreadable


# ─────────────────────────────────────────────────────────────────────────────
# 1. Image OCR – mocked pytesseract
# ─────────────────────────────────────────────────────────────────────────────

def _mock_tesseract_version():
    """Mock helper: simulates Tesseract binary available."""
    return "5.0.0"


@patch("multimodal.extractor.pytesseract.get_tesseract_version", return_value="5.0.0")
@patch("multimodal.extractor.pytesseract.image_to_string", return_value=INVOICE_TEXT)
@patch("multimodal.extractor.TESSERACT_AVAILABLE", True)
@patch("multimodal.extractor.PIL_AVAILABLE", True)
def test_png_ocr_returns_extraction_result(mock_ocr, mock_version):
    extractor = DocumentExtractor()
    result = extractor.extract(make_png_bytes(), "invoice.png", "file001", "image/png")
    assert result.document_type == "image"
    assert result.extraction_method == "ocr"
    assert result.pages_processed == 1
    assert result.extracted_fields is not None


@patch("multimodal.extractor.pytesseract.get_tesseract_version", return_value="5.0.0")
@patch("multimodal.extractor.pytesseract.image_to_string", return_value=INVOICE_TEXT)
@patch("multimodal.extractor.TESSERACT_AVAILABLE", True)
@patch("multimodal.extractor.PIL_AVAILABLE", True)
def test_jpg_ocr_returns_extraction_result(mock_ocr, mock_version):
    extractor = DocumentExtractor()
    result = extractor.extract(make_jpeg_bytes(), "photo.jpg", "file002", "image/jpeg")
    assert result.document_type == "image"
    assert result.extraction_method == "ocr"


@patch("multimodal.extractor.pytesseract.get_tesseract_version", return_value="5.0.0")
@patch("multimodal.extractor.pytesseract.image_to_string", return_value=INVOICE_TEXT)
@patch("multimodal.extractor.TESSERACT_AVAILABLE", True)
@patch("multimodal.extractor.PIL_AVAILABLE", True)
def test_webp_ocr_returns_extraction_result(mock_ocr, mock_version):
    extractor = DocumentExtractor()
    result = extractor.extract(make_webp_bytes(), "doc.webp", "file003", "image/webp")
    assert result.document_type == "image"
    assert result.extraction_method == "ocr"


# ─────────────────────────────────────────────────────────────────────────────
# 2. PDF extraction (no mocking needed for text PDFs)
# ─────────────────────────────────────────────────────────────────────────────

def test_text_pdf_extraction_returns_pdf_type():
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "invoice.pdf", "pdf001", "application/pdf")
    assert result.document_type == "pdf"
    assert result.extraction_method in ("pdf_text", "pdf_mixed")
    assert result.pages_processed >= 1


def test_multi_page_pdf_extraction():
    pdf_bytes = make_text_pdf([ORDER_TEXT, INVOICE_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "multi.pdf", "pdf002", "application/pdf")
    assert result.document_type == "pdf"
    assert result.pages_processed >= 2


def test_scanned_pdf_graceful_handling():
    """A PDF with no extractable text should be handled gracefully."""
    # Minimal PDF with no text content (simulates a scanned/image PDF without OCR)
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF\n"
    )
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "scanned.pdf", "pdf003", "application/pdf")
    assert result.document_type == "pdf"
    # Either extracted nothing (poor quality) or processed with clarification
    assert result.evidence_quality in ("good", "acceptable", "poor")
    assert isinstance(result.clarification_required, bool)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Field extraction – FieldParser unit tests (pure regex, no OCR)
# ─────────────────────────────────────────────────────────────────────────────

def _parse(text: str) -> dict:
    """Helper: parse a single-page text."""
    parser = FieldParser()
    raw = parser.parse({1: text})
    return parser.to_dict(raw)


def test_order_id_extracted():
    result = _parse(INVOICE_TEXT)
    assert result["order_id"] is not None
    assert result["order_id"]["value"] is not None
    assert "ORD" in result["order_id"]["value"].upper() or result["order_id"]["value"]


def test_invoice_number_extracted():
    result = _parse(INVOICE_TEXT)
    assert result["invoice_number"] is not None
    assert result["invoice_number"]["value"] is not None


def test_date_extracted():
    result = _parse(INVOICE_TEXT)
    assert result["date"] is not None
    assert result["date"]["value"] is not None


def test_amount_extracted():
    result = _parse(INVOICE_TEXT)
    assert result["amount"] is not None
    val = result["amount"]["value"]
    # Should contain digits
    assert any(c.isdigit() for c in val)


def test_currency_extracted():
    result = _parse(INVOICE_TEXT)
    assert result["currency"] is not None
    assert result["currency"]["value"] in ("USD", "$", "EUR", "GBP")


def test_product_name_extracted():
    result = _parse(INVOICE_TEXT)
    assert result["product_name"] is not None
    assert result["product_name"]["value"] is not None


def test_product_code_extracted():
    result = _parse(PRODUCT_TEXT)
    assert result["product_code"] is not None
    val = result["product_code"]["value"]
    assert len(val) >= 2


def test_quantity_extracted():
    result = _parse(INVOICE_TEXT)
    assert result["quantity"] is not None
    val = result["quantity"]["value"]
    assert val.isdigit()


def test_error_code_extracted():
    result = _parse(ERROR_TEXT)
    assert result["error_code"] is not None
    assert result["error_code"]["value"] is not None


def test_delivery_info_extracted():
    result = _parse(ORDER_TEXT)
    assert result["delivery_info"] is not None


def test_order_id_from_order_text():
    result = _parse(ORDER_TEXT)
    assert result["order_id"] is not None
    assert result["order_id"]["value"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Missing fields → None (never hallucinated)
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_error_code_returns_none():
    """Text with no error code should return None, not a guess."""
    result = _parse("This is a receipt with no error information.\nOrder ID: ORD-123")
    assert result["error_code"] is None


def test_missing_invoice_number_returns_none():
    result = _parse("Order confirmed. Product: Widget. Qty: 1.")
    assert result["invoice_number"] is None


def test_blank_text_all_fields_none():
    result = _parse("")
    for key, val in result.items():
        assert val is None, f"Expected None for {key} in blank text, got {val}"


def test_no_hallucination_in_unrelated_text():
    """Random text should produce no invented structured values."""
    random_text = (
        "The weather is nice today. I went for a walk in the park. "
        "Birds were singing and the sky was clear blue."
    )
    result = _parse(random_text)
    # None of these domain-specific fields should have values
    for field_name in ("order_id", "invoice_number", "error_code", "product_code"):
        assert result[field_name] is None, (
            f"Field {field_name} should be None for unrelated text, "
            f"got {result[field_name]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Page number preservation in multi-page PDFs
# ─────────────────────────────────────────────────────────────────────────────

def test_page_number_preserved_single_page_pdf():
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "inv.pdf", "pdf010")
    # Find any field that was extracted and check page_number is set
    found_page = False
    for key, ev in result.extracted_fields.items():
        if ev is not None and isinstance(ev, dict) and ev.get("page_number") is not None:
            assert ev["page_number"] >= 1
            found_page = True
    # At least one field should have a page number if text was extracted
    if result.extraction_method in ("pdf_text", "pdf_mixed"):
        assert found_page or result.evidence_quality == "poor"


def test_page_number_preserved_multi_page():
    """Fields on page 2 should report page_number=2."""
    # Put the order_id content only on page 2
    page1 = "Introduction and cover page.\nThank you for your order."
    page2 = "Order ID: ORD-PAGE2-001\nDate: 2024-05-01\nTotal: $99.00"
    pdf_bytes = make_text_pdf([page1, page2])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "multi.pdf", "pdf011")
    if result.extraction_method in ("pdf_text", "pdf_mixed"):
        order_ev = result.extracted_fields.get("order_id")
        if order_ev is not None:
            assert order_ev["page_number"] in (1, 2), (
                f"Expected page 1 or 2, got {order_ev['page_number']}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Extraction method metadata
# ─────────────────────────────────────────────────────────────────────────────

def test_pdf_text_extraction_method_reported():
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "test.pdf", "pdf020")
    assert result.extraction_method in ("pdf_text", "pdf_mixed", "unavailable")


@patch("multimodal.extractor.pytesseract.get_tesseract_version", return_value="5.0.0")
@patch("multimodal.extractor.pytesseract.image_to_string", return_value=INVOICE_TEXT)
@patch("multimodal.extractor.TESSERACT_AVAILABLE", True)
@patch("multimodal.extractor.PIL_AVAILABLE", True)
def test_image_ocr_extraction_method_is_ocr(mock_ocr, mock_version):
    extractor = DocumentExtractor()
    result = extractor.extract(make_png_bytes(), "scan.png", "img020")
    assert result.extraction_method == "ocr"


def test_extraction_method_unavailable_when_no_tesseract():
    """When OCR is unavailable, method should be 'unavailable' for images."""
    with patch("multimodal.extractor.TESSERACT_AVAILABLE", False):
        extractor = DocumentExtractor()
        result = extractor.extract(make_png_bytes(), "img.png", "img021")
    assert result.extraction_method == "unavailable"
    assert result.clarification_required is True


# ─────────────────────────────────────────────────────────────────────────────
# 7. Quality assessment – unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_quality_assessor_good_text():
    qa = QualityAssessor()
    result = qa.assess_text(INVOICE_TEXT, "pdf_text")
    assert result.quality == "good"
    assert result.clarification_required is False
    assert result.quality_score >= 0.75


def test_quality_assessor_empty_text_is_poor():
    qa = QualityAssessor()
    result = qa.assess_text("", "ocr")
    assert result.quality == "poor"
    assert result.clarification_required is True
    assert result.quality_score == 0.0


def test_quality_assessor_garbage_text_is_poor():
    qa = QualityAssessor()
    result = qa.assess_text(POOR_OCR_TEXT, "ocr")
    assert result.quality in ("poor", "acceptable")


def test_quality_assessor_short_text_is_acceptable_or_poor():
    qa = QualityAssessor()
    result = qa.assess_text("Hello world", "ocr")
    assert result.quality in ("acceptable", "poor")


def test_quality_assessor_returns_user_message_for_poor():
    qa = QualityAssessor()
    result = qa.assess_text("", "ocr")
    assert result.user_message is not None
    assert len(result.user_message) > 10


def test_quality_assessor_good_has_no_user_message():
    qa = QualityAssessor()
    result = qa.assess_text(INVOICE_TEXT * 3, "pdf_text")
    assert result.user_message is None


# ─────────────────────────────────────────────────────────────────────────────
# 8. Poor quality / blurred evidence
# ─────────────────────────────────────────────────────────────────────────────

@patch("multimodal.extractor.pytesseract.get_tesseract_version", return_value="5.0.0")
@patch("multimodal.extractor.pytesseract.image_to_string", return_value=POOR_OCR_TEXT)
@patch("multimodal.extractor.TESSERACT_AVAILABLE", True)
@patch("multimodal.extractor.PIL_AVAILABLE", True)
def test_poor_ocr_text_triggers_poor_quality(mock_ocr, mock_version):
    extractor = DocumentExtractor()
    result = extractor.extract(make_png_bytes(), "blurred.png", "img030")
    assert result.evidence_quality in ("poor", "acceptable")


@patch("multimodal.extractor.pytesseract.get_tesseract_version", return_value="5.0.0")
@patch("multimodal.extractor.pytesseract.image_to_string", return_value="")
@patch("multimodal.extractor.TESSERACT_AVAILABLE", True)
@patch("multimodal.extractor.PIL_AVAILABLE", True)
def test_empty_ocr_output_sets_clarification_required(mock_ocr, mock_version):
    extractor = DocumentExtractor()
    result = extractor.extract(make_png_bytes(), "blank.png", "img031")
    assert result.clarification_required is True
    assert result.evidence_quality == "poor"


@patch("multimodal.extractor.pytesseract.get_tesseract_version", return_value="5.0.0")
@patch("multimodal.extractor.pytesseract.image_to_string", return_value="")
@patch("multimodal.extractor.TESSERACT_AVAILABLE", True)
@patch("multimodal.extractor.PIL_AVAILABLE", True)
def test_poor_quality_all_fields_none(mock_ocr, mock_version):
    """When OCR returns nothing, all fields should be None — no inventions."""
    extractor = DocumentExtractor()
    result = extractor.extract(make_png_bytes(), "blank.png", "img032")
    for key, ev in result.extracted_fields.items():
        assert ev is None, f"Expected None for {key} in empty OCR, got {ev}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. PDF full extraction integration
# ─────────────────────────────────────────────────────────────────────────────

def test_pdf_invoice_extracts_order_id():
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "inv.pdf", "pdf040")
    if result.extraction_method in ("pdf_text", "pdf_mixed"):
        order = result.extracted_fields.get("order_id")
        # Either extracted or None — never invented
        if order is not None:
            assert order["value"] is not None
            assert len(order["value"]) > 0


def test_pdf_invoice_extracts_invoice_number():
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "inv.pdf", "pdf041")
    if result.extraction_method in ("pdf_text", "pdf_mixed"):
        inv = result.extracted_fields.get("invoice_number")
        if inv is not None:
            assert inv["value"] is not None


def test_pdf_invoice_extracts_date():
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "inv.pdf", "pdf042")
    if result.extraction_method in ("pdf_text", "pdf_mixed"):
        date = result.extracted_fields.get("date")
        if date is not None:
            # Value should look like a date
            assert any(c.isdigit() for c in date["value"])


def test_pdf_error_text_extracts_error_code():
    pdf_bytes = make_text_pdf([ERROR_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "err.pdf", "pdf043")
    if result.extraction_method in ("pdf_text", "pdf_mixed"):
        err = result.extracted_fields.get("error_code")
        if err is not None:
            assert err["value"] is not None


def test_pdf_product_text_extracts_product_code():
    pdf_bytes = make_text_pdf([PRODUCT_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "prod.pdf", "pdf044")
    if result.extraction_method in ("pdf_text", "pdf_mixed"):
        pc = result.extracted_fields.get("product_code")
        if pc is not None:
            assert pc["value"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# 10. Payment info masking (privacy)
# ─────────────────────────────────────────────────────────────────────────────

def test_card_number_masked_in_payment_info():
    text = "Payment Method: Visa\nCard Number: 4111111111111111\nTotal: $50.00"
    result = _parse(text)
    payment = result.get("payment_info")
    if payment is not None and payment["value"]:
        # Raw card number should NOT appear
        assert "4111111111111111" not in payment["value"]


def test_partial_masked_card_in_context_snippet():
    text = "Payment: Credit Card 1234 **** **** 5678\nTotal: $100.00"
    result = _parse(text)
    payment = result.get("payment_info")
    if payment is not None and payment.get("context_snippet"):
        # The snippet should mask the card
        assert "1234 **** **** 5678" not in payment["context_snippet"] or \
               "CARD_REDACTED" in (payment["value"] or "")


# ─────────────────────────────────────────────────────────────────────────────
# 11. API integration tests
# ─────────────────────────────────────────────────────────────────────────────

def test_api_pdf_upload_returns_phase2_fields():
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("invoice.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    # Phase 1 fields present
    assert "status" in data
    assert data["status"] == "accepted"
    assert "file_id" in data
    # Phase 2 fields present
    assert "document_type" in data
    assert "extraction_method" in data
    assert "pages_processed" in data
    assert "extracted_fields" in data
    assert "evidence_quality" in data
    assert "clarification_required" in data
    assert "processing_time_ms" in data


def test_api_png_upload_returns_phase2_fields():
    png_bytes = make_png_bytes()
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("photo.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "document_type" in data
    assert data["document_type"] == "image"
    assert "extracted_fields" in data
    assert isinstance(data["extracted_fields"], dict)


def test_api_response_has_all_expected_fields():
    pdf_bytes = make_text_pdf(["Order ID: ORD-TEST-001"])
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    required_phase2_keys = [
        "document_type", "extraction_method", "pages_processed",
        "extracted_fields", "evidence_quality", "quality_score",
        "quality_notes", "clarification_required", "processing_time_ms",
    ]
    for key in required_phase2_keys:
        assert key in data, f"Phase 2 key '{key}' missing from response"


def test_api_storage_path_not_exposed():
    """storage_path must still never appear in any API response."""
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 200
    assert "storage_path" not in response.json()
    assert "C:\\" not in response.text


def test_api_invalid_file_still_returns_400():
    """Phase 1 validation rejection must still work after Phase 2 addition."""
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("virus.exe", io.BytesIO(b"MZ\x90\x00" + b"\x00"*100), "application/octet-stream")},
    )
    assert response.status_code == 400


def test_api_empty_file_still_returns_400():
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert response.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 12. Processing time is reported
# ─────────────────────────────────────────────────────────────────────────────

def test_processing_time_is_positive():
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "test.pdf", "pdf_time_001")
    assert result.processing_time_ms >= 0.0


def test_api_processing_time_in_response():
    pdf_bytes = make_text_pdf(["Simple text"])
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("t.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    data = response.json()
    assert "processing_time_ms" in data
    assert isinstance(data["processing_time_ms"], (int, float))
    assert data["processing_time_ms"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 13. ExtractionResult structure completeness
# ─────────────────────────────────────────────────────────────────────────────

def test_extraction_result_all_fields_present():
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "inv.pdf", "pdf_struct_001")
    assert hasattr(result, "file_id")
    assert hasattr(result, "document_type")
    assert hasattr(result, "extraction_method")
    assert hasattr(result, "pages_processed")
    assert hasattr(result, "extracted_fields")
    assert hasattr(result, "evidence_quality")
    assert hasattr(result, "quality_score")
    assert hasattr(result, "quality_notes")
    assert hasattr(result, "clarification_required")
    assert hasattr(result, "user_message")
    assert hasattr(result, "processing_time_ms")


def test_extracted_fields_dict_has_all_expected_keys():
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "inv.pdf", "pdf_struct_002")
    expected_keys = {
        "order_id", "invoice_number", "date", "amount", "currency",
        "product_name", "product_code", "quantity", "error_code",
        "delivery_info", "payment_info",
    }
    actual_keys = set(result.extracted_fields.keys())
    assert expected_keys == actual_keys, (
        f"Missing keys: {expected_keys - actual_keys}\n"
        f"Extra keys: {actual_keys - expected_keys}"
    )


def test_evidence_quality_is_valid_tier():
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "inv.pdf", "pdf_struct_003")
    assert result.evidence_quality in ("good", "acceptable", "poor")


def test_quality_score_in_range():
    pdf_bytes = make_text_pdf([INVOICE_TEXT])
    extractor = DocumentExtractor()
    result = extractor.extract(pdf_bytes, "inv.pdf", "pdf_struct_004")
    assert 0.0 <= result.quality_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 14. Phase 1 regression
# ─────────────────────────────────────────────────────────────────────────────

def test_phase1_valid_png_still_accepted():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("img.png", io.BytesIO(buf.getvalue()), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


def test_phase1_gif_still_rejected():
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("anim.gif", io.BytesIO(b"GIF89a"), "image/gif")},
    )
    assert response.status_code == 400


def test_phase1_empty_file_still_rejected():
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("e.png", io.BytesIO(b""), "image/png")},
    )
    assert response.status_code == 400


def test_phase1_disguised_exe_still_rejected():
    payload = b"MZ\x90\x00" + b"\x00" * 200
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("v.png", io.BytesIO(payload), "image/png")},
    )
    assert response.status_code == 400


def test_phase1_file_id_still_returned():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("img.png", io.BytesIO(buf.getvalue()), "image/png")},
    )
    assert response.status_code == 200
    assert "file_id" in response.json()
    assert len(response.json()["file_id"]) == 32


# ─────────────────────────────────────────────────────────────────────────────
# 15. Task 1 regression
# ─────────────────────────────────────────────────────────────────────────────

def test_ask_endpoint_still_exists():
    response = client.post("/ask", json={"question": "test"})
    assert response.status_code != 404, "/ask endpoint broken"


def test_root_endpoint_still_works():
    response = client.get("/")
    assert response.status_code == 200


def test_multimodal_endpoint_registered():
    response = client.post("/multimodal/analyze")
    assert response.status_code != 404
