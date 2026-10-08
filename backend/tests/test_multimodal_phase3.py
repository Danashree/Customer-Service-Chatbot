"""
Task 2 Phase 3: Evidence Comparison & Conflict Detection Tests

Tests:
- Message parser field extraction (order ID, invoice, date, amount, currency, product, error code, quantity, etc.)
- Normalization (order ID, currency, amount, date, quantity, error code)
- Matching order ID
- Conflicting order ID
- Matching invoice number
- Conflicting invoice number
- Matching amount
- Conflicting amount
- Date normalization & matching
- Currency normalization & matching
- Product code matching
- Quantity matching
- Error code conflict
- Message-only field (customer mentions field not in document)
- Evidence-only field (document has field not mentioned by customer)
- Missing field handling
- Poor evidence quality handling (insufficient evidence & clarification)
- Ambiguous/unreadable evidence
- Multiple fields in one customer message (independent evaluation)
- No detectable values in customer message
- No hallucination safeguards (never invent missing values)
- Clarification required behavior
- API integration: POST /multimodal/analyze with message
- API integration: POST /multimodal/analyze without message (backwards compatibility)
- Phase 1 upload validation regression
- Phase 2 extraction regression
- Task 1 chatbot regression (/ask endpoint)
"""

import io
import os
import sys
import json
import pytest
from unittest.mock import patch

# Ensure backend/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from fastapi.testclient import TestClient
from main import app

from multimodal.message_parser import MessageParser
from multimodal.comparator import (
    EvidenceComparator,
    ComparisonResult,
    ConflictingField,
    normalize_id,
    normalize_amount,
    normalize_currency,
    normalize_date,
    normalize_quantity,
)

client = TestClient(app)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_EVIDENCE = {
    "order_id": {
        "value": "ORD-56789",
        "page_number": 1,
        "extraction_method_detail": "order_id",
        "context_snippet": "Order ID: ORD-56789",
    },
    "invoice_number": {
        "value": "INV-2024-00789",
        "page_number": 1,
        "extraction_method_detail": "invoice_number",
        "context_snippet": "Invoice Number: INV-2024-00789",
    },
    "date": {
        "value": "2024-03-15",
        "page_number": 1,
        "extraction_method_detail": "date",
        "context_snippet": "Date: 2024-03-15",
    },
    "amount": {
        "value": "161.97",
        "page_number": 1,
        "extraction_method_detail": "amount",
        "context_snippet": "Total: $161.97",
    },
    "currency": {
        "value": "USD",
        "page_number": None,
        "extraction_method_detail": "currency",
        "context_snippet": "Currency: USD",
    },
    "product_name": {
        "value": "Premium Widget Pro",
        "page_number": 1,
        "extraction_method_detail": "product_name",
        "context_snippet": "Product: Premium Widget Pro",
    },
    "product_code": {
        "value": "SKU-WGT-001",
        "page_number": 1,
        "extraction_method_detail": "product_code",
        "context_snippet": "SKU: SKU-WGT-001",
    },
    "quantity": {
        "value": "3",
        "page_number": 1,
        "extraction_method_detail": "quantity",
        "context_snippet": "Quantity: 3",
    },
    "error_code": None,
    "delivery_info": {
        "value": "TRK-987654321",
        "page_number": 1,
        "extraction_method_detail": "delivery_info",
        "context_snippet": "Tracking Number: TRK-987654321",
    },
    "payment_info": {
        "value": "Credit Card",
        "page_number": 1,
        "extraction_method_detail": "payment_info",
        "context_snippet": "Payment Method: Credit Card",
    },
}

SAMPLE_INVOICE_TEXT = """
INVOICE
Invoice Number: INV-2024-00789
Date: 2024-03-15
Order ID: ORD-56789

Product: Premium Widget Pro
SKU: SKU-WGT-001
Quantity: 3

Subtotal: $149.97
Tax:      $12.00
Total:    $161.97
Currency: USD

Tracking Number: TRK-987654321
Payment Method: Credit Card
"""


def make_text_pdf(text: str) -> bytes:
    """Generate in-memory PDF with sample text."""
    try:
        import pymupdf as fitz
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 100), text, fontsize=11)
        return doc.tobytes()
    except Exception:
        # Fallback raw 1-page PDF
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_stream = f"BT /F1 12 Tf 50 750 Td ({escaped}) Tj ET".encode()
        stream_len = len(content_stream)
        return (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
            + f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode()
            + content_stream
            + b"\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n300\n%%EOF\n"
        )


def make_png_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (10, 10), color=(200, 200, 200))
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Normalization Utility Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_id():
    assert normalize_id("ORD-56789") == "ORD56789"
    assert normalize_id("ord 56789") == "ORD56789"
    assert normalize_id("#ORD-56789") == "ORD56789"
    assert normalize_id("INV_2024_001") == "INV2024001"
    assert normalize_id(None) is None


def test_normalize_amount():
    assert normalize_amount("$161.97") == 161.97
    assert normalize_amount("161.97") == 161.97
    assert normalize_amount("₹1,500.00") == 1500.00
    assert normalize_amount("1500") == 1500.00
    assert normalize_amount(None) is None


def test_normalize_currency():
    assert normalize_currency("$") == "USD"
    assert normalize_currency("USD") == "USD"
    assert normalize_currency("₹") == "INR"
    assert normalize_currency("INR") == "INR"
    assert normalize_currency("€") == "EUR"
    assert normalize_currency("£") == "GBP"
    assert normalize_currency(None) is None


def test_normalize_date():
    assert normalize_date("2024-03-15") == "2024-03-15"
    assert normalize_date("15/03/2024") == "2024-03-15"
    assert normalize_date("15 Mar 2024") == "2024-03-15"
    assert normalize_date("March 15, 2024") == "2024-03-15"
    assert normalize_date(None) is None


def test_normalize_quantity():
    assert normalize_quantity("3") == 3
    assert normalize_quantity("3 items") == 3
    assert normalize_quantity("Qty: 10") == 10
    assert normalize_quantity(None) is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Message Parser Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_message_parser_order_id():
    parser = MessageParser()
    res = parser.parse("My order ID is ORD-56789 and I want a refund.")
    assert res["order_id"] == "ORD-56789"


def test_message_parser_amount_and_currency():
    parser = MessageParser()
    res = parser.parse("I was charged ₹1500 for this order.")
    assert res["amount"] == "1500"
    assert res["currency"] == "INR"


def test_message_parser_invoice_and_date():
    parser = MessageParser()
    res = parser.parse("Invoice number is INV-2024-001 dated 2024-03-15.")
    assert res["invoice_number"] == "INV-2024-001"
    assert res["date"] == "2024-03-15"


def test_message_parser_error_code():
    parser = MessageParser()
    res = parser.parse("I encountered Error Code: ERR-PAYMENT-403 when checking out.")
    assert res["error_code"] == "ERR-PAYMENT-403"


def test_message_parser_no_hallucination_empty_text():
    parser = MessageParser()
    res = parser.parse("Hello, can you help me?")
    assert res["order_id"] is None
    assert res["amount"] is None
    assert res["invoice_number"] is None
    assert res["error_code"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Matching Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_matching_order_id():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="My order number is ORD56789",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert res.status == "matched"
    assert "order_id" in res.matched_fields
    assert len(res.conflicting_fields) == 0
    assert not res.clarification_required


def test_matching_invoice_number():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="Invoice #INV-2024-00789",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert res.status == "matched"
    assert "invoice_number" in res.matched_fields
    assert not res.clarification_required


def test_matching_amount():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="The total amount paid was $161.97",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert "amount" in res.matched_fields
    assert not res.clarification_required


def test_matching_product_code():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="The SKU is SKU-WGT-001",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert "product_code" in res.matched_fields
    assert not res.clarification_required


def test_matching_quantity():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="I ordered quantity 3 items",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert "quantity" in res.matched_fields
    assert not res.clarification_required


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conflict Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_conflicting_order_id():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="My order ID is ORD-99999",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert res.status == "conflict"
    assert len(res.conflicting_fields) == 1
    conflict = res.conflicting_fields[0]
    assert conflict["field"] == "order_id"
    assert conflict["message_value"] == "ORD-99999"
    assert conflict["evidence_value"] == "ORD-56789"
    assert res.clarification_required
    assert "order ID" in res.clarification_message


def test_conflicting_invoice_number():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="Invoice number INV-2024-99999",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert res.status == "conflict"
    assert any(c["field"] == "invoice_number" for c in res.conflicting_fields)
    assert res.clarification_required


def test_conflicting_amount():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="The amount charged was $250.00",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert res.status == "conflict"
    conflict = next(c for c in res.conflicting_fields if c["field"] == "amount")
    assert conflict["message_value"] == "250.00"
    assert conflict["evidence_value"] == "161.97"
    assert res.clarification_required
    assert "amount" in res.clarification_message


def test_conflicting_error_code():
    evidence_with_err = dict(SAMPLE_EVIDENCE)
    evidence_with_err["error_code"] = {
        "value": "ERR-PAYMENT-403",
        "page_number": 1,
        "extraction_method_detail": "error_code",
    }
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="I got error code ERR-AUTH-401",
        extracted_evidence=evidence_with_err,
    )
    assert res.status == "conflict"
    assert any(c["field"] == "error_code" for c in res.conflicting_fields)
    assert res.clarification_required


# ─────────────────────────────────────────────────────────────────────────────
# 5. Field Presence: Message-Only, Evidence-Only, Missing
# ─────────────────────────────────────────────────────────────────────────────

def test_message_only_important_field_requires_clarification():
    """Customer mentions an error code, but document has no error code."""
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="I have error code E102 with this purchase",
        extracted_evidence=SAMPLE_EVIDENCE,  # has error_code: None
    )
    assert "error_code" in res.message_only_fields
    assert "error_code" in res.missing_fields
    assert res.clarification_required
    assert "error code" in res.clarification_message.lower()


def test_evidence_only_fields_reported():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="My order ID is ORD-56789",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert "order_id" in res.matched_fields
    # Other document fields should be listed as evidence_only
    assert "invoice_number" in res.evidence_only_fields
    assert "amount" in res.evidence_only_fields


def test_no_customer_message_handled():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message=None,
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert res.status == "no_message"
    assert not res.clarification_required
    assert len(res.matched_fields) == 0
    assert len(res.evidence_only_fields) > 0


def test_unrelated_customer_message_handled():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="Hello, can somebody assist me with my inquiry please?",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert res.status == "no_detectable_fields"
    assert not res.clarification_required
    assert len(res.matched_fields) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Quality-Aware Comparison
# ─────────────────────────────────────────────────────────────────────────────

def test_poor_evidence_quality_prevents_match_and_requires_clarification():
    comparator = EvidenceComparator()
    # Even if customer provides the exact order ID, poor quality evidence prevents confident match
    res = comparator.compare(
        customer_message="My order is ORD-56789",
        extracted_evidence=SAMPLE_EVIDENCE,
        evidence_quality="poor",
        quality_score=0.2,
    )
    assert res.status == "insufficient_evidence"
    assert len(res.matched_fields) == 0
    assert res.clarification_required
    assert "quality is too low" in res.clarification_message


def test_failed_evidence_quality_requires_clarification():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="My order is ORD-56789 and total was $161.97",
        extracted_evidence=SAMPLE_EVIDENCE,
        evidence_quality="failed",
        quality_score=0.0,
    )
    assert res.status == "insufficient_evidence"
    assert res.clarification_required


# ─────────────────────────────────────────────────────────────────────────────
# 7. Multiple Fields in One Message
# ─────────────────────────────────────────────────────────────────────────────

def test_multiple_fields_all_matching():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="Order ORD-56789, date is 15/03/2024 and total is $161.97 USD",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert res.status == "matched"
    assert "order_id" in res.matched_fields
    assert "amount" in res.matched_fields
    assert "date" in res.matched_fields
    assert "currency" in res.matched_fields
    assert len(res.conflicting_fields) == 0
    assert not res.clarification_required


def test_multiple_fields_one_match_one_conflict():
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="My order is ORD-56789, but I was charged $200.00",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    assert res.status == "conflict"
    assert "order_id" in res.matched_fields
    assert any(c["field"] == "amount" for c in res.conflicting_fields)
    assert res.clarification_required
    assert "amount" in res.clarification_message


# ─────────────────────────────────────────────────────────────────────────────
# 8. No Hallucination Safeguards
# ─────────────────────────────────────────────────────────────────────────────

def test_no_hallucination_empty_evidence():
    """If evidence has all None fields, customer fields must not falsely match."""
    empty_evidence = {k: None for k in SAMPLE_EVIDENCE}
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="Order ID: ORD-12345, Invoice: INV-999, Amount: $500",
        extracted_evidence=empty_evidence,
        evidence_quality="good",
    )
    assert len(res.matched_fields) == 0
    assert len(res.conflicting_fields) == 0
    assert "order_id" in res.message_only_fields
    assert res.clarification_required


def test_conflict_preserves_both_values():
    """System must preserve both values and never choose customer over document."""
    comparator = EvidenceComparator()
    res = comparator.compare(
        customer_message="My order is ORD-99999",
        extracted_evidence=SAMPLE_EVIDENCE,
    )
    conflict = res.conflicting_fields[0]
    assert conflict["message_value"] == "ORD-99999"
    assert conflict["evidence_value"] == "ORD-56789"
    # Verify original evidence in dict wasn't modified
    assert SAMPLE_EVIDENCE["order_id"]["value"] == "ORD-56789"


# ─────────────────────────────────────────────────────────────────────────────
# 9. API Integration Tests: POST /multimodal/analyze
# ─────────────────────────────────────────────────────────────────────────────

def test_api_with_matching_message():
    pdf_bytes = make_text_pdf(SAMPLE_INVOICE_TEXT)
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("invoice.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"message": "My order ID is ORD-56789 and the total is $161.97"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "comparison" in data
    comp = data["comparison"]
    assert "order_id" in comp["matched_fields"]
    assert len(comp["conflicting_fields"]) == 0
    assert not comp["clarification_required"]


def test_api_with_conflicting_message():
    pdf_bytes = make_text_pdf(SAMPLE_INVOICE_TEXT)
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("invoice.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"message": "My order ID is ORD-99999"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "comparison" in data
    comp = data["comparison"]
    assert comp["status"] == "conflict"
    assert len(comp["conflicting_fields"]) > 0
    assert comp["clarification_required"]
    assert data["clarification_required"]


def test_api_without_message_backwards_compatibility():
    """Requests without message form field must continue to succeed with valid comparison structure."""
    pdf_bytes = make_text_pdf(SAMPLE_INVOICE_TEXT)
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("invoice.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "comparison" in data
    assert data["comparison"]["status"] == "no_message"
    assert not data["comparison"]["clarification_required"]


# ─────────────────────────────────────────────────────────────────────────────
# 10. Non-Regression Tests (Phase 1, Phase 2, Task 1)
# ─────────────────────────────────────────────────────────────────────────────

def test_phase1_validation_regression_invalid_extension():
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("bad.exe", io.BytesIO(b"MZ\x90\x00" + b"\x00"*50), "application/octet-stream")},
    )
    assert response.status_code == 400


def test_phase1_validation_regression_empty_file():
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
    )
    assert response.status_code == 400


def test_phase2_extraction_regression_pdf_fields():
    pdf_bytes = make_text_pdf(SAMPLE_INVOICE_TEXT)
    response = client.post(
        "/multimodal/analyze",
        files={"file": ("invoice.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "extracted_fields" in data
    assert data["extracted_fields"]["order_id"]["value"] == "ORD-56789"
    assert data["extracted_fields"]["invoice_number"]["value"] == "INV-2024-00789"


def test_task1_ask_endpoint_regression():
    response = client.post("/ask", json={"question": "What is the return policy?"})
    assert response.status_code != 404
