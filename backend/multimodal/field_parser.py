"""
Task 2 Phase 2 – Structured Field Parser

Extracts structured evidence fields from raw OCR/PDF text using regex patterns.
NEVER invents or guesses values — only returns what is explicitly present in the text.

Extracted fields:
  order_id, invoice_number, date, amount, currency,
  product_name, product_code, quantity, error_code,
  delivery_info, payment_info
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# ─── Field-level result ────────────────────────────────────────────────────────

@dataclass
class FieldEvidence:
    """Holds a single extracted field value together with provenance metadata."""
    value: Optional[str]          # None = not found
    page_number: Optional[int]    # 1-indexed; None for single-page images
    matched_pattern: str          # the regex label that matched
    raw_snippet: Optional[str]    # short surrounding context (≤80 chars)


NOT_DETECTED = None  # sentinel for "not found in document"


# ─── Compiled regex patterns ───────────────────────────────────────────────────

# Order ID
_ORDER_ID_PATTERNS = [
    re.compile(r'\bOrder\s*(?:ID|Number|#|No\.?)\s*[:\-]?\s*([A-Z0-9\-]{3,30})', re.IGNORECASE),
    re.compile(r'\bORD[-#]?([A-Z0-9]{3,20})', re.IGNORECASE),
    re.compile(r'#([A-Z]{0,3}\d{4,15})', re.IGNORECASE),
]

# Invoice number
_INVOICE_PATTERNS = [
    re.compile(r'\bInvoice\s*(?:Number|No\.?|#|ID)?\s*[:\-]\s*([A-Z0-9\-]{3,30})', re.IGNORECASE),
    re.compile(r'\bInvoice\s+(?:Number|No\.?|#|ID)\s+([A-Z0-9\-]{3,30})', re.IGNORECASE),
    re.compile(r'\bINV[-#]?([A-Z0-9]{3,20})', re.IGNORECASE),
]

# Date (multiple formats)
_DATE_PATTERNS = [
    # ISO: 2024-01-15
    re.compile(r'\b(20\d{2}[\/\-]\d{1,2}[\/\-]\d{1,2})\b'),
    # US: 01/15/2024 or 01-15-2024
    re.compile(r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]20\d{2})\b'),
    # 15 Jan 2024 or January 15, 2024
    re.compile(r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+20\d{2})\b', re.IGNORECASE),
    re.compile(r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+20\d{2})\b', re.IGNORECASE),
    # With label
    re.compile(r'\bDate\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]20\d{2})', re.IGNORECASE),
    re.compile(r'\bDate\s*[:\-]?\s*(20\d{2}[\/\-]\d{1,2}[\/\-]\d{1,2})', re.IGNORECASE),
]

# Amount (prioritizes Grand Total and Total over Subtotal)
_AMOUNT_PATTERNS = [
    re.compile(r'\b(?:Grand Total|Total|Total Due|Amount Due|Paid|Due)\b\s*[:\-]?\s*(?:[A-Z]{3}\s*)?(\$?€?£?₹?[\d,]+(?:\.\d{1,2})?)', re.IGNORECASE),
    re.compile(r'\b(?:Amount|Price|Subtotal)\b\s*[:\-]?\s*(?:[A-Z]{3}\s*)?(\$?€?£?₹?[\d,]+(?:\.\d{1,2})?)', re.IGNORECASE),
    re.compile(r'(\$|€|£|₹|USD|EUR|GBP|INR)\s*([\d,]+(?:\.\d{1,2})?)'),
    re.compile(r'([\d,]+\.\d{2})\s*(?:\$|€|£|₹|USD|EUR|GBP|INR)'),
]

# Currency (standalone)
_CURRENCY_PATTERNS = [
    re.compile(r'\b(USD|EUR|GBP|CAD|AUD|INR|JPY|CHF|CNY|MXN)\b'),
    re.compile(r'Currency\s*[:\-]?\s*([A-Z]{3})', re.IGNORECASE),
    re.compile(r'(\$|€|£|₹|¥)'),  # symbols fallback
]

_CURRENCY_SYMBOL_MAP = {'$': 'USD', '€': 'EUR', '£': 'GBP', '₹': 'INR', '¥': 'JPY'}

# Product name (contextual)
_PRODUCT_NAME_PATTERNS = [
    re.compile(r'(?:Product|Item|Description|Article)\s*[:\-]?\s*(.{3,60}?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'(?:Purchased|Ordered)\s*[:\-]?\s*(.{3,60}?)(?:\n|$)', re.IGNORECASE),
]

# Product code / SKU
_PRODUCT_CODE_PATTERNS = [
    re.compile(r'\bSKU\s*[:\-]?\s*([A-Z0-9\-]{3,20})', re.IGNORECASE),
    re.compile(r'\b(?:Product\s*Code|Part\s*No\.?|Model)\s*[:\-]?\s*([A-Z0-9\-]{3,20})', re.IGNORECASE),
    re.compile(r'\bPROD[-#]?([A-Z0-9]{3,15})', re.IGNORECASE),
]

# Quantity
_QUANTITY_PATTERNS = [
    re.compile(r'\b(?:Qty|Quantity|QTY|Units?)\s*[:\-]?\s*(\d+)', re.IGNORECASE),
    re.compile(r'\bx\s*(\d+)\b'),  # e.g. "x 2"
]

# Error code — require explicit separator or established prefix to reduce false positives
_ERROR_CODE_PATTERNS = [
    # "Error Code: ERR-403" or "Error Code: E40004" — must have colon/dash separator
    re.compile(r'\bError\s+Code\s*[:\-]\s*([A-Z0-9_\-]{2,20})', re.IGNORECASE),
    # "Error: ERR-403" — 'Error' followed by : and a code-like value (uppercase or digit-prefixed)
    re.compile(r'\bError\s*[:\-]\s*([A-Z][A-Z0-9_\-]{2,20}|ERR[A-Z0-9_\-]+|\d{3,10})', re.IGNORECASE),
    # ERR- prefix (common pattern)
    re.compile(r'\bERR[-_]([A-Z0-9]{2,15})\b', re.IGNORECASE),
    # Standalone numeric codes like E40004, E403
    re.compile(r'\b(E\d{3,6})\b'),
    # Hex codes like 0x80070005
    re.compile(r'\b(0x[0-9A-Fa-f]{2,8})\b'),
]

# Delivery info
_DELIVERY_PATTERNS = [
    re.compile(r'(?:Tracking\s*(?:Number|No\.?|ID)?|Shipment\s*ID|Delivery\s*(?:Date|Address|Status)|Ship\s*To)\s*[:\-]?\s*(.{3,80}?)(?:\n|$)', re.IGNORECASE),
]

# Payment info (masked)
_PAYMENT_PATTERNS = [
    re.compile(r'(?:Payment\s*(?:Method|Mode|Status)?|Paid\s*(?:Via|By|With)?|Card\s*(?:Type|Number)?)\s*[:\-]?\s*(.{3,60}?)(?:\n|$)', re.IGNORECASE),
]


# ─── Parser class ──────────────────────────────────────────────────────────────

class FieldParser:
    """
    Applies regex patterns to OCR/PDF text and extracts structured evidence fields.
    Returns None for any field not found — never invents values.
    """

    def parse(
        self,
        text_by_page: Dict[int, str],
    ) -> Dict[str, Any]:
        """
        Parse all target fields from a page-keyed text dict.

        Args:
            text_by_page: {page_number: text_string} (page_number 1-indexed)

        Returns:
            Dict mapping field names to FieldEvidence objects (or None if not found).
        """
        # Combine all text for fields that don't need page attribution
        full_text = "\n".join(text_by_page.values())

        results: Dict[str, Any] = {}

        results["order_id"] = self._first_match_across_pages(
            text_by_page, _ORDER_ID_PATTERNS, "order_id"
        )
        results["invoice_number"] = self._first_match_across_pages(
            text_by_page, _INVOICE_PATTERNS, "invoice_number"
        )
        results["date"] = self._first_match_across_pages(
            text_by_page, _DATE_PATTERNS, "date"
        )
        results["amount"] = self._extract_amount(text_by_page)
        results["currency"] = self._extract_currency(full_text)
        results["product_name"] = self._first_match_across_pages(
            text_by_page, _PRODUCT_NAME_PATTERNS, "product_name", strip=True
        )
        results["product_code"] = self._first_match_across_pages(
            text_by_page, _PRODUCT_CODE_PATTERNS, "product_code"
        )
        results["quantity"] = self._first_match_across_pages(
            text_by_page, _QUANTITY_PATTERNS, "quantity"
        )
        results["error_code"] = self._first_match_across_pages(
            text_by_page, _ERROR_CODE_PATTERNS, "error_code"
        )
        results["delivery_info"] = self._first_match_across_pages(
            text_by_page, _DELIVERY_PATTERNS, "delivery_info", strip=True
        )
        results["payment_info"] = self._extract_payment_info(text_by_page)

        return results

    # ─── Internal helpers ──────────────────────────────────────────────────

    def _first_match_across_pages(
        self,
        text_by_page: Dict[int, str],
        patterns: List[re.Pattern],
        field_name: str,
        strip: bool = False,
    ) -> Optional[FieldEvidence]:
        """Try each pattern on each page; return first match with page provenance."""
        for page_num, text in sorted(text_by_page.items()):
            for pat in patterns:
                m = pat.search(text)
                if m:
                    value = m.group(1).strip() if strip else m.group(1)
                    value = value.strip()
                    if not value:
                        continue
                    if field_name == "invoice_number" and value.lower() in ("invoice", "number", "id", "no", "date"):
                        continue
                    snippet = self._snippet(text, m.start(), m.end())
                    return FieldEvidence(
                        value=value,
                        page_number=page_num,
                        matched_pattern=field_name,
                        raw_snippet=snippet,
                    )
        return NOT_DETECTED

    def _extract_amount(self, text_by_page: Dict[int, str]) -> Optional[FieldEvidence]:
        for page_num, text in sorted(text_by_page.items()):
            for pat in _AMOUNT_PATTERNS:
                m = pat.search(text)
                if m:
                    # Pick the group that looks like a number
                    groups = [g for g in m.groups() if g and re.search(r'\d', g)]
                    if not groups:
                        continue
                    # Prefer the group that looks like a pure number (not currency symbol)
                    numeric_groups = [g for g in groups if re.match(r'^[\d,\.]+$', g)]
                    value = (numeric_groups[0] if numeric_groups else groups[-1]).strip()
                    snippet = self._snippet(text, m.start(), m.end())
                    return FieldEvidence(
                        value=value,
                        page_number=page_num,
                        matched_pattern="amount",
                        raw_snippet=snippet,
                    )
        return NOT_DETECTED

    def _extract_currency(self, full_text: str) -> Optional[FieldEvidence]:
        for pat in _CURRENCY_PATTERNS:
            m = pat.search(full_text)
            if m:
                raw = m.group(1).strip()
                # Map symbol to ISO code
                value = _CURRENCY_SYMBOL_MAP.get(raw, raw.upper())
                snippet = self._snippet(full_text, m.start(), m.end())
                return FieldEvidence(
                    value=value,
                    page_number=None,
                    matched_pattern="currency",
                    raw_snippet=snippet,
                )
        return NOT_DETECTED

    def _extract_payment_info(self, text_by_page: Dict[int, str]) -> Optional[FieldEvidence]:
        """Extract payment info and mask card numbers / account details."""
        evidence = self._first_match_across_pages(
            text_by_page, _PAYMENT_PATTERNS, "payment_info", strip=True
        )
        if evidence and evidence.value:
            # Mask partial card numbers like 1234 **** **** 5678 → [CARD_REDACTED]
            masked = re.sub(r'\b\d{4}[\s\-]?\*{4}[\s\-]?\*{4}[\s\-]?\d{4}\b', '[CARD_REDACTED]', evidence.value)
            masked = re.sub(r'\b\d{13,19}\b', '[CARD_REDACTED]', masked)
            evidence.value = masked
        return evidence

    @staticmethod
    def _snippet(text: str, start: int, end: int, context: int = 40) -> str:
        """Return up to `context` chars before and after the match."""
        s = max(0, start - context)
        e = min(len(text), end + context)
        raw = text[s:e].replace('\n', ' ').strip()
        return raw[:160]  # hard cap

    def to_dict(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert FieldEvidence objects to JSON-serialisable dicts."""
        out = {}
        for key, evidence in parsed.items():
            if evidence is None:
                out[key] = None
            else:
                out[key] = {
                    "value": evidence.value,
                    "page_number": evidence.page_number,
                    "extraction_method_detail": evidence.matched_pattern,
                    "context_snippet": evidence.raw_snippet,
                }
        return out
