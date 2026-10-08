"""
Task 2 Phase 3 – Customer Message Field Parser

Extracts structured domain fields from a customer's conversational message.
Only extracts fields that are EXPLICITLY present in the text.
Never guesses, assumes, or invents missing fields — missing fields evaluate to None.
"""

import re
from typing import Optional, Dict, Any, List


# Map symbols to ISO currency codes
_CURRENCY_SYMBOL_MAP = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₹": "INR",
    "¥": "JPY",
}

# Regex patterns tailored for natural conversational messages
_ORDER_ID_PATTERNS = [
    re.compile(r'\b(?:My\s+)?Order\s*(?:ID|Number|#|No\.?)?\s*(?:is|=|:|-|\b)\s*([A-Z0-9\-]{3,30})\b', re.IGNORECASE),
    re.compile(r'\bORD[-#]?([A-Z0-9]{3,20})\b', re.IGNORECASE),
    re.compile(r'#([A-Z]{0,3}\d{4,15})\b', re.IGNORECASE),
]

_INVOICE_PATTERNS = [
    re.compile(r'\b(?:My\s+)?Invoice\s*(?:Number|No\.?|#|ID)?\s*(?:is|=|:|-|\b)\s*([A-Z0-9\-]{3,30})\b', re.IGNORECASE),
    re.compile(r'\bINV[-#]?([A-Z0-9]{3,20})\b', re.IGNORECASE),
]

_DATE_PATTERNS = [
    re.compile(r'\b(20\d{2}[\/\-]\d{1,2}[\/\-]\d{1,2})\b'),
    re.compile(r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]20\d{2})\b'),
    re.compile(r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+20\d{2})\b', re.IGNORECASE),
    re.compile(r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+20\d{2})\b', re.IGNORECASE),
    re.compile(r'\bDate\s*(?:is|=|:|-)\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]20\d{2})', re.IGNORECASE),
    re.compile(r'\bDate\s*(?:is|=|:|-)\s*(20\d{2}[\/\-]\d{1,2}[\/\-]\d{1,2})', re.IGNORECASE),
]

_AMOUNT_PATTERNS = [
    re.compile(r'(?:Total|Amount|Due|Paid|Price|Subtotal|Grand Total|Charged)\s*(?:is|was|=|:|-)?\s*(?:[A-Z]{3}\s*)?(\$?€?£?₹?[\d,]+(?:\.\d{1,2})?)', re.IGNORECASE),
    re.compile(r'(\$|€|£|₹|USD|EUR|GBP|INR)\s*([\d,]+(?:\.\d{1,2})?)', re.IGNORECASE),
    re.compile(r'([\d,]+\.\d{2})\s*(?:\$|€|£|₹|USD|EUR|GBP|INR)', re.IGNORECASE),
]

_CURRENCY_PATTERNS = [
    re.compile(r'\b(USD|EUR|GBP|CAD|AUD|INR|JPY|CHF|CNY|MXN)\b', re.IGNORECASE),
    re.compile(r'Currency\s*(?:is|=|:|-)?\s*([A-Z]{3})', re.IGNORECASE),
    re.compile(r'(\$|€|£|₹|¥)'),
]

_PRODUCT_NAME_PATTERNS = [
    re.compile(r'(?:Product|Item|Article)\s*(?:name)?\s*(?:is|=|:|-)\s*([^,.\n]+?)(?:,|\.|\n|and\b|with\b|for\b|$)', re.IGNORECASE),
    re.compile(r'(?:Purchased|Ordered|Bought)\s*(?:a|an|the)?\s*([^,.\n]+?)(?:,|\.|\n|and\b|with\b|for\b|$)', re.IGNORECASE),
]

_PRODUCT_CODE_PATTERNS = [
    re.compile(r'\bSKU\s*(?:is|=|:|-)?\s*([A-Z0-9\-]{3,20})', re.IGNORECASE),
    re.compile(r'\b(?:Product\s*Code|Part\s*No\.?|Model)\s*(?:is|=|:|-)?\s*([A-Z0-9\-]{3,20})', re.IGNORECASE),
    re.compile(r'\bPROD[-#]?([A-Z0-9]{3,15})\b', re.IGNORECASE),
]

_QUANTITY_PATTERNS = [
    re.compile(r'\b(?:Qty|Quantity|QTY)\s*(?:is|=|:|-)?\s*(\d+)', re.IGNORECASE),
    re.compile(r'(?:ordered|bought|purchased|have|got)\s+(\d+)\s+(?:items?|units?|pieces?|products?)', re.IGNORECASE),
    re.compile(r'\b(\d+)\s+(?:items?|units?|pieces?)\b', re.IGNORECASE),
    re.compile(r'\bx\s*(\d+)\b', re.IGNORECASE),
]

_ERROR_CODE_PATTERNS = [
    re.compile(r'\bError\s+Code\s*(?:is|=|:|-)?\s*([A-Z0-9_\-]{2,20})', re.IGNORECASE),
    re.compile(r'\bError\s*(?:is|=|:|-)\s*([A-Z][A-Z0-9_\-]{2,20}|ERR[A-Z0-9_\-]+|\d{3,10})', re.IGNORECASE),
    re.compile(r'\bERR[-_]([A-Z0-9]{2,15})\b', re.IGNORECASE),
    re.compile(r'\b(E\d{3,6})\b'),
    re.compile(r'\b(0x[0-9A-Fa-f]{2,8})\b'),
]

_DELIVERY_PATTERNS = [
    re.compile(r'(?:Tracking\s*(?:Number|No\.?|ID)?|Shipment\s*ID|Delivery\s*(?:Date|Address|Status)|Ship\s*To)\s*(?:is|=|:|-)?\s*([^,.\n]+?)(?:,|\.|\n|and\b|$)', re.IGNORECASE),
]

_PAYMENT_PATTERNS = [
    re.compile(r'(?:Payment\s*(?:Method|Mode|Status)?)\s*(?:is|=|:|-)\s*([^,.\n]+?)(?:,|\.|\n|and\b|$)', re.IGNORECASE),
    re.compile(r'Paid\s+(?:via|by|with)\s+([^,.\n]+?)(?:,|\.|\n|and\b|$)', re.IGNORECASE),
    re.compile(r'Card\s*(?:type)?\s*(?:is|=|:|-)\s*([^,.\n]+?)(?:,|\.|\n|and\b|$)', re.IGNORECASE),
]


class MessageParser:
    """
    Parses conversational customer messages for explicit customer service evidence fields.
    Never hallucinates or invents values.
    """

    def parse(self, text: Optional[str]) -> Dict[str, Optional[str]]:
        """
        Parse customer message into a dictionary of field names to extracted string values (or None).
        """
        if not text or not text.strip():
            return self._empty_result()

        cleaned_text = text.strip()

        order_id = self._extract_order_id(cleaned_text)
        invoice_number = self._first_match(cleaned_text, _INVOICE_PATTERNS)
        date_val = self._first_match(cleaned_text, _DATE_PATTERNS)
        amount_val = self._extract_amount(cleaned_text)
        currency_val = self._extract_currency(cleaned_text)
        product_name = self._first_match(cleaned_text, _PRODUCT_NAME_PATTERNS, strip=True)
        product_code = self._first_match(cleaned_text, _PRODUCT_CODE_PATTERNS)
        quantity_val = self._first_match(cleaned_text, _QUANTITY_PATTERNS)
        error_code = self._first_match(cleaned_text, _ERROR_CODE_PATTERNS)
        delivery_info = self._first_match(cleaned_text, _DELIVERY_PATTERNS, strip=True)
        payment_info = self._first_match(cleaned_text, _PAYMENT_PATTERNS, strip=True)

        # Sanity checks to prevent false positives:
        if product_name:
            p_strip = product_name.strip()
            if len(p_strip) < 2 or re.match(r'^(?:qty|quantity|\d+)', p_strip, re.IGNORECASE):
                product_name = None

        return {
            "order_id": order_id,
            "invoice_number": invoice_number,
            "date": date_val,
            "amount": amount_val,
            "currency": currency_val,
            "product_name": product_name,
            "product_code": product_code,
            "quantity": quantity_val,
            "error_code": error_code,
            "delivery_info": delivery_info,
            "payment_info": payment_info,
        }

    def _extract_order_id(self, text: str) -> Optional[str]:
        for pat in _ORDER_ID_PATTERNS:
            m = pat.search(text)
            if m:
                val = m.group(1).strip()
                # If matched via generic pattern, verify it has digits or recognized prefix
                if re.search(r'\d', val) or val.upper().startswith("ORD"):
                    return val
        return None

    def _extract_amount(self, text: str) -> Optional[str]:
        for pat in _AMOUNT_PATTERNS:
            m = pat.search(text)
            if m:
                groups = [g for g in m.groups() if g and re.search(r'\d', g)]
                if groups:
                    numeric_groups = [g for g in groups if re.match(r'^[\d,\.]+$', g)]
                    raw_val = (numeric_groups[0] if numeric_groups else groups[-1]).strip()
                    val = re.sub(r'^[^\d]+', '', raw_val)
                    val = re.sub(r'[^\d.]+$', '', val)
                    if val:
                        return val
        return None

    def _extract_currency(self, text: str) -> Optional[str]:
        for pat in _CURRENCY_PATTERNS:
            m = pat.search(text)
            if m:
                raw = m.group(1).strip()
                return _CURRENCY_SYMBOL_MAP.get(raw, raw.upper())
        return None

    def _first_match(
        self, text: str, patterns: List[re.Pattern], strip: bool = False
    ) -> Optional[str]:
        for pat in patterns:
            m = pat.search(text)
            if m:
                val = m.group(1).strip() if strip else m.group(1)
                val = val.strip()
                if val:
                    return val
        return None

    @staticmethod
    def _empty_result() -> Dict[str, Optional[str]]:
        return {
            "order_id": None,
            "invoice_number": None,
            "date": None,
            "amount": None,
            "currency": None,
            "product_name": None,
            "product_code": None,
            "quantity": None,
            "error_code": None,
            "delivery_info": None,
            "payment_info": None,
        }
