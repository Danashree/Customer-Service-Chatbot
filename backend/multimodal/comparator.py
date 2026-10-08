"""
Task 2 Phase 3 – Evidence Comparison & Conflict Detection

Compares customer message inputs with Phase 2 extracted document evidence.
Normalizes values (IDs, amounts, currencies, dates, quantities) to detect matches and conflicts.
Enforces quality-awareness: poor evidence quality never yields false matches and requires clarification.
Strictly adheres to the no-hallucination principle — never guesses or auto-resolves conflicts.
"""

import re
import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .message_parser import MessageParser


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ConflictingField:
    """Preserves both customer-stated and document-extracted values when a conflict occurs."""
    field: str
    message_value: str
    evidence_value: str
    status: str = "conflicting"

    def to_dict(self) -> Dict[str, str]:
        return {
            "field": self.field,
            "message_value": self.message_value,
            "evidence_value": self.evidence_value,
            "status": self.status,
        }


@dataclass
class ComparisonResult:
    """Structured outcome of customer message vs document evidence comparison."""
    status: str                         # "matched" | "conflict" | "partial" | "insufficient_evidence" | "no_message" | "no_detectable_fields"
    matched_fields: List[str]
    conflicting_fields: List[Dict[str, str]]
    message_only_fields: List[str]
    evidence_only_fields: List[str]
    missing_fields: List[str]
    clarification_required: bool
    clarification_message: Optional[str]
    field_comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "matched_fields": self.matched_fields,
            "conflicting_fields": self.conflicting_fields,
            "message_only_fields": self.message_only_fields,
            "evidence_only_fields": self.evidence_only_fields,
            "missing_fields": self.missing_fields,
            "clarification_required": self.clarification_required,
            "clarification_message": self.clarification_message,
            "field_comparisons": self.field_comparisons,
        }


# ─── Normalization Utilities ──────────────────────────────────────────────────

_MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_CURRENCY_MAP = {
    "$": "USD",
    "USD": "USD",
    "€": "EUR",
    "EUR": "EUR",
    "£": "GBP",
    "GBP": "GBP",
    "₹": "INR",
    "INR": "INR",
    "¥": "JPY",
    "JPY": "JPY",
    "CAD": "CAD",
    "AUD": "AUD",
}


def normalize_id(val: Optional[str]) -> Optional[str]:
    """Normalize order IDs, invoice numbers, SKUs by stripping punctuation and uppercasing."""
    if not val:
        return None
    cleaned = re.sub(r'[\s\-_#:]+', '', str(val)).upper()
    return cleaned if cleaned else None


def normalize_currency(val: Optional[str]) -> Optional[str]:
    """Normalize currency symbols and codes to canonical ISO 3-letter codes."""
    if not val:
        return None
    raw = str(val).strip().upper()
    return _CURRENCY_MAP.get(raw, raw)


def normalize_amount(val: Optional[str]) -> Optional[float]:
    """Extract float numeric value from amount string."""
    if val is None:
        return None
    cleaned = re.sub(r'[^\d.]', '', str(val).replace(',', ''))
    try:
        return round(float(cleaned), 2)
    except (ValueError, TypeError):
        return None


def normalize_quantity(val: Optional[str]) -> Optional[int]:
    """Extract integer quantity."""
    if val is None:
        return None
    cleaned = re.sub(r'[^\d]', '', str(val))
    try:
        return int(cleaned)
    except (ValueError, TypeError):
        return None


def normalize_date(val: Optional[str]) -> Optional[str]:
    """
    Normalize various date formats into canonical 'YYYY-MM-DD'.
    Handles ISO (2024-03-15), EU (15/03/2024), US, and alphanumeric (15 Mar 2024, March 15, 2024).
    """
    if not val:
        return None
    s = str(val).strip()

    # ISO: 2024-03-15 or 2024/03/15
    m = re.match(r'^(20\d{2})[\/\-](\d{1,2})[\/\-](\d{1,2})$', s)
    if m:
        y, month, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{month:02d}-{d:02d}"

    # DD/MM/YYYY or DD-MM-YYYY
    m = re.match(r'^(\d{1,2})[\/\-](\d{1,2})[\/\-](20\d{2})$', s)
    if m:
        p1, p2, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # If p1 > 12, it must be DD/MM/YYYY
        if p1 > 12:
            d, month = p1, p2
        else:
            # Default to DD/MM/YYYY
            d, month = p1, p2
        return f"{y:04d}-{month:02d}-{d:02d}"

    # 15 Mar 2024 or 15 March 2024
    m = re.match(r'^(\d{1,2})\s+([A-Za-z]+)\.?\s+(20\d{2})$', s)
    if m:
        d, mon_str, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        mon_num = _MONTH_NAMES.get(mon_str[:3])
        if mon_num:
            return f"{y:04d}-{mon_num:02d}-{d:02d}"

    # March 15, 2024 or Mar 15 2024
    m = re.match(r'^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(20\d{2})$', s)
    if m:
        mon_str, d, y = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        mon_num = _MONTH_NAMES.get(mon_str[:3])
        if mon_num:
            return f"{y:04d}-{mon_num:02d}-{d:02d}"

    # Fallback: cleaned lowercase string
    return s.lower()


def normalize_text(val: Optional[str]) -> Optional[str]:
    """Normalize generic text by lowercasing and condensing whitespace."""
    if not val:
        return None
    return re.sub(r'\s+', ' ', str(val)).strip().lower()


# ─── Evidence Comparator ──────────────────────────────────────────────────────

class EvidenceComparator:
    """
    Compares customer-provided message fields against Phase 2 document evidence fields.
    Accurately classifies field matches, conflicts, and missing data with quality awareness.
    """

    COMPARED_FIELDS = [
        "order_id",
        "invoice_number",
        "date",
        "amount",
        "currency",
        "product_name",
        "product_code",
        "quantity",
        "error_code",
        "delivery_info",
        "payment_info",
    ]

    # Fields whose absence when stated by user warrants immediate clarification
    HIGH_IMPORTANCE_FIELDS = {"order_id", "invoice_number", "amount", "error_code"}

    def __init__(self, message_parser: Optional[MessageParser] = None):
        self.parser = message_parser or MessageParser()

    def compare(
        self,
        customer_message: Optional[str],
        extracted_evidence: Dict[str, Any],
        evidence_quality: str = "good",
        quality_score: float = 1.0,
    ) -> ComparisonResult:
        """
        Execute comparison between customer message and document evidence.

        Args:
            customer_message: Natural language text from the customer.
            extracted_evidence: Dict of fields from Phase 2 DocumentExtractor.
            evidence_quality: "good" | "acceptable" | "poor" | "failed".
            quality_score: 0.0 to 1.0 numerical quality rating.
        """
        # 1. Check if customer message is present
        if customer_message is None or not customer_message.strip():
            evidence_only = [
                f for f in self.COMPARED_FIELDS
                if self._get_evidence_value(extracted_evidence, f) is not None
            ]
            return ComparisonResult(
                status="no_message",
                matched_fields=[],
                conflicting_fields=[],
                message_only_fields=[],
                evidence_only_fields=evidence_only,
                missing_fields=[],
                clarification_required=False,
                clarification_message=None,
                field_comparisons={},
            )

        # 2. Parse customer message
        msg_fields = self.parser.parse(customer_message)
        active_msg_fields = {k: v for k, v in msg_fields.items() if v is not None}

        # If customer message has no detectable fields
        if not active_msg_fields:
            evidence_only = [
                f for f in self.COMPARED_FIELDS
                if self._get_evidence_value(extracted_evidence, f) is not None
            ]
            return ComparisonResult(
                status="no_detectable_fields",
                matched_fields=[],
                conflicting_fields=[],
                message_only_fields=[],
                evidence_only_fields=evidence_only,
                missing_fields=[],
                clarification_required=False,
                clarification_message=None,
                field_comparisons={},
            )

        # 3. Quality-aware guard: if document quality is poor or failed
        is_poor_quality = (
            evidence_quality.lower() in ("poor", "failed")
            or quality_score < 0.45
        )
        if is_poor_quality:
            msg_only = list(active_msg_fields.keys())
            return ComparisonResult(
                status="insufficient_evidence",
                matched_fields=[],
                conflicting_fields=[],
                message_only_fields=msg_only,
                evidence_only_fields=[],
                missing_fields=msg_only,
                clarification_required=True,
                clarification_message=(
                    "The uploaded document quality is too low to reliably verify your information. "
                    "Please upload a clearer image or a text-based PDF."
                ),
                field_comparisons={
                    f: {"status": "not_comparable", "reason": "poor_evidence_quality"}
                    for f in active_msg_fields
                },
            )

        # 4. Perform field-by-field comparison
        matched_fields: List[str] = []
        conflicting_fields: List[Dict[str, str]] = []
        message_only_fields: List[str] = []
        evidence_only_fields: List[str] = []
        missing_fields: List[str] = []
        field_comparisons: Dict[str, Dict[str, Any]] = {}

        for field_name in self.COMPARED_FIELDS:
            msg_val = active_msg_fields.get(field_name)
            ev_val = self._get_evidence_value(extracted_evidence, field_name)

            if msg_val is not None and ev_val is not None:
                is_match = self._values_match(field_name, msg_val, ev_val)
                if is_match:
                    matched_fields.append(field_name)
                    field_comparisons[field_name] = {
                        "status": "matched",
                        "message_value": msg_val,
                        "evidence_value": ev_val,
                    }
                else:
                    conflict_item = ConflictingField(
                        field=field_name,
                        message_value=msg_val,
                        evidence_value=ev_val,
                    )
                    conflicting_fields.append(conflict_item.to_dict())
                    field_comparisons[field_name] = conflict_item.to_dict()

            elif msg_val is not None and ev_val is None:
                message_only_fields.append(field_name)
                missing_fields.append(field_name)
                field_comparisons[field_name] = {
                    "status": "message_only",
                    "message_value": msg_val,
                    "evidence_value": None,
                }

            elif msg_val is None and ev_val is not None:
                evidence_only_fields.append(field_name)
                field_comparisons[field_name] = {
                    "status": "evidence_only",
                    "message_value": None,
                    "evidence_value": ev_val,
                }

        # 5. Determine overall comparison status and clarification requirements
        has_conflicts = len(conflicting_fields) > 0
        has_matches = len(matched_fields) > 0
        has_missing_important = any(f in self.HIGH_IMPORTANCE_FIELDS for f in message_only_fields)

        clarification_required = False
        clarification_message = None
        status = "matched"

        if has_conflicts:
            status = "conflict"
            clarification_required = True
            clarification_message = self._generate_conflict_clarification(conflicting_fields)
        elif has_missing_important:
            status = "partial"
            clarification_required = True
            clarification_message = self._generate_missing_clarification(message_only_fields)
        elif has_matches and message_only_fields:
            status = "partial"
            clarification_required = False
        elif has_matches and not message_only_fields:
            status = "matched"
            clarification_required = False
        else:
            # Only message-only fields were found, but none matched or conflicted
            status = "insufficient_evidence"
            clarification_required = True
            clarification_message = (
                "None of the details mentioned in your message could be identified in the uploaded document. "
                "Please verify you have uploaded the correct document."
            )

        return ComparisonResult(
            status=status,
            matched_fields=matched_fields,
            conflicting_fields=conflicting_fields,
            message_only_fields=message_only_fields,
            evidence_only_fields=evidence_only_fields,
            missing_fields=missing_fields,
            clarification_required=clarification_required,
            clarification_message=clarification_message,
            field_comparisons=field_comparisons,
        )

    # ─── Helper Methods ───────────────────────────────────────────────────────

    def _values_match(self, field_name: str, msg_val: str, ev_val: str) -> bool:
        """Compare two values under appropriate domain normalization rules."""
        if field_name in ("order_id", "invoice_number", "product_code"):
            norm_msg = normalize_id(msg_val)
            norm_ev = normalize_id(ev_val)
            return bool(norm_msg and norm_ev and norm_msg == norm_ev)

        elif field_name == "amount":
            amt_msg = normalize_amount(msg_val)
            amt_ev = normalize_amount(ev_val)
            if amt_msg is not None and amt_ev is not None:
                return abs(amt_msg - amt_ev) < 0.01
            return False

        elif field_name == "currency":
            c_msg = normalize_currency(msg_val)
            c_ev = normalize_currency(ev_val)
            return bool(c_msg and c_ev and c_msg == c_ev)

        elif field_name == "date":
            d_msg = normalize_date(msg_val)
            d_ev = normalize_date(ev_val)
            return bool(d_msg and d_ev and d_msg == d_ev)

        elif field_name == "quantity":
            q_msg = normalize_quantity(msg_val)
            q_ev = normalize_quantity(ev_val)
            return (q_msg is not None and q_ev is not None and q_msg == q_ev)

        elif field_name == "error_code":
            code_msg = normalize_id(msg_val)
            code_ev = normalize_id(ev_val)
            return bool(code_msg and code_ev and code_msg == code_ev)

        elif field_name == "product_name":
            p_msg = normalize_text(msg_val)
            p_ev = normalize_text(ev_val)
            if p_msg and p_ev:
                return (p_msg == p_ev) or (p_msg in p_ev) or (p_ev in p_msg)
            return False

        else:
            # Generic text fields (delivery_info, payment_info)
            t_msg = normalize_text(msg_val)
            t_ev = normalize_text(ev_val)
            if t_msg and t_ev:
                return (t_msg == t_ev) or (t_msg in t_ev) or (t_ev in t_msg)
            return False

    @staticmethod
    def _get_evidence_value(extracted_evidence: Dict[str, Any], field_name: str) -> Optional[str]:
        """Safely extract string value from Phase 2 evidence structure."""
        if not extracted_evidence:
            return None
        item = extracted_evidence.get(field_name)
        if item is None:
            return None
        if isinstance(item, dict):
            val = item.get("value")
            return str(val) if val is not None else None
        return str(item)

    @staticmethod
    def _generate_conflict_clarification(conflicts: List[Dict[str, str]]) -> str:
        """Create a targeted, user-friendly clarification request when conflicts occur."""
        clauses = []
        for c in conflicts:
            f = c["field"]
            m_val = c["message_value"]
            e_val = c["evidence_value"]
            if f == "order_id":
                clauses.append(f"The order ID in your message ({m_val}) does not match the uploaded document ({e_val}).")
            elif f == "invoice_number":
                clauses.append(f"The invoice number in your message ({m_val}) does not match the uploaded document ({e_val}).")
            elif f == "amount":
                clauses.append(f"The amount in your message ({m_val}) differs from the amount shown on the uploaded document ({e_val}).")
            elif f == "error_code":
                clauses.append(f"The error code reported ({m_val}) does not match the error code in the uploaded document ({e_val}).")
            else:
                clauses.append(f"The {f.replace('_', ' ')} in your message ({m_val}) differs from the uploaded document ({e_val}).")

        clauses.append("Please confirm the correct details.")
        return " ".join(clauses)

    @staticmethod
    def _generate_missing_clarification(missing_fields: List[str]) -> str:
        """Create clarification request when an important stated field is missing in evidence."""
        readable = [f.replace("_", " ") for f in missing_fields if f in EvidenceComparator.HIGH_IMPORTANCE_FIELDS]
        if not readable:
            readable = [f.replace("_", " ") for f in missing_fields]
        joined = ", ".join(readable)
        return (
            f"The uploaded document does not show the {joined} mentioned in your message. "
            f"Please provide a document clearly displaying this information."
        )
