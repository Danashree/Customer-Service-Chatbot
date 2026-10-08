"""
Task 3 Phase 1: Mandatory Field Validator.
Validates extracted ticket fields, identifies missing information,
and formats precise, non-redundant clarification requests.

missing_fields uses machine-readable keys ("order_id", "contact_info", etc.).
clarification_request converts those keys to human-readable prose.
"""

from typing import Dict, List, Optional
from .models import TicketValidationResult, TicketBase


class MandatoryFieldValidator:
    """
    Validates ticket fields against mandatory criteria for online-course support.

    Configurable mandatory fields (machine-readable keys):
      - "issue"        — what problem the learner is facing (mandatory for all tickets)
      - "contact_info" — email or phone (at least one required)
      - "course_name"  — required to locate learner's enrollment
      - "order_id"     — required for billing, access, and payment disputes
      - "evidence"     — required specifically for payment disputes / double-charge claims
    """

    DEFAULT_MANDATORY_FIELDS = ["issue", "contact_info", "course_name", "order_id"]

    # Human-readable labels used only in clarification_request text
    _FIELD_LABELS: Dict[str, str] = {
        "issue": "issue description",
        "contact_info": "contact email or phone number",
        "course_name": "course name",
        "order_id": "order or reference ID",
        "evidence": "payment screenshot or receipt",
    }

    def __init__(self, mandatory_fields: Optional[List[str]] = None):
        self.mandatory_fields = mandatory_fields or list(self.DEFAULT_MANDATORY_FIELDS)

    def validate(self, ticket: TicketBase) -> TicketValidationResult:
        """Return a TicketValidationResult with machine-readable missing_fields keys."""
        missing: List[str] = []

        # 1. Issue check
        if "issue" in self.mandatory_fields:
            if not ticket.issue or not str(ticket.issue).strip():
                missing.append("issue")

        # 2. Contact check (email OR phone is sufficient)
        if "contact_info" in self.mandatory_fields:
            has_email = bool(ticket.contact_email and str(ticket.contact_email).strip())
            has_phone = bool(ticket.contact_phone and str(ticket.contact_phone).strip())
            if not (has_email or has_phone):
                missing.append("contact_info")

        # 3. Course name check
        if "course_name" in self.mandatory_fields:
            if not ticket.course_name or not str(ticket.course_name).strip():
                missing.append("course_name")

        # 4. Order ID check
        if "order_id" in self.mandatory_fields:
            if not ticket.order_id or not str(ticket.order_id).strip():
                missing.append("order_id")

        # 5. Evidence check (conditional: required for payment disputes / double charges)
        if "evidence" in self.mandatory_fields or self._is_evidence_required_for_issue(ticket.issue):
            if not ticket.evidence or not str(ticket.evidence).strip():
                if "evidence" not in missing:
                    missing.append("evidence")

        is_complete = len(missing) == 0
        clarification_request = (
            self._generate_clarification_request(missing) if not is_complete else None
        )

        return TicketValidationResult(
            is_complete=is_complete,
            missing_fields=missing,
            clarification_request=clarification_request,
        )

    def _is_evidence_required_for_issue(self, issue: Optional[str]) -> bool:
        if not issue:
            return False
        issue_lower = issue.lower()
        keywords = [
            "charged twice",
            "double payment",
            "payment dispute",
            "money deducted",
            "failed payment",
        ]
        return any(kw in issue_lower for kw in keywords)

    def _generate_clarification_request(self, missing_fields: List[str]) -> str:
        """
        Build a focused human-readable message asking for ONLY the missing fields.
        Converts machine-readable keys to friendly labels.
        """
        if not missing_fields:
            return ""

        labels = [self._FIELD_LABELS.get(f, f) for f in missing_fields]

        if len(labels) == 1:
            fields_str = labels[0]
        elif len(labels) == 2:
            fields_str = f"{labels[0]} and {labels[1]}"
        else:
            fields_str = f"{', '.join(labels[:-1])}, and {labels[-1]}"

        return (
            f"Please provide your {fields_str} so we can create and process "
            "your support ticket."
        )
