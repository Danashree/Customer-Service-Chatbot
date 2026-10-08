"""
Task 3 Phase 1: Conversation Information Extractor.
Extracts structured ticket fields from unresolved customer conversations.
Uses deterministic regex extraction combined with prompt-grounded LLM extraction (when available),
strictly adhering to the NO-HALLUCINATION rule: never guesses or invents missing fields.
"""

import re
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("tickets.extractor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [TicketExtractor] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Regex patterns for deterministic extraction
_EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
_PHONE_PATTERN = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{10}\b')
_ORDER_ID_PATTERNS = [
    re.compile(r'\b(?:Order\s*(?:ID|Number|#|No\.?)?|Reference\s*(?:ID|Number|#|No\.?)?|Ref\s*#?)\s*(?:is|=|:|-|\b)\s*([A-Z0-9]+(?:-[A-Z0-9]+)+|[A-Z0-9\-]{3,30})\b', re.IGNORECASE),
    re.compile(r'\b(ORD(?:-[A-Z0-9]+)+)\b', re.IGNORECASE),
    re.compile(r'\b(ORD[-#]?[A-Z0-9]{3,20})\b', re.IGNORECASE),
    re.compile(r'\b([A-Z]{2,4}\d{4,12})\b'),  # e.g., ORD12345, REF99881
]

_COURSE_PATTERNS = [
    re.compile(r'\b(?:for\s+the|enrolled\s+in|purchased\s+the|bought\s+the|taking\s+the|access\s+the|course:?)\s+([A-Z][A-Za-z0-9\s\+\#\.\-]{2,40}\s+course)\b', re.IGNORECASE),
    re.compile(r'\b([A-Z][A-Za-z0-9\s\+\#\.\-]{2,30}\s+(?:bootcamp|program|masterclass|specialization|training))\b', re.IGNORECASE),
    re.compile(r'\b(Python\s+for\s+Data\s+Science|Power\s*BI|Full\s*Stack\s*Web\s*Development|Data\s*Analytics|Machine\s*Learning|SQL\s+Mastery|Excel\s+Advanced|MySQL\s+Bootcamp)\b', re.IGNORECASE),
]

_EVIDENCE_PATTERNS = [
    re.compile(r'\b((?:attached|attaching|uploaded|sent|provided)\s+(?:the\s+)?(?:payment\s+screenshot|screenshot|receipt|invoice|slip|proof|document|pdf|image))\b', re.IGNORECASE),
    re.compile(r'\b(payment\s+screenshot|payment\s+receipt|transaction\s+screenshot|bank\s+slip|screenshot\s+attached|receipt\s+attached)\b', re.IGNORECASE),
]

_CUSTOMER_PATTERNS = [
    re.compile(r'\b(?:My\s+name\s+is|I\s+am|This\s+is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'),
    re.compile(r'\bCustomer\s*(?:name)?\s*[:=]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'),
]


class ConversationExtractor:
    """
    Extracts structured ticket fields from unresolved customer conversations.
    Enforces no hallucination: returns None for any field not explicitly provided.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def extract(self, conversation_text: str) -> Dict[str, Optional[str]]:
        """
        Extract structured ticket data.
        Returns a dictionary with keys:
            customer_name, contact_email, contact_phone, course_name,
            order_id, issue, evidence
        """
        if not conversation_text or not conversation_text.strip():
            return {
                "customer_name": None,
                "contact_email": None,
                "contact_phone": None,
                "course_name": None,
                "order_id": None,
                "issue": None,
                "evidence": None,
            }

        text = conversation_text.strip()

        # 1. Deterministic Rule-Based Extraction
        extracted_email = self._extract_email(text)
        extracted_phone = self._extract_phone(text)
        extracted_order_id = self._extract_order_id(text)
        extracted_course = self._extract_course(text)
        extracted_evidence = self._extract_evidence(text)
        extracted_name = self._extract_customer_name(text)
        extracted_issue = self._extract_issue_rule_based(text)

        # 2. LLM Enhancement (for subtle issues/course names if LLM is enabled and configured)
        if self.use_llm and (not extracted_issue or not extracted_course or not extracted_name):
            llm_result = self._extract_with_llm(text)
            if llm_result:
                if not extracted_name and llm_result.get("customer_name"):
                    extracted_name = llm_result["customer_name"]
                if not extracted_email and llm_result.get("contact_email"):
                    extracted_email = llm_result["contact_email"]
                if not extracted_phone and llm_result.get("contact_phone"):
                    extracted_phone = llm_result["contact_phone"]
                if not extracted_course and llm_result.get("course_name"):
                    extracted_course = llm_result["course_name"]
                if not extracted_order_id and llm_result.get("order_id"):
                    extracted_order_id = llm_result["order_id"]
                if not extracted_issue and llm_result.get("issue"):
                    extracted_issue = llm_result["issue"]
                if not extracted_evidence and llm_result.get("evidence"):
                    extracted_evidence = llm_result["evidence"]

        return {
            "customer_name": extracted_name,
            "contact_email": extracted_email,
            "contact_phone": extracted_phone,
            "course_name": extracted_course,
            "order_id": extracted_order_id,
            "issue": extracted_issue,
            "evidence": extracted_evidence,
        }

    def _extract_email(self, text: str) -> Optional[str]:
        m = _EMAIL_PATTERN.search(text)
        return m.group(0).strip() if m else None

    def _extract_phone(self, text: str) -> Optional[str]:
        m = _PHONE_PATTERN.search(text)
        if m:
            val = m.group(0).strip()
            # Verify digits count between 10 and 15
            digits = re.sub(r'\D', '', val)
            if 10 <= len(digits) <= 15:
                return val
        return None

    def _extract_order_id(self, text: str) -> Optional[str]:
        for pat in _ORDER_ID_PATTERNS:
            m = pat.search(text)
            if m:
                val = m.group(1).strip()
                # Ensure it's not a common stop word
                if len(val) >= 3 and not val.lower() in {"the", "and", "not", "yes", "none"}:
                    return val
        return None

    def _extract_course(self, text: str) -> Optional[str]:
        for pat in _COURSE_PATTERNS:
            m = pat.search(text)
            if m:
                val = m.group(1).strip()
                # Clean trailing "course" if redundant
                val_clean = re.sub(r'\s+course$', '', val, flags=re.IGNORECASE).strip()
                return val_clean if val_clean else val
        return None

    def _extract_evidence(self, text: str) -> Optional[str]:
        for pat in _EVIDENCE_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(0).strip().lower()
        return None

    def _extract_customer_name(self, text: str) -> Optional[str]:
        for pat in _CUSTOMER_PATTERNS:
            m = pat.search(text)
            if m:
                name = m.group(1).strip()
                if name.lower() not in {"student", "learner", "customer", "user", "admin"}:
                    return name
        return None

    def _extract_issue_rule_based(self, text: str) -> Optional[str]:
        # Identify common e-learning customer service issues
        issue_keywords = [
            (r'\b(?:cannot|can\'t|unable to|not able to|not received|never got)\s+(?:access|open|login|log in|view|enroll)\b', "course access not received"),
            (r'\b(?:refund|money back|reimbursement)\b', "refund request"),
            (r'\b(?:charged twice|double payment|payment deducted|payment failed)\b', "payment issue"),
            (r'\b(?:certificate|completion certificate|certificate not received)\b', "certificate issue"),
            (r'\b(?:video|player|audio|lecture|playback)\s+(?:not working|error|black screen)\b', "video playback issue"),
            (r'\b(?:doubt|mentor|instructor|discord)\s+(?:support|help|channel)\b', "mentorship/doubt support issue"),
        ]
        text_lower = text.lower()
        for pat, desc in issue_keywords:
            if re.search(pat, text_lower):
                return desc

        # If conversation mentions an issue in natural words
        m = re.search(r'\b(?:issue|problem)\s*(?:is|=|:|-)?\s*([^.,\n]+)', text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if len(val) >= 4:
                return val

        # Fallback to sentence expressing trouble if text is brief
        sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip()]
        for s in sentences:
            if any(w in s.lower() for w in ["cannot", "can't", "failed", "error", "unable", "not received", "didn't get"]):
                return s

        return None

    def _extract_with_llm(self, text: str) -> Optional[Dict[str, Optional[str]]]:
        """Optionally uses Gemini LLM from langchain_helper with strict JSON output."""
        try:
            from langchain_helper import llm
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = PromptTemplate(
                template="""You are a strict data extraction assistant for an online-course support ticket system.
Extract the following information from the customer message.
CRITICAL RULE: DO NOT INVENT, ASSUME, OR HALLUCINATE ANY VALUES.
If any field is not explicitly stated in the text, set its value to null.

Text:
"{conversation}"

Return ONLY a JSON object with these exact keys:
{{
  "customer_name": string or null,
  "contact_email": string or null,
  "contact_phone": string or null,
  "course_name": string or null,
  "order_id": string or null,
  "issue": string or null,
  "evidence": string or null
}}
JSON:""",
                input_variables=["conversation"]
            )
            chain = prompt | llm | StrOutputParser()
            raw_response = chain.invoke({"conversation": text})
            # Clean markdown formatting if present
            cleaned = re.sub(r'^```(?:json)?\s*', '', raw_response.strip())
            cleaned = re.sub(r'\s*```$', '', cleaned.strip())
            return json.loads(cleaned)
        except Exception as e:
            logger.debug(f"LLM extraction skipped or failed: {e}")
            return None
