"""
Task 2 Phase 4 – Multimodal Security & Privacy Guard

Provides comprehensive security and privacy hardening:
- PII and sensitive data masking for logs, error messages, and monitoring.
- Payment information protection (masks raw card numbers, CVVs, PINs).
- Prompt injection detection for untrusted document evidence and customer messages.
- Secure file handling with strict path traversal prevention.
- Sanitized security audit logging via Task 1's SecurityEventLogger.
- Safe error handling without stack trace or server path leakage.
"""

import os
import re
import logging
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

from pipeline.security import PIIMasker, PromptInjectionDetector, SecurityEventLogger

logger = logging.getLogger("multimodal.security")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [MultimodalSecurity] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Extended sensitive patterns beyond Task 1 PIIMasker
_ADDITIONAL_SENSITIVE_PATTERNS = [
    # CVV / CVC (3-4 digits preceded by cvv/cvc)
    (re.compile(r"(?i)\b(?:cvv|cvc|cid)\s*[:=]\s*['\"]?\d{3,4}['\"]?"), "cvv=[CVV_REDACTED]"),
    # PIN codes
    (re.compile(r"(?i)\b(?:pin|secret\s*pin)\s*[:=]\s*['\"]?\d{4,6}['\"]?"), "pin=[PIN_REDACTED]"),
    # Passwords / credentials
    (re.compile(r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^'\"\s,]+)['\"]?"), r"password=[SECRET_REDACTED]"),
    # Bank account numbers
    (re.compile(r"(?i)\b(?:account\s*(?:number|no\.?|#)|bank\s*acc(?:ount)?)\s*[:=]\s*['\"]?(\d{8,18})['\"]?"), r"account=[ACCOUNT_REDACTED]"),
]

# Additional prompt injection rules specific to document-based injection attacks
_ADDITIONAL_INJECTION_PATTERNS = [
    ("API_KEY_LEAK", re.compile(r"(?i)\b(?:reveal|print|show|give|leak)\s+(?:your\s+)?(?:api\s*key|secret|password|credentials|token)\b"), "high"),
    ("TOOL_CALL_INJECTION", re.compile(r"(?i)\b(?:call\s+(?:this\s+)?tool|execute\s+(?:command|tool))\b"), "high"),
    ("DATA_EXFILTRATION_DIRECTIVE", re.compile(r"(?i)\b(?:send\s+(?:this\s+)?information\s+to|transfer\s+data\s+to)\b"), "high"),
    ("SECURITY_DISREGARD", re.compile(r"(?i)\b(?:disregard\s+(?:security\s+rules|all\s+rules))\b"), "high"),
]


class MultimodalSecurityGuard:
    """
    Centralized Security & Privacy controller for multimodal image and PDF processing.
    Ensures all customer and document data is handled safely without leaking PII,
    payment secrets, filesystem paths, or allowing prompt injection execution.
    """

    # ─── PII & Sensitive Data Masking ──────────────────────────────────────────

    @classmethod
    def mask_sensitive_text(cls, text: Optional[str]) -> Optional[str]:
        """
        Masks PII, secrets, API keys, payment cards, CVVs, PINs, and bank accounts.
        Never modifies original document or customer inputs; used for logging and error reporting.
        """
        if not text or not isinstance(text, str):
            return text

        masked = text
        # 1. Apply additional payment and credential patterns
        for pattern, replacement in _ADDITIONAL_SENSITIVE_PATTERNS:
            masked = pattern.sub(replacement, masked)

        # 2. Apply Task 1 PIIMasker (emails, phones, credit cards, SSN, Aadhaar, API keys, tokens, IP)
        masked = PIIMasker.mask_text(masked)
        return masked

    @classmethod
    def mask_sensitive_dict(cls, data: Any) -> Any:
        """Recursively sanitizes nested dictionaries, lists, and primitives."""
        if isinstance(data, dict):
            return {k: cls.mask_sensitive_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.mask_sensitive_dict(item) for item in data]
        elif isinstance(data, str):
            return cls.mask_sensitive_text(data)
        return data

    # ─── Secure File Handling & Path Traversal Prevention ──────────────────────

    @classmethod
    def validate_filename(cls, filename: Optional[str]) -> Tuple[bool, str]:
        """
        Validates that a filename does not contain path traversal patterns,
        null bytes, or attempt directory escaping.
        """
        if not filename:
            return True, ""

        name = str(filename)

        # Check for null byte injection
        if "\x00" in name:
            return False, "Null byte injection detected in filename"

        # Check for directory traversal sequences
        if ".." in name or "/" in name or "\\" in name:
            return False, "Path traversal sequence detected in filename"

        # Check for absolute path prefixes (Windows drive letters or Unix root / home)
        if re.match(r'^[a-zA-Z]:', name) or name.startswith("~"):
            return False, "Absolute path prefix detected in filename"

        return True, ""

    @classmethod
    def ensure_safe_upload_path(cls, file_path: str, upload_dir: str) -> bool:
        """
        Verifies that the target file path resides strictly inside the configured upload directory.
        """
        try:
            resolved_file = os.path.abspath(file_path)
            resolved_dir = os.path.abspath(upload_dir)
            return os.path.commonpath([resolved_file, resolved_dir]) == resolved_dir
        except Exception:
            return False

    # ─── Prompt Injection Scanning ─────────────────────────────────────────────

    @classmethod
    def scan_prompt_injection(
        cls,
        text: Optional[str],
        source: str = "document",
        file_id: Optional[str] = None,
        security_logger: Optional[SecurityEventLogger] = None,
    ) -> Dict[str, Any]:
        """
        Scans extracted text or customer messages for prompt-injection attacks.
        Treats matching content purely as untrusted data and logs a sanitized security event.
        Never executes or follows suspicious instructions.
        """
        if not text or not isinstance(text, str):
            return {
                "detected": False,
                "risk_level": "none",
                "categories": [],
                "action": "allow",
            }

        # 1. Base scan via Task 1 PromptInjectionDetector
        base_result = PromptInjectionDetector.scan(text)
        categories = list(base_result.get("categories", []))
        risk_level = base_result.get("risk_level", "none")

        # 2. Additional multimodal-specific patterns
        for cat, pattern, risk in _ADDITIONAL_INJECTION_PATTERNS:
            if pattern.search(text):
                if cat not in categories:
                    categories.append(cat)
                if risk == "high":
                    risk_level = "high"
                elif risk == "medium" and risk_level != "high":
                    risk_level = "medium"

        detected = len(categories) > 0

        if detected and security_logger:
            # Record sanitized security event (no raw text stored!)
            try:
                security_logger.log_event(
                    event_type="MULTIMODAL_PROMPT_INJECTION_DETECTED",
                    severity="HIGH",
                    action="content_scan",
                    result="FLAGGED",
                    resource=file_id or "customer_input",
                    details={
                        "source": source,
                        "risk_level": risk_level,
                        "matched_categories": categories,
                        "sanitized_note": "Suspicious instruction patterns detected and flagged as untrusted data.",
                    }
                )
            except Exception as e:
                logger.warning(f"Could not record prompt injection event: {e}")

        return {
            "detected": detected,
            "risk_level": risk_level,
            "categories": categories,
            "action": "flag_untrusted" if detected else "allow",
        }

    # ─── Safe Error Sanitization ───────────────────────────────────────────────

    @classmethod
    def sanitize_error_message(cls, err: Any) -> str:
        """
        Converts any internal error to a clean, safe user message.
        Removes stack traces, Windows/Unix filesystem paths, and secrets.
        """
        raw_msg = str(err)

        # Mask PII / secrets
        clean_msg = cls.mask_sensitive_text(raw_msg)

        # Strip Windows paths (e.g. C:\Users\... or \temp_uploads\...)
        clean_msg = re.sub(r'[A-Za-z]:\\[^\s:"\']+', '[INTERNAL_PATH]', clean_msg)
        clean_msg = re.sub(r'/[^\s:"\']+(?:temp_uploads|backend|pipeline|faiss_index)[^\s:"\']*', '[INTERNAL_PATH]', clean_msg)
        clean_msg = re.sub(r'\\temp_uploads\\[^\s:"\']*', '[INTERNAL_PATH]', clean_msg)

        # If error message looks like a raw python exception trace, collapse to safe summary
        if "Traceback" in clean_msg or "File \"" in clean_msg:
            return "Unable to process the uploaded file due to an internal error."

        return clean_msg
