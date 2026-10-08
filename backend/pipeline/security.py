import os
import re
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from .config import PipelineConfig

logger = logging.getLogger("pipeline.security")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Security] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class SecurityException(Exception):
    """Base security exception."""
    pass


class UnauthorizedAccessError(SecurityException):
    """Raised when a role attempts an action outside its granted permissions."""
    pass


class UnsafeFileError(SecurityException):
    """Raised when an unsafe or malformed file is rejected by the safety validator."""
    pass


# ------------------------------------------------------------------
# Role-Based Access Control (RBAC)
# ------------------------------------------------------------------
class AccessController:
    """
    Task 1: Phase 5 Role-Based Access Control (RBAC).
    Enforces deny-by-default permission checks across viewer, operator, and admin roles.
    """
    ROLE_VIEWER = "viewer"
    ROLE_OPERATOR = "operator"
    ROLE_ADMIN = "admin"

    PERMISSIONS: Dict[str, set] = {
        ROLE_VIEWER: {
            "view_status",
            "view_metadata",
            "view_evaluation",
        },
        ROLE_OPERATOR: {
            "view_status",
            "view_metadata",
            "view_evaluation",
            "run_pipeline",
            "create_version",
            "run_evaluation",
            "trigger_schedule",
        },
        ROLE_ADMIN: {
            "view_status",
            "view_metadata",
            "view_evaluation",
            "run_pipeline",
            "create_version",
            "run_evaluation",
            "trigger_schedule",
            "activate_version",
            "rollback_version",
            "modify_security_config",
        },
    }

    def __init__(self, security_logger: Optional["SecurityEventLogger"] = None):
        self.security_logger = security_logger

    def check_permission(self, role: Optional[str], action: str) -> bool:
        """
        Evaluates whether a role is authorized for an action using deny-by-default logic.
        Returns True if granted, False if denied.
        """
        if not role:
            return False
        role_lower = str(role).lower().strip()
        allowed_actions = self.PERMISSIONS.get(role_lower, set())
        return action in allowed_actions

    def authorize(
        self,
        role: Optional[str],
        action: str,
        resource: Optional[str] = None
    ) -> bool:
        """
        Enforces authorization. Raises UnauthorizedAccessError if denied.
        Logs security event on denial.
        """
        if self.check_permission(role, action):
            return True

        reason = f"Role '{role}' is not authorized to perform action '{action}' on resource '{resource or '*'}'."
        logger.warning(f"ACCESS DENIED: {reason}")

        if self.security_logger:
            self.security_logger.log_event(
                event_type="UNAUTHORIZED_ACCESS",
                severity="WARNING",
                action=action,
                result="DENIED",
                actor_role=role,
                resource=resource,
                details={"reason": reason}
            )

        raise UnauthorizedAccessError(reason)


# ------------------------------------------------------------------
# PII & Sensitive Data Masker
# ------------------------------------------------------------------
class PIIMasker:
    """
    Task 1: Phase 5 PII & Sensitive Data Protection Masker.
    Masks emails, phone numbers, payment card numbers, sensitive IDs, IP addresses,
    and API keys/secrets in logs, audit records, and reports without damaging the original documents.
    """
    # Regex patterns — ORDER MATTERS: API keys/secrets must be matched before
    # generic phone patterns to prevent digit-tail overlap.
    PATTERNS: List[Tuple[str, re.Pattern, str]] = [
        # Secrets & API Keys (must come FIRST to prevent phone regex stealing digits)
        ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z\-_]{20,50}\b"), "[SECRET_REDACTED]"),
        # sk- tokens: match sk- followed by any mix of alphanumeric, hyphens, underscores (10+ chars)
        ("SK_TOKEN", re.compile(r"\bsk-[A-Za-z0-9\-_]{10,}\b"), "[API_KEY_REDACTED]"),
        ("BEARER_TOKEN", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-\._~\+\/]+=*"), "Bearer [SECRET_REDACTED]"),
        ("AUTH_ASSIGNMENT", re.compile(r"(?i)\b(password|api_key|token|secret)\s*[:=]\s*['\"]?([^'\"\s,]+)['\"]?"), r"\1=[SECRET_REDACTED]"),

        # Payment & Card Numbers (16 digits in groups of 4 or contiguous)
        ("PAYMENT_CARD", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[PAYMENT_REDACTED]"),

        # Emails
        ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL_REDACTED]"),

        # Aadhaar (12 digits formatted: 1234 5678 9012)
        ("AADHAAR", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"), "[IDENTIFIER_REDACTED]"),
        # US SSN (3-2-4 digits)
        ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[IDENTIFIER_REDACTED]"),

        # Phone numbers (10 digits starting with 6-9, or formatted with +country/dashes/parens)
        # These come AFTER secrets so already-redacted tokens are not re-matched
        ("PHONE_INTL", re.compile(r"(?:\+?\d{1,3}[-.\\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE_REDACTED]"),
        ("PHONE_STANDALONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE_REDACTED]"),

        # IPv4 Addresses (strictly 4 numbers 0-255)
        ("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"), "[IP_REDACTED]"),
    ]

    @classmethod
    def mask_text(cls, text: str) -> str:
        """Sanitizes text by replacing sensitive PII and secrets with redaction labels.

        Patterns are applied sequentially in declaration order. API key / secret
        patterns appear BEFORE generic phone / digit patterns so that a token like
        'sk-live-99887766554433221100' is fully redacted as [API_KEY_REDACTED]
        and not partially consumed by the phone-number regex.
        """
        if not text or not isinstance(text, str):
            return text

        masked = text
        for _, pattern, replacement in cls.PATTERNS:
            masked = pattern.sub(replacement, masked)
        return masked

    @classmethod
    def mask_dict(cls, data: Any) -> Any:
        """Recursively sanitizes dictionaries, lists, and strings."""
        if isinstance(data, dict):
            return {k: cls.mask_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.mask_dict(item) for item in data]
        elif isinstance(data, str):
            return cls.mask_text(data)
        return data


# ------------------------------------------------------------------
# Prompt Injection Detector
# ------------------------------------------------------------------
class PromptInjectionDetector:
    """
    Task 1: Phase 5 Document Prompt-Injection Detector.
    Identifies malicious instructions embedded inside knowledge-base documents
    so they are treated purely as untrusted data and never as execution instructions.
    """
    INJECTION_RULES = [
        (
            "INSTRUCTION_OVERRIDE",
            r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions\b",
            "high"
        ),
        (
            "RULE_DISREGARD",
            r"(?i)\bdisregard\s+(?:all\s+)?(?:rules|instructions|safety rules|guidelines)\b",
            "high"
        ),
        (
            "SYSTEM_PROMPT_LEAK",
            r"(?i)\b(?:reveal|show|display|print)\s+(?:the\s+)?(?:system\s+prompt|developer\s+instructions|hidden\s+instructions)\b",
            "high"
        ),
        (
            "SYSTEM_OVERRIDE",
            r"(?i)\b(?:system\s+override|follow\s+these\s+instructions\s+instead)\b",
            "high"
        ),
        (
            "DEVELOPER_MODE",
            r"(?i)\b(?:you\s+are\s+now\s+in\s+developer\s+mode|act\s+as\s+(?:an?\s+)?unrestricted)\b",
            "high"
        ),
        (
            "DATA_EXFILTRATION",
            r"(?i)\b(?:exfiltrate|leak|steal)\s+(?:confidential|private|secret)\b",
            "high"
        ),
        (
            "SUSPICIOUS_DIRECTIVE",
            r"(?i)\b(?:developer\s+message|admin\s+override|jailbreak)\b",
            "medium"
        )
    ]

    @classmethod
    def scan(cls, text: str) -> Dict[str, Any]:
        """
        Scans text for prompt-injection signatures.
        Ensures normal text with common words ('ignore warnings', 'system requirements')
        is not falsely flagged.
        """
        if not text or not isinstance(text, str):
            return {
                "detected": False,
                "risk_level": "none",
                "matched_patterns": [],
                "categories": [],
                "action": "allow"
            }

        matched_patterns = []
        categories = []
        highest_risk = "none"

        for category, pattern, risk in cls.INJECTION_RULES:
            match = re.search(pattern, text)
            if match:
                matched_patterns.append(match.group(0))
                categories.append(category)
                if risk == "high":
                    highest_risk = "high"
                elif risk == "medium" and highest_risk != "high":
                    highest_risk = "medium"

        detected = len(matched_patterns) > 0
        return {
            "detected": detected,
            "risk_level": highest_risk,
            "matched_patterns": matched_patterns,
            "categories": categories,
            "action": "flag_untrusted" if detected else "allow"
        }


# ------------------------------------------------------------------
# File Safety Validator
# ------------------------------------------------------------------
class FileSafetyValidator:
    """
    Task 1: Phase 5 Application-Level File Safety Validator.
    Validates file sizes, allowed extensions, non-empty content, and binary integrity.
    """
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def validate_file(self, filepath: str) -> Tuple[bool, str]:
        """
        Validates file safety:
        - Must exist
        - Must have an allowed extension
        - Must not exceed max_document_size_mb
        - Must not be 0-byte empty file
        - Must have valid PDF header if .pdf
        """
        p = Path(filepath)
        if not p.exists():
            return False, f"File does not exist: {p.name}"

        # 1. Extension check
        ext = p.suffix.lower()
        if ext not in self.config.allowed_document_extensions:
            return False, f"Prohibited file extension '{ext}'. Allowed extensions: {self.config.allowed_document_extensions}"

        # 2. File size check
        size_bytes = p.stat().st_size
        if size_bytes == 0:
            return False, f"File '{p.name}' is empty (0 bytes)."

        max_bytes = self.config.max_document_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            return False, f"File '{p.name}' exceeds maximum allowed size of {self.config.max_document_size_mb}MB ({size_bytes / (1024*1024):.2f}MB)."

        # 3. PDF Header & integrity check
        if ext == ".pdf":
            try:
                with open(p, "rb") as f:
                    header = f.read(10)
                    if b"%PDF-" not in header:
                        return False, f"File '{p.name}' lacks a valid PDF magic header (%PDF-)."
            except Exception as e:
                return False, f"Error reading file '{p.name}': {e}"

        return True, "File safety validation passed."


# ------------------------------------------------------------------
# Security Event Logger
# ------------------------------------------------------------------
class SecurityEventLogger:
    """
    Task 1: Phase 5 Security Event Logger.
    Logs structured, sanitized security events to an atomic persistent JSON file.
    Guarantees no raw PII or secrets are written to the security log.
    """
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file

    def _get_log_path(self) -> str:
        if self.log_file:
            return self.log_file
        config = PipelineConfig()
        return config.security_log_file

    def load_events(self) -> List[Dict[str, Any]]:
        """Loads existing security events from the JSON log file."""
        log_path = self._get_log_path()
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as e:
                logger.warning(f"Could not read security log ({e}). Starting fresh.")
        return []

    def log_event(
        self,
        event_type: str,
        severity: str,
        action: str,
        result: str,
        actor_role: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Creates, sanitizes, and persists a structured security event.
        All details are automatically passed through PIIMasker.
        """
        raw_details = details or {}
        sanitized_details = PIIMasker.mask_dict(raw_details)

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "severity": severity,
            "actor_role": actor_role or "anonymous",
            "action": action,
            "resource": resource or "knowledge_base",
            "result": result,
            "details": sanitized_details
        }

        # Atomically append event
        log_path = self._get_log_path()
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        events = self.load_events()
        events.append(event)

        tmp_path = f"{log_path}.tmp_{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)
        os.replace(tmp_path, log_path)

        logger.info(f"SECURITY EVENT [{event_type}] Severity={severity} Action={action} Result={result}")
        return event
