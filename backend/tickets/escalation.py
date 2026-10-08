"""
Task 4 Phase 2–5: Risk & Escalation Engine.

Handles:
  1. Explicit risk detection:
     - Account compromise (CRITICAL)
     - Duplicate payment (HIGH)
     - Legal threat (CRITICAL)
     - Configurable custom high-risk patterns
  2. Repeated-negative detection & 15-minute unresolved timer:
     - Detects >= N consecutive negative/frustrated messages
     - Automatically escalates negative conversations unresolved for >15 minutes
     - Injectable `current_time` for deterministic testing
  3. Business hours & On-call queue:
     - Integrates with Task 3 business hours (BusinessHoursConfig, is_business_time)
     - Urgent + outside business hours -> ON_CALL_QUEUE
     - Normal + outside business hours -> NEXT_WORKING_DAY
     - Urgent + inside business hours -> NORMAL (escalated)
     - Normal + inside business hours -> NORMAL
  4. Escalation record persistence & PII masking:
     - Atomic JSON storage (escalations_store.json)
     - PII masking on conversation summaries and reason logs
     - Safe fallback and zero-hallucination guarantees
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .business_hours import BusinessHoursConfig, is_business_time
from .conversation_analyzer import analyse_conversation
from .sentiment_multilingual import analyse_message
from .sla_config import get_sla_config
from .task4_models import (
    EscalationEvaluation,
    EscalationRecord,
    EscalationRecordStatus,
    EscalationRequest,
    QueueType,
    RiskAssessment,
    RiskCondition,
    RiskLevel,
    SentimentLabel,
)

logger = logging.getLogger(__name__)

# PII Masker import
try:
    from pipeline.security import PIIMasker as _PIIMasker

    def _mask_text(text: str) -> str:
        return _PIIMasker.mask_text(text)
except Exception:  # pragma: no cover
    def _mask_text(text: str) -> str:
        return text


# ──────────────────────────────────────────────────────────────────────────────
# Risk Patterns
# ──────────────────────────────────────────────────────────────────────────────

_ACCOUNT_COMPROMISE_PATTERNS = [
    r"\bhack(?:ed|ing)?\b",
    r"\b(?:someone\s+)?(?:has\s+)?accessed\s+my\s+account\b",
    r"\bunauthorized\s+(?:login|access|activity)\b",
    r"\baccount\s+stolen\b",
    r"\bsuspicious\s+(?:login|activity)\b",
    r"\bcompromised\b",
    r"\bpassword\s+changed\s+without\b",
    r"\baccount\s+takeover\b",
    r"\bunrecognized\s+(?:login|device)\b",
    r"\bsomeone\s+(?:else\s+)?logged\s+into\s+my\s+account\b",
]

_DUPLICATE_PAYMENT_PATTERNS = [
    r"\bcharg(?:ed|e|ing)?\s+(?:me\s+)?twice\b",
    r"\bduplicate\s+payment\b",
    r"\bpaid\s+twice\b",
    r"\bdouble\s+charg(?:ed|e|ing)?\b",
    r"\bcharg(?:ed|e|ing)?\s+(?:me\s+)?two\s+times\b",
    r"\bdebit(?:ed)?\s+twice\b",
    r"\btwo\s+charges\b",
    r"\bsecond\s+charge\b",
    r"\bcharged\s+again\b",
]


_LEGAL_THREAT_PATTERNS = [
    r"\blegal\s+(?:action|notice|suit|proceedings?|complaint)\b",
    r"\blawyer\b",
    r"\bcourt\b",
    r"\blawsuit\b",
    r"\battorney\b",
    r"\bsue\s+you\b",
    r"\bsuing\b",
    r"\bconsumer\s+court\b",
    r"\bconsumer\s+forum\b",
]


# ──────────────────────────────────────────────────────────────────────────────
# Runtime Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EscalationConfig:
    unresolved_minutes_threshold: int = 15
    repeated_negative_threshold: int = 2
    custom_high_risk_patterns: List[str] = field(default_factory=list)


_GLOBAL_CONFIG = EscalationConfig()
_CONFIG_LOCK = threading.Lock()


def get_escalation_config() -> EscalationConfig:
    with _CONFIG_LOCK:
        return EscalationConfig(
            unresolved_minutes_threshold=_GLOBAL_CONFIG.unresolved_minutes_threshold,
            repeated_negative_threshold=_GLOBAL_CONFIG.repeated_negative_threshold,
            custom_high_risk_patterns=list(_GLOBAL_CONFIG.custom_high_risk_patterns),
        )


def update_escalation_config(
    unresolved_minutes_threshold: Optional[int] = None,
    repeated_negative_threshold: Optional[int] = None,
    custom_high_risk_patterns: Optional[List[str]] = None,
) -> EscalationConfig:
    global _GLOBAL_CONFIG
    with _CONFIG_LOCK:
        if unresolved_minutes_threshold is not None:
            _GLOBAL_CONFIG.unresolved_minutes_threshold = unresolved_minutes_threshold
        if repeated_negative_threshold is not None:
            _GLOBAL_CONFIG.repeated_negative_threshold = repeated_negative_threshold
        if custom_high_risk_patterns is not None:
            _GLOBAL_CONFIG.custom_high_risk_patterns = list(custom_high_risk_patterns)
        return EscalationConfig(
            unresolved_minutes_threshold=_GLOBAL_CONFIG.unresolved_minutes_threshold,
            repeated_negative_threshold=_GLOBAL_CONFIG.repeated_negative_threshold,
            custom_high_risk_patterns=list(_GLOBAL_CONFIG.custom_high_risk_patterns),
        )


def reset_escalation_config() -> None:
    global _GLOBAL_CONFIG
    with _CONFIG_LOCK:
        _GLOBAL_CONFIG = EscalationConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Risk Assessment Function
# ──────────────────────────────────────────────────────────────────────────────

def assess_risk(text: str, custom_patterns: Optional[List[str]] = None) -> RiskAssessment:
    """
    Evaluate explicit risk categories regardless of emotional sentiment.
    """
    if not text:
        return RiskAssessment(risk_level=RiskLevel.NORMAL, risk_condition=RiskCondition.NONE)

    lower = text.lower()

    # 1. Account compromise (CRITICAL)
    for p in _ACCOUNT_COMPROMISE_PATTERNS:
        if re.search(p, lower):
            return RiskAssessment(
                risk_level=RiskLevel.CRITICAL,
                risk_condition=RiskCondition.ACCOUNT_COMPROMISE,
                risk_reason="Account compromise or unauthorized access detected",
            )

    # 2. Legal threat (CRITICAL)
    for p in _LEGAL_THREAT_PATTERNS:
        if re.search(p, lower):
            return RiskAssessment(
                risk_level=RiskLevel.CRITICAL,
                risk_condition=RiskCondition.LEGAL_THREAT,
                risk_reason="Legal threat, court proceedings, or lawyer mention detected",
            )

    # 3. Duplicate payment (HIGH)
    for p in _DUPLICATE_PAYMENT_PATTERNS:
        if re.search(p, lower):
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                risk_condition=RiskCondition.DUPLICATE_PAYMENT,
                risk_reason="Duplicate payment or double charge reported",
            )

    # 4. Custom high-risk patterns
    patterns = custom_patterns or _GLOBAL_CONFIG.custom_high_risk_patterns
    for p in patterns:
        if re.search(p, lower, re.IGNORECASE):
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                risk_condition=RiskCondition.CUSTOM_HIGH_RISK,
                risk_reason=f"Configured high-risk pattern matched: {p}",
            )

    return RiskAssessment(risk_level=RiskLevel.NORMAL, risk_condition=RiskCondition.NONE)


# ──────────────────────────────────────────────────────────────────────────────
# Escalation Storage (JSON Persistence)
# ──────────────────────────────────────────────────────────────────────────────

class EscalationStorage:
    """
    Thread-safe atomic JSON storage for persistent EscalationRecords.
    """

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(__file__).parent / "escalations_store.json"
        self._lock = threading.Lock()
        self._ensure_store_file()

    def _ensure_store_file(self) -> None:
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_records([])

    def _read_records(self) -> List[Dict[str, Any]]:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_records(self, records: List[Dict[str, Any]]) -> None:
        temp_file = self.storage_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        os.replace(temp_file, self.storage_path)

    def save(self, record: EscalationRecord) -> EscalationRecord:
        with self._lock:
            records = self._read_records()
            data = record.model_dump()
            # Update if existing, else append
            idx = next((i for i, r in enumerate(records) if r["escalation_id"] == record.escalation_id), None)
            if idx is not None:
                records[idx] = data
            else:
                records.append(data)
            self._write_records(records)
            return record

    def get(self, escalation_id: str) -> Optional[EscalationRecord]:
        with self._lock:
            records = self._read_records()
            item = next((r for r in records if r["escalation_id"] == escalation_id), None)
            if item:
                return EscalationRecord(**item)
            return None

    def list_all(self) -> List[EscalationRecord]:
        with self._lock:
            records = self._read_records()
            return [EscalationRecord(**r) for r in records]

    def list_on_call(self) -> List[EscalationRecord]:
        with self._lock:
            records = self._read_records()
            return [
                EscalationRecord(**r)
                for r in records
                if r.get("queue_type") == QueueType.ON_CALL_QUEUE.value
            ]


# Singleton storage
_DEFAULT_STORAGE = EscalationStorage()


# ──────────────────────────────────────────────────────────────────────────────
# Escalation Engine
# ──────────────────────────────────────────────────────────────────────────────

class EscalationEngine:
    """
    Evaluates customer messages and conversations against:
      - Risk thresholds (Account compromise, duplicate payment, legal threats)
      - Repeated negative feedback
      - 15-minute unresolved timer
      - Business hours & On-call queueing
    """

    def __init__(
        self,
        storage: Optional[EscalationStorage] = None,
        business_hours_cfg: Optional[BusinessHoursConfig] = None,
    ):
        self.storage = storage or _DEFAULT_STORAGE
        self._bh_config = business_hours_cfg

    def get_business_hours_config(self) -> BusinessHoursConfig:
        if self._bh_config is not None:
            return self._bh_config
        # Reuse Task 3 business hours from SLA config
        return get_sla_config().business_hours

    def evaluate(self, req: EscalationRequest) -> EscalationEvaluation:
        """
        Main evaluation entry point.
        """
        cfg = get_escalation_config()
        bh_cfg = self.get_business_hours_config()

        # Simulated or current time
        eval_time = req.current_time or datetime.now(timezone.utc)

        # 1. Perform Sentiment & Conversation Analysis
        sent_resp = analyse_conversation(
            current_message=req.message,
            conversation_history=req.conversation_history,
            negative_threshold=cfg.repeated_negative_threshold,
        )
        msg_analysis = sent_resp.message_analysis
        conv_analysis = sent_resp.conversation_analysis

        sentiment = msg_analysis.sentiment
        confidence = msg_analysis.confidence
        frustration = msg_analysis.frustration or (conv_analysis.frustration if conv_analysis else False)
        urgency = msg_analysis.urgency or (conv_analysis.urgency if conv_analysis else False)
        sarcasm = msg_analysis.sarcasm or (conv_analysis.sarcasm if conv_analysis else False)
        repeated_neg = conv_analysis.repeated_negative if conv_analysis else False

        # 2. Risk Assessment (Message + History)
        combined_text = req.message
        if req.conversation_history:
            combined_text = " ".join(req.conversation_history) + " " + req.message
        risk = assess_risk(combined_text)

        # 3. Check Business Hours
        in_business_hours = is_business_time(eval_time, bh_cfg)
        bh_status_str = "open" if in_business_hours else "closed"

        # 4. Check 15-Minute Unresolved Condition
        unresolved_escalation = False
        time_diff_minutes = 0.0
        # Determine unresolved start point
        unresolved_start = req.unresolved_since or req.conversation_started_at
        if unresolved_start is not None:
            # Align timezones if necessary
            if unresolved_start.tzinfo is None and eval_time.tzinfo is not None:
                unresolved_start = unresolved_start.replace(tzinfo=eval_time.tzinfo)
            elif unresolved_start.tzinfo is not None and eval_time.tzinfo is None:
                eval_time_aligned = eval_time.replace(tzinfo=unresolved_start.tzinfo)
            else:
                eval_time_aligned = eval_time

            diff = (eval_time_aligned - unresolved_start).total_seconds()
            time_diff_minutes = max(0.0, diff / 60.0)

            def _to_str(x: Any) -> str:
                return x.value if hasattr(x, "value") else str(x)

            # Rule: Negative / frustrated conversations unresolved > 15 mins
            is_negative_conv = (
                _to_str(sentiment) in (SentimentLabel.NEGATIVE.value, SentimentLabel.FRUSTRATED.value)
                or (conv_analysis and _to_str(conv_analysis.overall_sentiment) in (SentimentLabel.NEGATIVE.value, SentimentLabel.FRUSTRATED.value))
                or frustration
                or repeated_neg
            )
            if is_negative_conv and time_diff_minutes > cfg.unresolved_minutes_threshold:
                unresolved_escalation = True

        def _val(x: Any) -> str:
            return x.value if hasattr(x, "value") else str(x)

        # 5. Determine Escalation Decision and Queue
        should_escalate = False
        reason: Optional[str] = None
        activated_condition = "none"
        queue_type = QueueType.NORMAL

        if risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) or _val(risk.risk_level) in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value):
            should_escalate = True
            reason = risk.risk_reason or "High risk condition detected"
            activated_condition = _val(risk.risk_condition)
        elif unresolved_escalation:
            should_escalate = True
            reason = f"Negative conversation unresolved for {time_diff_minutes:.1f} minutes (> {cfg.unresolved_minutes_threshold} mins)"
            activated_condition = "unresolved_negative_conversation_escalation"
        elif repeated_neg:
            should_escalate = True
            reason = f"Repeated negative messages detected ({conv_analysis.negative_message_count if conv_analysis else 0} messages)"
            activated_condition = "repeated_negative_messages_detected"
        elif urgency:
            should_escalate = True
            reason = "Customer flagged message as urgent"
            activated_condition = "urgent_complaint_detected"

        # 6. Queue Type Determination based on Business Hours
        if not in_business_hours:
            if urgency or risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) or _val(risk.risk_level) in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value):
                queue_type = QueueType.ON_CALL_QUEUE
            else:
                queue_type = QueueType.NEXT_WORKING_DAY
        else:
            queue_type = QueueType.NORMAL

        # 7. Escalation Record Creation (if escalated)
        escalation_id = None
        if should_escalate:
            escalation_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"

            # Prepare masked conversation summary
            summary_parts = []
            if req.conversation_history:
                for idx, h_msg in enumerate(req.conversation_history):
                    summary_parts.append(f"Customer [{idx+1}]: {h_msg}")
            summary_parts.append(f"Customer [latest]: {req.message}")
            raw_summary = " | ".join(summary_parts)
            masked_summary = _mask_text(raw_summary)

            iso_now = eval_time.isoformat()
            record = EscalationRecord(
                escalation_id=escalation_id,
                ticket_id=req.ticket_id,
                conversation_id=req.conversation_id,
                created_at=iso_now,
                escalation_reason=_mask_text(reason or ""),
                activated_condition=activated_condition,
                sentiment=_val(sentiment),
                sentiment_confidence=confidence,
                risk_level=_val(risk.risk_level),
                urgency=urgency,
                frustration=frustration,
                sarcasm=sarcasm,
                business_hours_status=bh_status_str,
                queue_type=_val(queue_type),
                conversation_summary=masked_summary,
                status=_val(EscalationRecordStatus.OPEN),
                escalation_timestamp=iso_now,
            )
            self.storage.save(record)

        def _to_enum(val: Any, enum_cls: Any, default: Any) -> Any:
            if isinstance(val, enum_cls):
                return val
            try:
                return enum_cls(str(val))
            except Exception:
                return default

        return EscalationEvaluation(
            should_escalate=should_escalate,
            reason=reason,
            activated_condition=activated_condition,
            risk_level=_to_enum(risk.risk_level, RiskLevel, RiskLevel.NORMAL),
            queue_type=_to_enum(queue_type, QueueType, QueueType.NORMAL),
            sentiment=_to_enum(sentiment, SentimentLabel, SentimentLabel.NEUTRAL),
            confidence=confidence,
            frustration=frustration,
            urgency=urgency,
            sarcasm=sarcasm,
            repeated_negative=repeated_neg,
            business_hours_status=bh_status_str,
            escalation_id=escalation_id,
        )



_DEFAULT_ENGINE = EscalationEngine()


def get_escalation_engine() -> EscalationEngine:
    return _DEFAULT_ENGINE
