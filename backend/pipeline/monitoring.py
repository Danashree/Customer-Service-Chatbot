import os
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from .config import PipelineConfig
from .security import PIIMasker, SecurityEventLogger


def calculate_application_confidence(
    retrieval_score: Optional[float],
    grounding_score: Optional[float]
) -> Optional[float]:
    """
    Derives deterministic application-level confidence from available retrieval
    and grounding quality evidence (normalized to 0.0 - 1.0 range).
    Documented as application confidence rather than model probability.
    """
    if retrieval_score is None and grounding_score is None:
        return None
    r = retrieval_score if retrieval_score is not None else 0.0
    g = grounding_score if grounding_score is not None else 0.0
    if retrieval_score is not None and grounding_score is not None:
        conf = round(0.5 * r + 0.5 * g, 4)
    else:
        conf = round(r if retrieval_score is not None else g, 4)
    return min(1.0, max(0.0, conf))


logger = logging.getLogger("pipeline.monitoring")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Monitoring] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class PipelineExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    WAITING_FOR_MAINTENANCE = "waiting_for_maintenance_window"
    QUALITY_REJECTED = "quality_rejected"
    ROLLED_BACK = "rolled_back"
    NO_CHANGES = "no_changes"


@dataclass
class PipelineMetrics:
    """Structured metrics record for a single pipeline run."""
    run_id: str
    start_time: str
    end_time: str
    duration_ms: float
    status: str
    trigger_type: str = "scheduled"
    documents_discovered: int = 0
    new_documents: int = 0
    modified_documents: int = 0
    duplicate_documents: int = 0
    quarantined_documents: int = 0
    chunks_created: int = 0
    candidate_version: Optional[str] = None
    evaluation_status: Optional[str] = None
    retrieval_score: Optional[float] = None
    grounding_score: Optional[float] = None
    regression_amount: Optional[float] = None
    activation_status: Optional[str] = None
    rollback_status: Optional[str] = None
    retry_attempt: int = 0
    error_message: Optional[str] = None
    health_check_passed: Optional[bool] = None
    response_confidence: Optional[float] = None

    def __post_init__(self):
        # Derive response_confidence deterministically from retrieval & grounding if not explicitly provided
        if self.response_confidence is None:
            self.response_confidence = calculate_application_confidence(
                self.retrieval_score, self.grounding_score
            )
        elif self.response_confidence is not None:
            # Normalize and clamp to [0.0, 1.0]
            self.response_confidence = min(1.0, max(0.0, round(float(self.response_confidence), 4)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



class MetricsCollector:
    """
    Collects, aggregates, and persists pipeline performance and execution metrics.
    Enforces bounded history and ensures no raw PII or secrets are stored.
    """
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._start_perf_counter: Optional[float] = None
        self._start_iso: Optional[str] = None

    def start_run(self) -> None:
        """Starts monotonic timing for a pipeline execution."""
        self._start_perf_counter = time.perf_counter()
        self._start_iso = datetime.now(timezone.utc).isoformat()

    def end_run(self) -> float:
        """Stops timing and returns duration in milliseconds."""
        if self._start_perf_counter is None:
            return 0.0
        elapsed_sec = time.perf_counter() - self._start_perf_counter
        self._start_perf_counter = None
        return round(elapsed_sec * 1000.0, 2)

    def load_state(self) -> Dict[str, Any]:
        """Loads existing monitoring state JSON safely."""
        state_file = self.config.monitoring_state_file
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not read monitoring state ({e}). Starting fresh.")

        return {
            "metrics_history": [],
            "alerts": [],
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "retry_count": 0,
            "quality_rejections": 0,
            "rollback_count": 0,
            "total_escalations": 0,
            "escalations": [],
            "last_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "last_response_confidence": None,
            "confidence_history": [],
            "last_health_check": None
        }

    def save_state(self, state: Dict[str, Any]) -> None:
        """Atomically persists monitoring state JSON."""
        state_file = self.config.monitoring_state_file
        dir_name = os.path.dirname(state_file)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Enforce history bounds
        limit = self.config.metrics_history_limit
        if len(state.get("metrics_history", [])) > limit:
            state["metrics_history"] = state["metrics_history"][-limit:]
        if len(state.get("escalations", [])) > limit:
            state["escalations"] = state["escalations"][-limit:]
        if len(state.get("confidence_history", [])) > limit:
            state["confidence_history"] = state["confidence_history"][-limit:]

        # Mask PII and secrets before writing to disk
        sanitized = PIIMasker.mask_dict(state)

        tmp_file = f"{state_file}.tmp_{os.getpid()}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, indent=2)
        os.replace(tmp_file, state_file)

    def record_execution(self, metrics: PipelineMetrics) -> Dict[str, Any]:
        """Records a single pipeline execution, updates aggregates, and saves state."""
        state = self.load_state()

        # Update counters
        state["total_runs"] = state.get("total_runs", 0) + 1
        if metrics.status in (PipelineExecutionStatus.SUCCESS.value, "activated"):
            state["successful_runs"] = state.get("successful_runs", 0) + 1
        elif metrics.status == PipelineExecutionStatus.FAILED.value:
            state["failed_runs"] = state.get("failed_runs", 0) + 1
        elif metrics.status == PipelineExecutionStatus.QUALITY_REJECTED.value:
            state["quality_rejections"] = state.get("quality_rejections", 0) + 1
        elif metrics.status == PipelineExecutionStatus.ROLLED_BACK.value:
            state["rollback_count"] = state.get("rollback_count", 0) + 1

        if metrics.retry_attempt > 0:
            state["retry_count"] = state.get("retry_count", 0) + 1

        # Track confidence if present
        if metrics.response_confidence is not None:
            state["last_response_confidence"] = metrics.response_confidence

        # Append execution record
        history = state.setdefault("metrics_history", [])
        history.append(metrics.to_dict())

        # Update latency statistics
        durations = [m.get("duration_ms", 0.0) for m in history if m.get("duration_ms", 0.0) > 0]
        if durations:
            state["last_latency_ms"] = metrics.duration_ms
            state["avg_latency_ms"] = round(sum(durations) / len(durations), 2)
            state["min_latency_ms"] = round(min(durations), 2)
            state["max_latency_ms"] = round(max(durations), 2)


        # Check for alert conditions
        alerts = state.setdefault("alerts", [])
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. High Latency Alert
        if metrics.duration_ms > self.config.max_pipeline_latency_ms:
            alerts.append({
                "alert_type": "HIGH_LATENCY",
                "severity": "WARNING",
                "timestamp": now_iso,
                "message": f"Pipeline duration {metrics.duration_ms}ms exceeded limit of {self.config.max_pipeline_latency_ms}ms",
                "related_run": metrics.run_id
            })

        # 2. Repeated Failures Alert
        recent_statuses = [m.get("status") for m in history[-self.config.max_recent_failures:]]
        if len(recent_statuses) >= self.config.max_recent_failures and all(s == "failed" for s in recent_statuses):
            alerts.append({
                "alert_type": "REPEATED_FAILURES",
                "severity": "CRITICAL",
                "timestamp": now_iso,
                "message": f"Detected {self.config.max_recent_failures} consecutive pipeline failures",
                "related_run": metrics.run_id
            })

        # 3. Quality Regression Alert
        if metrics.regression_amount is not None and metrics.regression_amount > self.config.max_allowed_regression:
            alerts.append({
                "alert_type": "QUALITY_REGRESSION",
                "severity": "WARNING",
                "timestamp": now_iso,
                "message": f"Candidate quality regression {metrics.regression_amount:.4f} exceeded threshold {self.config.max_allowed_regression:.4f}",
                "related_version": metrics.candidate_version
            })

        # 4. Rollback Alert
        if metrics.rollback_status == "rolled_back":
            alerts.append({
                "alert_type": "AUTOMATIC_ROLLBACK",
                "severity": "CRITICAL",
                "timestamp": now_iso,
                "message": f"Automatic rollback triggered for candidate version {metrics.candidate_version}",
                "related_version": metrics.candidate_version
            })

        # Keep alerts bounded to last 50
        if len(alerts) > 50:
            state["alerts"] = alerts[-50:]

        self.save_state(state)
        return state

    def record_health_check(
        self,
        version: str,
        passed: bool,
        previous_version: Optional[str] = None,
        rollback_triggered: bool = False
    ) -> Dict[str, Any]:
        """Records the outcome of a post-activation health check and any resulting rollback."""
        state = self.load_state()
        now_iso = datetime.now(timezone.utc).isoformat()

        record = {
            "timestamp": now_iso,
            "version": version,
            "passed": passed,
            "previous_version": previous_version,
            "rollback_triggered": rollback_triggered
        }
        state["last_health_check"] = record

        if not passed:
            state.setdefault("alerts", []).append({
                "alert_type": "HEALTH_CHECK_FAILURE",
                "severity": "CRITICAL",
                "timestamp": now_iso,
                "message": f"Health check failed for activated version '{version}'",
                "related_version": version
            })
            if rollback_triggered:
                state["rollback_count"] = state.get("rollback_count", 0) + 1
                state.setdefault("alerts", []).append({
                    "alert_type": "AUTOMATIC_ROLLBACK",
                    "severity": "CRITICAL",
                    "timestamp": now_iso,
                    "message": f"Automatic rollback initiated from '{version}' to '{previous_version}'",
                    "related_version": version
                })

        self.save_state(state)
        return record

    def record_confidence(
        self,
        confidence: float,
        query: Optional[str] = None,
        source: str = "query",
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Records an explicit query/response-level confidence metric (normalized 0.0 - 1.0).
        Persisted in monitoring state.
        """
        normalized_conf = min(1.0, max(0.0, round(float(confidence), 4)))
        state = self.load_state()
        state["last_response_confidence"] = normalized_conf

        conf_history = state.setdefault("confidence_history", [])
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": normalized_conf,
            "source": source,
            "query": PIIMasker.mask_text(query) if query else None,
            "metadata": PIIMasker.mask_dict(metadata or {})
        }
        conf_history.append(entry)
        limit = self.config.metrics_history_limit
        if len(conf_history) > limit:
            state["confidence_history"] = conf_history[-limit:]

        self.save_state(state)
        return normalized_conf

    def record_escalation(
        self,
        reason: str,
        severity: str = "medium",
        source: str = "chatbot",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Records a customer escalation event with sanitized reason and metadata.
        Atomically updates total_escalations counter and persists escalation history.
        """
        state = self.load_state()
        state["total_escalations"] = state.get("total_escalations", 0) + 1

        sanitized_reason = PIIMasker.mask_text(str(reason)) if reason else "Unspecified escalation"
        sanitized_session = PIIMasker.mask_text(str(session_id)) if session_id else None
        sanitized_metadata = PIIMasker.mask_dict(metadata or {})

        escalation_event = {
            "escalation_id": str(uuid.uuid4())[:8],
            "escalation_count": state["total_escalations"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": sanitized_reason,
            "severity": severity,
            "source": source,
            "session_id": sanitized_session,
            "metadata": sanitized_metadata
        }

        escalations = state.setdefault("escalations", [])
        escalations.append(escalation_event)
        limit = self.config.metrics_history_limit
        if len(escalations) > limit:
            state["escalations"] = escalations[-limit:]

        self.save_state(state)
        logger.info(f"Recorded escalation #{state['total_escalations']}: {sanitized_reason} [{severity}]")
        return escalation_event

    def get_escalations(self) -> List[Dict[str, Any]]:
        """Returns list of recorded escalations."""
        state = self.load_state()
        return state.get("escalations", [])



class HealthMonitor:
    """
    Task 1: Phase 6 Knowledge Base Health & Integrity Evaluator.
    Determines HEALTHY / DEGRADED / UNHEALTHY state based on deterministic rules:
    - Verifies active pointer and production FAISS files
    - Checks latest quality evaluation scores
    - Checks recent failures, retries, and rollback history
    - Produces complete dashboard summary data
    """
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.config = config or PipelineConfig()
        self.collector = metrics_collector or MetricsCollector(config=self.config)

    def check_active_version_health(self) -> Dict[str, Any]:
        """Inspects the active version files and metadata."""
        active_file = self.config.active_version_file
        if not os.path.exists(active_file):
            return {"healthy": False, "reason": "active_version.json does not exist"}

        try:
            with open(active_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                active_v = data.get("active_version")
        except Exception as e:
            return {"healthy": False, "reason": f"Failed to read active_version.json: {e}"}

        if not active_v:
            return {"healthy": False, "reason": "active_version is null or empty"}

        active_dir = os.path.join(self.config.versions_dir, active_v)
        faiss_file = os.path.join(active_dir, "index.faiss")
        pkl_file = os.path.join(active_dir, "index.pkl")

        if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
            return {
                "healthy": False,
                "version": active_v,
                "reason": f"Missing FAISS index files in version directory '{active_v}'"
            }

        return {"healthy": True, "version": active_v}

    def evaluate_overall_health(self) -> Tuple[HealthStatus, List[str]]:
        """
        Deterministic Health Assessment:
        - UNHEALTHY: Active version missing, index files missing, or critical unhandled rollback.
        - DEGRADED: Active version valid, but recent failures, repeated retries, high latency, or pending maintenance.
        - HEALTHY: Normal operation, active index valid, latest quality passing, no critical errors.
        """
        reasons: List[str] = []

        # 1. Check Active Version Validity
        v_check = self.check_active_version_health()
        if not v_check["healthy"]:
            reasons.append(v_check["reason"])
            return HealthStatus.UNHEALTHY, reasons

        # 2. Check Metrics State
        state = self.collector.load_state()
        history = state.get("metrics_history", [])

        # Check last health check
        last_hc = state.get("last_health_check")
        if last_hc and not last_hc.get("passed", True) and not last_hc.get("rollback_triggered", False):
            reasons.append(f"Recent health check failed for version '{last_hc.get('version')}' without rollback")
            return HealthStatus.UNHEALTHY, reasons

        # Check consecutive failures
        if history:
            recent_fails = 0
            for run in reversed(history):
                if run.get("status") == PipelineExecutionStatus.FAILED.value:
                    recent_fails += 1
                else:
                    break
            if recent_fails >= self.config.max_recent_failures:
                reasons.append(f"{recent_fails} consecutive pipeline execution failures detected")
                return HealthStatus.UNHEALTHY, reasons

        # 3. Check for DEGRADED conditions
        # Elevated latency
        avg_lat = state.get("avg_latency_ms", 0.0)
        if avg_lat > self.config.max_pipeline_latency_ms:
            reasons.append(f"Average latency {avg_lat}ms exceeds threshold {self.config.max_pipeline_latency_ms}ms")

        # Retries
        if state.get("retry_count", 0) > 0 and history and history[-1].get("status") == PipelineExecutionStatus.RETRYING.value:
            reasons.append("Pipeline currently in retry backoff state")

        # Recent single failure
        if history and history[-1].get("status") == PipelineExecutionStatus.FAILED.value:
            reasons.append("Most recent pipeline run failed")

        if reasons:
            return HealthStatus.DEGRADED, reasons

        return HealthStatus.HEALTHY, ["Knowledge base is healthy and operating within all performance parameters."]

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """
        Returns a complete, sanitized operational summary ready for dashboards or monitoring APIs.
        """
        state = self.collector.load_state()
        history = state.get("metrics_history", [])
        overall_health, health_reasons = self.evaluate_overall_health()

        # Active version info
        active_check = self.check_active_version_health()
        active_version = active_check.get("version", "unknown")

        # Latest quality scores from versions.json metadata
        latest_retrieval = None
        latest_grounding = None
        latest_confidence = None
        if os.path.exists(self.config.versions_metadata_file):
            try:
                with open(self.config.versions_metadata_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    act_meta = meta.get("versions", {}).get(active_version, {})
                    eval_info = act_meta.get("evaluation", {})
                    latest_retrieval = eval_info.get("retrieval_score")
                    latest_grounding = eval_info.get("grounding_score")
                    latest_confidence = eval_info.get("response_confidence")
            except Exception:
                pass

        if latest_confidence is None:
            if latest_retrieval is not None or latest_grounding is not None:
                latest_confidence = calculate_application_confidence(latest_retrieval, latest_grounding)
            elif state.get("last_response_confidence") is not None:
                latest_confidence = state.get("last_response_confidence")

        # Security events count
        sec_logger = SecurityEventLogger(log_file=self.config.security_log_file)
        sec_events = sec_logger.load_events()

        # Scheduler state
        scheduler_status = {}
        if os.path.exists(self.config.scheduler_state_file):
            try:
                with open(self.config.scheduler_state_file, "r", encoding="utf-8") as f:
                    sch_state = json.load(f)
                    scheduler_status = {
                        "enabled": sch_state.get("enabled", True),
                        "last_run_at": sch_state.get("last_run_at"),
                        "pending_candidate": sch_state.get("pending_candidate"),
                        "active_retry": sch_state.get("active_retry")
                    }
            except Exception:
                pass

        last_run = history[-1] if history else {}

        summary = {
            "overall_health": overall_health.value,
            "health_reasons": health_reasons,
            "active_version": active_version,
            "last_pipeline_status": last_run.get("status"),
            "last_pipeline_duration_ms": last_run.get("duration_ms", 0.0),
            "total_runs": state.get("total_runs", 0),
            "successful_runs": state.get("successful_runs", 0),
            "failed_runs": state.get("failed_runs", 0),
            "retry_count": state.get("retry_count", 0),
            "quality_rejections": state.get("quality_rejections", 0),
            "rollback_count": state.get("rollback_count", 0),
            "total_escalations": state.get("total_escalations", 0),
            "average_latency_ms": state.get("avg_latency_ms", 0.0),
            "min_latency_ms": state.get("min_latency_ms", 0.0),
            "max_latency_ms": state.get("max_latency_ms", 0.0),
            "latest_retrieval_score": latest_retrieval,
            "latest_grounding_score": latest_grounding,
            "latest_response_confidence": latest_confidence,
            "response_confidence": latest_confidence,
            "security_event_count": len(sec_events),
            "scheduler_status": scheduler_status,
            "active_alerts": state.get("alerts", [])
        }

        return PIIMasker.mask_dict(summary)

    def record_confidence(self, *args, **kwargs) -> float:
        """Convenience forwarder to MetricsCollector.record_confidence."""
        return self.collector.record_confidence(*args, **kwargs)

    def record_escalation(self, *args, **kwargs) -> Dict[str, Any]:
        """Convenience forwarder to MetricsCollector.record_escalation."""
        return self.collector.record_escalation(*args, **kwargs)

    def get_escalations(self) -> List[Dict[str, Any]]:
        """Convenience forwarder to MetricsCollector.get_escalations."""
        return self.collector.get_escalations()

