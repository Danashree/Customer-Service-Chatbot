import os
import json
import logging
from datetime import datetime, time, timezone, timedelta
from typing import List, Dict, Optional, Callable, Any

from .config import PipelineConfig
from .document_processor import DocumentProcessor
from .vector_store_manager import VectorStoreManager
from .quality_evaluator import QualityEvaluator

logger = logging.getLogger("pipeline.scheduler")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Scheduler] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class PermanentPipelineError(Exception):
    """Non-retryable pipeline exception (e.g. fatal configuration or permission bug)."""
    pass


class KnowledgeBaseScheduler:
    """
    Task 1: Phase 4 Automated Pipeline Scheduler & Maintenance Window Controller.
    Orchestrates Phase 1 (Ingestion) -> Phase 2 (Candidate Creation) -> Phase 3 (Quality Gate).
    Applies 15/30/60 minute retry backoff for transient failures, checks configurable
    maintenance windows (including midnight-crossing windows), and ensures failed candidates
    never compromise the active knowledge base.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        document_processor: Optional[DocumentProcessor] = None,
        vector_store_manager: Optional[VectorStoreManager] = None,
        quality_evaluator: Optional[QualityEvaluator] = None,
        time_provider: Optional[Callable[[], datetime]] = None
    ):
        self.config = config or PipelineConfig()
        self.doc_processor = document_processor or DocumentProcessor(config=self.config)
        self.vsm = vector_store_manager or VectorStoreManager(config=self.config)
        self.evaluator = quality_evaluator or QualityEvaluator(config=self.config, vector_store_manager=self.vsm)
        # Injectable time provider for deterministic tests without sleeping
        self.time_provider = time_provider or (lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Time & Maintenance Window Checking
    # ------------------------------------------------------------------
    def get_current_time(self) -> datetime:
        """Returns the current timezone-aware timestamp via time_provider."""
        dt = self.time_provider()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def is_in_maintenance_window(self, dt: Optional[datetime] = None) -> bool:
        """
        Evaluates whether the provided datetime (or current time) falls within
        the configured maintenance window. Correctly handles windows that cross midnight.
        Example windows:
          '02:00' -> '04:00' (Standard window)
          '22:00' -> '02:00' (Crosses midnight: 23:30 is inside, 01:30 is inside, 03:00 is outside)
        """
        check_dt = dt or self.get_current_time()
        curr_t = check_dt.time()

        try:
            start_parts = [int(p) for p in self.config.maintenance_window_start.split(":")]
            end_parts = [int(p) for p in self.config.maintenance_window_end.split(":")]
            start_t = time(start_parts[0], start_parts[1])
            end_t = time(end_parts[0], end_parts[1])
        except Exception as e:
            logger.error(f"Failed to parse maintenance window bounds ({e}). Defaulting to open.")
            return True

        # Standard window: start <= end (e.g. 02:00 to 04:00)
        if start_t <= end_t:
            return start_t <= curr_t < end_t
        # Midnight-crossing window: start > end (e.g. 22:00 to 02:00)
        else:
            return curr_t >= start_t or curr_t < end_t

    # ------------------------------------------------------------------
    # Persistent State Management
    # ------------------------------------------------------------------
    def load_state(self) -> Dict[str, Any]:
        """Loads persistent scheduler state JSON."""
        state_file = self.config.scheduler_state_file
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not read scheduler state ({e}). Starting fresh.")

        return {
            "enabled": self.config.scheduler_enabled,
            "last_run_at": None,
            "last_result": None,
            "next_scheduled_run_at": None,
            "pending_candidate": None,
            "active_retry": None,
            "audit_history": []
        }

    def save_state(self, state: Dict[str, Any]) -> None:
        """Persists scheduler state JSON atomically."""
        state_file = self.config.scheduler_state_file
        state_dir = os.path.dirname(state_file)
        if state_dir and not os.path.exists(state_dir):
            os.makedirs(state_dir, exist_ok=True)

        # Append to audit history if last_result exists
        if state.get("last_result"):
            history = state.setdefault("audit_history", [])
            history.append({
                "timestamp": self.get_current_time().isoformat(),
                "result": state["last_result"]
            })
            # Cap audit history to last 50 entries
            if len(history) > 50:
                state["audit_history"] = history[-50:]

        tmp_file = f"{state_file}.tmp_{os.getpid()}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_file, state_file)

    # ------------------------------------------------------------------
    # Transient Failure & Retry Handler
    # ------------------------------------------------------------------
    def _handle_transient_failure(self, err: Exception, now: datetime, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Schedules retries after 15, 30, and 60 minutes.
        If all retries are exhausted, marks the operation as failed.
        """
        active_retry = state.get("active_retry")
        current_attempt = active_retry.get("attempt", 0) if active_retry else 0
        delays = list(self.config.retry_delays_minutes)
        max_attempts = len(delays)

        if current_attempt < max_attempts:
            delay_min = delays[current_attempt]
            next_retry = now + timedelta(minutes=delay_min)
            new_retry_state = {
                "attempt": current_attempt + 1,
                "max_attempts": max_attempts,
                "next_retry_at": next_retry.isoformat(),
                "delay_minutes": delay_min,
                "last_error": str(err),
                "status": "pending_retry"
            }
            state["active_retry"] = new_retry_state
            result = {
                "status": "retry_scheduled",
                "attempt": current_attempt + 1,
                "max_attempts": max_attempts,
                "next_retry_at": next_retry.isoformat(),
                "delay_minutes": delay_min,
                "error": str(err)
            }
            logger.warning(
                f"Transient failure (attempt {current_attempt + 1}/{max_attempts}): {err}. "
                f"Next retry scheduled in {delay_min} minutes at {next_retry.isoformat()}."
            )
        else:
            # Retries exhausted
            state["active_retry"] = None
            active_v = self.vsm.get_active_version()
            result = {
                "status": "failed",
                "attempts": current_attempt,
                "max_attempts": max_attempts,
                "last_error": str(err),
                "active_version": active_v,
                "reason": "Retry attempts exhausted"
            }
            logger.error(f"All {max_attempts} retry attempts exhausted. Final failure: {err}. Active version: {active_v}")

        state["last_result"] = result
        state["last_run_at"] = now.isoformat()
        self.save_state(state)
        return result

    # ------------------------------------------------------------------
    # Single-Cycle Execution Engine
    # ------------------------------------------------------------------
    def run_once(
        self,
        force_run: bool = False,
        simulated_failure: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """
        Executes a single scheduled cycle of the update & monitoring pipeline:
        1. Checks scheduler enabled state.
        2. Evaluates pending retries (runs retry if due).
        3. Checks pending candidates waiting for maintenance window (activates if window open).
        4. Ingests knowledge base via Phase 1 DocumentProcessor.
        5. If changes detected, creates candidate version via Phase 2 VectorStoreManager.
        6. Evaluates candidate quality via Phase 3 QualityEvaluator.
        7. If quality approved:
           - If in maintenance window: activates candidate via Phase 2 and runs health check hook.
           - If outside maintenance window: defers activation (queues pending candidate).
        8. On transient error: applies 15, 30, 60 minute retry backoff.
        """
        now = self.get_current_time()
        state = self.load_state()

        # 1. Check enabled
        if not self.config.scheduler_enabled and not force_run:
            logger.info("Scheduler is disabled. Skipping cycle.")
            return {"status": "disabled"}

        # 2. Check pending retry
        active_retry = state.get("active_retry")
        if active_retry and not force_run:
            next_retry_at = datetime.fromisoformat(active_retry["next_retry_at"])
            if now < next_retry_at:
                logger.info(f"Pending retry not yet due (due at {next_retry_at.isoformat()}, current {now.isoformat()}).")
                return {
                    "status": "waiting_for_retry",
                    "attempt": active_retry["attempt"],
                    "next_retry_at": active_retry["next_retry_at"],
                    "last_error": active_retry.get("last_error")
                }

        # 3. Pipeline Execution (with error & retry handling)
        try:
            # Handle simulated transient failure for testing retry mechanics
            if simulated_failure:
                raise simulated_failure

            # Check pending candidate waiting for maintenance window
            pending_candidate = state.get("pending_candidate")
            if pending_candidate:
                candidate_v = pending_candidate["version"]
                if self.is_in_maintenance_window(now):
                    logger.info(f"Maintenance window is OPEN. Activating queued candidate '{candidate_v}'...")
                    prev_v = self.vsm.get_active_version()
                    act_ok = self.vsm.activate_version(candidate_v)
                    health_res = self.vsm.schedule_post_activation_health_check(
                        version=candidate_v,
                        previous_version=prev_v,
                        delay_seconds=0
                    )
                    state["pending_candidate"] = None
                    state["active_retry"] = None
                    result = {
                        "status": "activated",
                        "version": candidate_v,
                        "previous_version": prev_v,
                        "activation_success": act_ok,
                        "health_check": health_res
                    }
                    state["last_result"] = result
                    state["last_run_at"] = now.isoformat()
                    self.save_state(state)
                    return result
                else:
                    logger.info(f"Candidate '{candidate_v}' is waiting for maintenance window.")
                    result = {
                        "status": "waiting_for_maintenance_window",
                        "version": candidate_v,
                        "message": "Candidate approved by quality gate but current time is outside maintenance window"
                    }
                    state["last_result"] = result
                    state["last_run_at"] = now.isoformat()
                    self.save_state(state)
                    return result

            # Phase 1: Ingestion & Discovery
            doc_res = self.doc_processor.process_knowledge_base()

            # Check if valid new chunks exist
            processed_count = len(doc_res.processed_documents)
            chunk_count = len(doc_res.chunks)

            if processed_count == 0 or chunk_count == 0:
                logger.info("No valid newly processed documents or chunks detected. Knowledge base is up-to-date.")
                state["active_retry"] = None
                result = {
                    "status": "no_changes",
                    "active_version": self.vsm.get_active_version()
                }
                state["last_result"] = result
                state["last_run_at"] = now.isoformat()
                self.save_state(state)
                return result

            # Phase 2: Create Candidate Version
            logger.info(f"Detected {processed_count} processed doc(s) with {chunk_count} chunk(s). Building candidate version...")
            candidate_v = self.vsm.create_version(
                documents=doc_res.chunks,
                source_filenames=doc_res.processed_documents
            )

            # Phase 3: Quality Evaluation Gate
            logger.info(f"Running Phase 3 quality gate for candidate '{candidate_v}'...")
            eval_report = self.evaluator.evaluate_candidate(candidate_v)

            if eval_report.get("decision") != "APPROVED":
                # Quality rejection is a final decision, NOT a transient failure.
                # Must NOT consume retry attempts.
                state["active_retry"] = None
                active_v = self.vsm.get_active_version()
                logger.warning(f"Candidate '{candidate_v}' REJECTED by quality gate. Keeping active version '{active_v}'.")
                result = {
                    "status": "quality_rejected",
                    "version": candidate_v,
                    "active_version": active_v,
                    "failure_reasons": eval_report.get("failure_reasons", [])
                }
                state["last_result"] = result
                state["last_run_at"] = now.isoformat()
                self.save_state(state)
                return result

            # Phase 4: Maintenance Window Check
            if self.is_in_maintenance_window(now):
                logger.info(f"Candidate '{candidate_v}' APPROVED and current time is INSIDE maintenance window. Activating...")
                prev_v = self.vsm.get_active_version()
                act_ok = self.vsm.activate_version(candidate_v)
                health_res = self.vsm.schedule_post_activation_health_check(
                    version=candidate_v,
                    previous_version=prev_v,
                    delay_seconds=0
                )
                state["pending_candidate"] = None
                state["active_retry"] = None
                result = {
                    "status": "activated",
                    "version": candidate_v,
                    "previous_version": prev_v,
                    "activation_success": act_ok,
                    "health_check": health_res
                }
            else:
                logger.info(
                    f"Candidate '{candidate_v}' APPROVED but current time is OUTSIDE maintenance window. "
                    f"Queueing candidate for next maintenance window."
                )
                state["pending_candidate"] = {
                    "version": candidate_v,
                    "created_at": now.isoformat(),
                    "quality_status": "approved"
                }
                state["active_retry"] = None
                result = {
                    "status": "waiting_for_maintenance_window",
                    "version": candidate_v,
                    "message": "Candidate approved by quality gate but current time is outside maintenance window"
                }

            state["last_result"] = result
            state["last_run_at"] = now.isoformat()
            self.save_state(state)
            return result

        except PermanentPipelineError as pe:
            # Explicit permanent failure - do not retry
            logger.error(f"Permanent pipeline error encountered: {pe}")
            state["active_retry"] = None
            result = {
                "status": "failed",
                "error": str(pe),
                "is_permanent": True,
                "active_version": self.vsm.get_active_version()
            }
            state["last_result"] = result
            state["last_run_at"] = now.isoformat()
            self.save_state(state)
            return result

        except Exception as exc:
            # Transient error - trigger retry logic
            return self._handle_transient_failure(exc, now, state)

    # ------------------------------------------------------------------
    # Scheduler Status Reporting
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Returns structured diagnostic status dictionary of the scheduler."""
        state = self.load_state()
        now = self.get_current_time()
        in_window = self.is_in_maintenance_window(now)

        return {
            "enabled": self.config.scheduler_enabled,
            "current_time": now.isoformat(),
            "in_maintenance_window": in_window,
            "maintenance_window": {
                "start": self.config.maintenance_window_start,
                "end": self.config.maintenance_window_end
            },
            "schedule_interval_minutes": self.config.schedule_interval_minutes,
            "retry_delays_minutes": list(self.config.retry_delays_minutes),
            "active_version": self.vsm.get_active_version(),
            "last_run_at": state.get("last_run_at"),
            "last_result": state.get("last_result"),
            "pending_candidate": state.get("pending_candidate"),
            "active_retry": state.get("active_retry"),
            "history_count": len(state.get("audit_history", []))
        }
