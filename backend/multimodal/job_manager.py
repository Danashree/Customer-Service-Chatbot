import os
import time
import uuid
import json
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, Callable, Set

from .config import MultimodalConfig
from .validator import MultimodalValidator, MultimodalValidationError
from .extractor import DocumentExtractor
from .comparator import EvidenceComparator
from .security import MultimodalSecurityGuard
import threading

logger = logging.getLogger("multimodal.job_manager")


def execute_multimodal_pipeline(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    message: Optional[str] = None,
    file_id: Optional[str] = None,
    config: Optional[MultimodalConfig] = None,
) -> Dict[str, Any]:
    """
    Executes the full Phase 1-4 multimodal analysis pipeline.
    Reused identically across synchronous and background worker flows.
    """
    cfg = config or MultimodalConfig()
    validator = MultimodalValidator(config=cfg)

    # Phase 1: validate + store
    save_result = validator.save_upload(
        file_bytes=file_bytes,
        filename=filename,
        content_type=content_type,
    )
    actual_file_id = file_id or save_result["file_id"]

    # Phase 2: extract / OCR
    extractor = DocumentExtractor()
    extraction = extractor.extract(
        file_bytes=file_bytes,
        filename=filename,
        file_id=actual_file_id,
        content_type=content_type,
    )

    # Phase 4: Customer message security & prompt injection scan
    if message:
        msg_injection = MultimodalSecurityGuard.scan_prompt_injection(
            message,
            source="customer_message",
            file_id=actual_file_id,
            security_logger=validator.security_logger,
        )
        if msg_injection["detected"]:
            extraction.prompt_injection_detected = True
            extraction.security_notes.append(
                f"Prompt injection detected in customer message ({', '.join(msg_injection['categories'])})."
            )

    # Phase 3: compare customer message vs evidence
    comparator = EvidenceComparator()
    comparison = comparator.compare(
        customer_message=message,
        extracted_evidence=extraction.extracted_fields,
        evidence_quality=extraction.evidence_quality,
        quality_score=extraction.quality_score,
    )

    effective_clarification = extraction.clarification_required or comparison.clarification_required
    effective_user_msg = comparison.clarification_message or extraction.user_message

    return {
        "status": "accepted",
        "file_id": actual_file_id,
        "original_filename": save_result["original_filename"],
        "content_type": save_result["content_type"],
        "file_size_bytes": save_result["file_size_bytes"],
        "metadata": save_result.get("metadata", {}),
        "document_type": extraction.document_type,
        "extraction_method": extraction.extraction_method,
        "pages_processed": extraction.pages_processed,
        "extracted_fields": extraction.extracted_fields,
        "evidence_quality": extraction.evidence_quality,
        "quality_score": extraction.quality_score,
        "quality_notes": extraction.quality_notes,
        "clarification_required": effective_clarification,
        "user_message": effective_user_msg,
        "processing_time_ms": extraction.processing_time_ms,
        "comparison": comparison.to_dict(),
        "security_status": {
            "prompt_injection_detected": extraction.prompt_injection_detected,
            "pii_masked": True,
            "safe": True,
            "security_notes": extraction.security_notes,
        },
        "message": "File successfully uploaded, validated, and processed.",
    }


def compute_request_fingerprint(file_bytes: bytes, message: Optional[str] = None) -> str:
    """
    Computes a deterministic SHA-256 fingerprint from file content and normalized customer message.
    Never stores or logs raw sensitive data.
    """
    hasher = hashlib.sha256()
    hasher.update(file_bytes)
    normalized_message = (message or "").strip().lower()
    hasher.update(normalized_message.encode("utf-8"))
    return hasher.hexdigest()


ALLOWED_JOB_STATUSES = {"queued", "processing", "completed", "retrying", "failed", "cancelled"}
ACTIVE_JOB_STATUSES = {"queued", "processing", "retrying"}


@dataclass
class Job:
    job_id: str
    file_id: str
    fingerprint: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    status_message: str = "Job queued."
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_mode: str = "auto"
    file_retained: bool = True

    def to_safe_dict(self) -> Dict[str, Any]:
        """
        Returns client-safe job metadata without internal paths, traces, or secrets.
        """
        data: Dict[str, Any] = {
            "job_id": self.job_id,
            "file_id": self.file_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status_message": self.status_message,
            "error": self.error,
            "file_retained": self.file_retained,
        }
        if self.status == "completed" and self.result is not None:
            data["result"] = self.result
        return data


class JobManager:
    """
    Thread-safe background job manager with in-memory state, optional atomic persistence,
    retry management, duplicate active-job protection, and threshold estimation.
    """

    def __init__(
        self,
        config: Optional[MultimodalConfig] = None,
        max_workers: int = 2,
        pipeline_executor: Optional[Callable[..., Dict[str, Any]]] = None,
        time_estimator: Optional[Callable[..., float]] = None,
        enable_persistence: bool = True,
    ):
        self.config = config or MultimodalConfig()
        self.max_workers = max_workers
        self.pipeline_executor = pipeline_executor or execute_multimodal_pipeline
        self.time_estimator = time_estimator or self._default_time_estimator
        self.enable_persistence = enable_persistence
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._fingerprints: Dict[str, str] = {}  # fingerprint -> job_id
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="multimodal_worker")
        self._persistence_file = os.path.join(self.config.upload_dir, "jobs.json")
        if self.enable_persistence:
            self._load_persisted_jobs()

    def _default_time_estimator(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        message: Optional[str] = None,
    ) -> float:
        """
        Default processing time estimator. Small images/docs take ~0.5s.
        Multi-page or large files estimate higher.
        """
        size_kb = len(file_bytes) / 1024.0
        # Fast baseline for typical files
        est = 0.5 + (size_kb / 5000.0)
        return est

    def set_time_estimator(self, estimator: Callable[..., float]):
        """Inject a custom processing time estimator (e.g. for testing threshold behavior)."""
        self.time_estimator = estimator

    def is_long_running(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        message: Optional[str] = None,
    ) -> bool:
        """
        Determines whether processing is expected to exceed the configured threshold.
        """
        try:
            estimated_time = self.time_estimator(
                file_bytes=file_bytes,
                filename=filename,
                content_type=content_type,
                message=message,
            )
            return estimated_time >= self.config.multimodal_processing_threshold_seconds
        except Exception:
            return False

    def create_and_enqueue_job(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        message: Optional[str] = None,
        processing_mode: str = "auto",
    ) -> Tuple[Job, bool]:
        """
        Creates or retrieves an active background job.
        Returns (job, is_existing).
        If an identical active request exists, returns the existing job.
        """
        fingerprint = compute_request_fingerprint(file_bytes, message)

        with self._lock:
            existing_job_id = self._fingerprints.get(fingerprint)
            if existing_job_id and existing_job_id in self._jobs:
                existing_job = self._jobs[existing_job_id]
                if existing_job.status in ACTIVE_JOB_STATUSES:
                    return existing_job, True

            job_id = str(uuid.uuid4())
            file_id = str(uuid.uuid4())
            job = Job(
                job_id=job_id,
                file_id=file_id,
                fingerprint=fingerprint,
                status="queued",
                max_retries=self.config.max_background_retries,
                status_message="Your file has been queued for background processing.",
                processing_mode=processing_mode,
            )
            self._jobs[job_id] = job
            self._fingerprints[fingerprint] = job_id
            self._persist_jobs()

        # Submit to background executor
        payload = {
            "file_bytes": file_bytes,
            "filename": filename,
            "content_type": content_type,
            "message": message,
        }
        self._executor.submit(self._worker_run_job, job_id, payload)
        return job, False

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by its UUID."""
        with self._lock:
            return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel an active or queued job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in ACTIVE_JOB_STATUSES:
                job.status = "cancelled"
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.status_message = "Job was cancelled."
                self._persist_jobs()
                return True
            return False

    def get_active_file_ids(self) -> Set[str]:
        """Return set of file_ids associated with currently active jobs (queued, processing, retrying)."""
        with self._lock:
            return {
                job.file_id
                for job in self._jobs.values()
                if job.status in ACTIVE_JOB_STATUSES and job.file_id
            }

    def get_all_job_file_ids(self) -> Set[str]:
        """Return set of all file_ids across all known jobs."""
        with self._lock:
            return {job.file_id for job in self._jobs.values() if job.file_id}

    def mark_file_cleaned(self, file_id: str):
        """Mark job records referencing file_id as file_retained=False."""
        with self._lock:
            updated = False
            for job in self._jobs.values():
                if job.file_id == file_id:
                    job.file_retained = False
                    updated = True
            if updated:
                self._persist_jobs()

    def _worker_run_job(self, job_id: str, payload: Dict[str, Any]):
        """
        Background worker execution loop with deterministic state transitions and retries.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status == "cancelled":
                return
            job.status = "processing"
            job.started_at = datetime.now(timezone.utc).isoformat()
            job.status_message = "Processing file."
            self._persist_jobs()

        attempt = 0
        max_retries = job.max_retries
        retry_delays = self.config.background_retry_delays

        while True:
            try:
                result = self.pipeline_executor(
                    file_bytes=payload["file_bytes"],
                    filename=payload["filename"],
                    content_type=payload["content_type"],
                    message=payload["message"],
                    file_id=job.file_id,
                    config=self.config,
                )
                with self._lock:
                    job.status = "completed"
                    job.completed_at = datetime.now(timezone.utc).isoformat()
                    job.status_message = "Job completed successfully."
                    job.result = result
                    job.error = None
                    self._persist_jobs()
                return

            except MultimodalValidationError as ve:
                # Permanent validation failure: do not retry
                safe_err = MultimodalSecurityGuard.sanitize_error_message(str(ve))
                with self._lock:
                    job.status = "failed"
                    job.completed_at = datetime.now(timezone.utc).isoformat()
                    job.status_message = "Job failed due to validation error."
                    job.error = safe_err
                    self._persist_jobs()
                return

            except Exception as e:
                safe_err = MultimodalSecurityGuard.sanitize_error_message(str(e))
                if "[INTERNAL_PATH]" in safe_err or "Traceback" in safe_err:
                    safe_err = "Unable to process the uploaded file due to an internal error."

                attempt += 1
                if attempt <= max_retries:
                    # Delay before retry
                    delay_idx = attempt - 1
                    delay = retry_delays[delay_idx] if delay_idx < len(retry_delays) else retry_delays[-1]

                    with self._lock:
                        job.status = "retrying"
                        job.retry_count = attempt
                        job.status_message = f"Transient failure, retrying (attempt {attempt}/{max_retries})."
                        job.error = safe_err
                        self._persist_jobs()

                    time.sleep(delay)

                    with self._lock:
                        if job.status == "cancelled":
                            return
                        job.status = "processing"
                        job.status_message = f"Processing retry attempt {attempt}/{max_retries}."
                        self._persist_jobs()
                else:
                    # All retries exhausted
                    with self._lock:
                        job.status = "failed"
                        job.retry_count = max_retries
                        job.completed_at = datetime.now(timezone.utc).isoformat()
                        job.status_message = "Job failed after maximum retries."
                        job.error = safe_err
                        self._persist_jobs()
                    return

    def _persist_jobs(self):
        """Atomically persist jobs to disk safely without leaking paths or blocking on error."""
        if not self.enable_persistence:
            return
        try:
            data = {}
            for jid, j in self._jobs.items():
                data[jid] = {
                    "job_id": j.job_id,
                    "file_id": j.file_id,
                    "fingerprint": j.fingerprint,
                    "status": j.status,
                    "created_at": j.created_at,
                    "started_at": j.started_at,
                    "completed_at": j.completed_at,
                    "retry_count": j.retry_count,
                    "max_retries": j.max_retries,
                    "status_message": j.status_message,
                    "result": j.result,
                    "error": j.error,
                    "processing_mode": j.processing_mode,
                    "file_retained": j.file_retained,
                }
            tmp_file = f"{self._persistence_file}.tmp.{uuid.uuid4().hex}"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_file, self._persistence_file)
        except Exception:
            # Graceful non-blocking error handling
            pass

    def _load_persisted_jobs(self):
        """Recover state gracefully from persistent storage if present."""
        if not self.enable_persistence or not os.path.exists(self._persistence_file):
            return
        try:
            with open(self._persistence_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for jid, d in data.items():
                status = d.get("status", "queued")
                error = d.get("error")
                status_msg = d.get("status_message", "")
                if status in ACTIVE_JOB_STATUSES:
                    status = "failed"
                    status_msg = "Server restarted before job completed."
                    error = "Unable to process the uploaded file due to an interrupted session."

                job = Job(
                    job_id=d["job_id"],
                    file_id=d["file_id"],
                    fingerprint=d["fingerprint"],
                    status=status,
                    created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
                    started_at=d.get("started_at"),
                    completed_at=d.get("completed_at"),
                    retry_count=d.get("retry_count", 0),
                    max_retries=d.get("max_retries", 3),
                    status_message=status_msg,
                    result=d.get("result"),
                    error=error,
                    processing_mode=d.get("processing_mode", "auto"),
                    file_retained=d.get("file_retained", True),
                )
                self._jobs[jid] = job
        except Exception:
            # Graceful recovery from corrupt/unreadable file
            pass

    def shutdown(self, wait: bool = False):
        """Clean shutdown of background executor."""
        self._executor.shutdown(wait=wait)
