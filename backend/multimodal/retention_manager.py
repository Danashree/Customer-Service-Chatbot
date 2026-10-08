import os
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Set, Callable

from .config import MultimodalConfig
from .job_manager import JobManager
from .security import MultimodalSecurityGuard
from pipeline.security import SecurityEventLogger

logger = logging.getLogger("multimodal.retention_manager")


class RetentionManager:
    """
    Manages the lifecycle, retention, and secure cleanup of uploaded multimodal files.
    Ensures safe directory containment, active-job protection, idempotent operations,
    audit event logging, and non-blocking background scheduling.
    """

    def __init__(
        self,
        config: Optional[MultimodalConfig] = None,
        job_manager: Optional[JobManager] = None,
        security_logger: Optional[SecurityEventLogger] = None,
        time_provider: Optional[Callable[[], float]] = None,
    ):
        self.config = config or MultimodalConfig()
        self.job_manager = job_manager
        self.security_logger = security_logger or SecurityEventLogger()
        self.time_provider = time_provider or time.time

        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def get_current_time(self) -> float:
        """Returns the current timestamp via the injected time provider."""
        return self.time_provider()

    def is_expired(self, file_path: str, now: Optional[float] = None) -> bool:
        """
        Determines whether an uploaded file has exceeded the configured retention period.
        Boundary behavior: age >= retention_hours is expired; age < retention_hours is kept.
        """
        try:
            current_time = now if now is not None else self.get_current_time()
            stat = os.stat(file_path)
            # Use mtime as reliable lifecycle timestamp
            file_mtime = stat.st_mtime
            age_seconds = current_time - file_mtime
            retention_seconds = self.config.retention_hours * 3600.0
            return age_seconds >= retention_seconds
        except Exception:
            return False

    def is_protected(self, file_id: str) -> bool:
        """
        Determines whether a file is actively protected from deletion.
        Files belonging to queued, processing, or retrying jobs are strictly protected.
        """
        if not self.config.active_job_protection or not self.job_manager:
            return False
        try:
            active_ids = self.job_manager.get_active_file_ids()
            return file_id in active_ids
        except Exception:
            # In case of check error, default to protecting active file
            return True

    def verify_directory_containment(self, file_path: str) -> bool:
        """
        Verifies that a candidate file path is strictly contained within the upload directory.
        Protects against directory traversal, absolute path injection, symlink escapes,
        and system/metadata files like jobs.json.
        """
        try:
            if not file_path:
                return False

            base_name = os.path.basename(file_path)
            # Never delete metadata or state files
            if base_name in ("jobs.json", "jobs.json.bak") or base_name.startswith("jobs.json.tmp"):
                return False

            # Normalize and resolve real paths to prevent symlink traversal
            real_upload_dir = os.path.realpath(os.path.abspath(self.config.upload_dir))
            real_file_path = os.path.realpath(os.path.abspath(file_path))

            # Must reside strictly within the real upload directory
            try:
                common = os.path.commonpath([real_file_path, real_upload_dir])
            except ValueError:
                # Different drives on Windows
                return False

            if common != real_upload_dir:
                return False

            # Ensure file is not the upload directory itself
            if real_file_path == real_upload_dir:
                return False

            return True
        except Exception:
            return False

    def find_expired_files(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Scans the upload directory and identifies candidate expired files.
        """
        expired_files = []
        if not os.path.exists(self.config.upload_dir):
            return expired_files

        current_time = now if now is not None else self.get_current_time()

        try:
            with os.scandir(self.config.upload_dir) as entries:
                for entry in entries:
                    try:
                        if not entry.is_file():
                            continue

                        if not self.verify_directory_containment(entry.path):
                            continue

                        if self.is_expired(entry.path, now=current_time):
                            filename = entry.name
                            file_id = os.path.splitext(filename)[0]
                            expired_files.append({
                                "file_id": file_id,
                                "filename": filename,
                                "path": entry.path,
                                "is_protected": self.is_protected(file_id),
                            })
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("Error scanning upload directory for expired files: %s", e)

        return expired_files

    def cleanup_expired_files(self, now: Optional[float] = None) -> Dict[str, int]:
        """
        Executes a complete cleanup pass:
        - Scans upload directory
        - Validates safe containment
        - Checks expiration
        - Honors active job protection
        - Unlinks expired files
        - Updates job records
        - Logs sanitized audit events
        - Tolerates missing files and isolated failures idempotently

        Returns a sanitized summary dictionary without internal filesystem paths.
        """
        summary = {
            "scanned": 0,
            "expired": 0,
            "deleted": 0,
            "protected": 0,
            "skipped": 0,
            "failed": 0,
        }

        if not os.path.exists(self.config.upload_dir):
            return summary

        current_time = now if now is not None else self.get_current_time()

        with self._lock:
            try:
                entries = list(os.scandir(self.config.upload_dir))
            except Exception as e:
                logger.error("Failed to list upload directory for cleanup: %s", e)
                return summary

            for entry in entries:
                try:
                    if not entry.is_file():
                        continue

                    # Filter out metadata files
                    if entry.name in ("jobs.json", "jobs.json.bak") or entry.name.startswith("jobs.json.tmp"):
                        continue

                    summary["scanned"] += 1
                    file_path = entry.path
                    filename = entry.name
                    file_id = os.path.splitext(filename)[0]

                    # 1. Path & Containment Safety
                    if not self.verify_directory_containment(file_path):
                        summary["skipped"] += 1
                        self._log_event(
                            event_type="MULTIMODAL_FILE_CLEANUP_PATH_SECURITY_FAILURE",
                            severity="CRITICAL",
                            action="cleanup_containment_check",
                            result="BLOCKED",
                            resource=file_id or "unknown_file",
                            details={"reason": "Path traversal or directory containment violation detected."},
                        )
                        continue

                    # 2. Expiration Check
                    if not self.is_expired(file_path, now=current_time):
                        # File is within retention period, keep it
                        continue

                    summary["expired"] += 1
                    self._log_event(
                        event_type="MULTIMODAL_FILE_RETENTION_EXPIRED",
                        severity="INFO",
                        action="retention_evaluation",
                        result="EXPIRED",
                        resource=file_id,
                        details={"reason": f"File exceeded retention period of {self.config.retention_hours} hours."},
                    )

                    # 3. Active Job Protection
                    if self.is_protected(file_id):
                        summary["protected"] += 1
                        self._log_event(
                            event_type="MULTIMODAL_FILE_CLEANUP_PROTECTED",
                            severity="INFO",
                            action="cleanup_protection_check",
                            result="PROTECTED",
                            resource=file_id,
                            details={"reason": "File is currently required by an active job."},
                        )
                        continue

                    # 4. Safe Deletion
                    try:
                        os.remove(file_path)
                        summary["deleted"] += 1
                        if self.job_manager:
                            self.job_manager.mark_file_cleaned(file_id)

                        self._log_event(
                            event_type="MULTIMODAL_FILE_CLEANUP_DELETED",
                            severity="INFO",
                            action="cleanup_delete",
                            result="SUCCESS",
                            resource=file_id,
                            details={"reason": "Expired file safely deleted."},
                        )
                    except FileNotFoundError:
                        # Idempotent: file was already removed
                        summary["skipped"] += 1
                        self._log_event(
                            event_type="MULTIMODAL_FILE_CLEANUP_SKIPPED",
                            severity="INFO",
                            action="cleanup_delete",
                            result="SKIPPED",
                            resource=file_id,
                            details={"reason": "File already removed or missing."},
                        )
                    except Exception as del_err:
                        summary["failed"] += 1
                        safe_err = MultimodalSecurityGuard.sanitize_error_message(str(del_err))
                        self._log_event(
                            event_type="MULTIMODAL_FILE_CLEANUP_FAILED",
                            severity="ERROR",
                            action="cleanup_delete",
                            result="FAILED",
                            resource=file_id,
                            details={"reason": f"Deletion failed: {safe_err}"},
                        )

                except Exception as loop_err:
                    summary["failed"] += 1
                    logger.warning("Error processing cleanup for entry: %s", loop_err)
                    continue

        return summary

    def run_startup_cleanup(self) -> Dict[str, int]:
        """
        Safely executes startup cleanup without blocking or raising exceptions that could halt startup.
        """
        try:
            if self.config.cleanup_enabled:
                return self.cleanup_expired_files()
        except Exception as e:
            logger.warning("Startup cleanup encountered an error: %s", e)
        return {"scanned": 0, "expired": 0, "deleted": 0, "protected": 0, "skipped": 0, "failed": 0}

    def start_scheduler(self):
        """
        Launches a background daemon thread that periodically executes cleanup_expired_files.
        """
        if not self.config.cleanup_enabled or self._scheduler_thread is not None:
            return

        self._stop_event.clear()

        def _scheduler_loop():
            interval_seconds = max(1.0, float(self.config.cleanup_interval_minutes * 60))
            while not self._stop_event.is_set():
                # Wait for interval or stop event
                if self._stop_event.wait(timeout=interval_seconds):
                    break
                try:
                    self.cleanup_expired_files()
                except Exception as e:
                    logger.warning("Periodic cleanup encountered an error: %s", e)

        self._scheduler_thread = threading.Thread(
            target=_scheduler_loop,
            daemon=True,
            name="multimodal_cleanup_scheduler",
        )
        self._scheduler_thread.start()

    def stop_scheduler(self, timeout: float = 2.0):
        """
        Stops the periodic cleanup scheduler cleanly.
        """
        self._stop_event.set()
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=timeout)
        self._scheduler_thread = None

    def _log_event(
        self,
        event_type: str,
        severity: str,
        action: str,
        result: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Logs a sanitized security/cleanup event using SecurityEventLogger.
        Ensures no absolute filesystem paths, unmasked PII, or secrets are ever recorded.
        """
        try:
            safe_details = MultimodalSecurityGuard.mask_sensitive_dict(details or {})
            self.security_logger.log_event(
                event_type=event_type,
                severity=severity,
                action=action,
                result=result,
                resource=resource,
                details=safe_details,
            )
        except Exception:
            pass
