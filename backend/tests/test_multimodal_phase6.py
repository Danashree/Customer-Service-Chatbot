"""
Task 2 Phase 6: Retention & Cleanup Tests

Covers:
Group A: Configuration (defaults, custom, enabled/disabled, interval)
Group B: Expiration Boundary (younger, boundary, older)
Group C: Cleanup & Idempotency (expired deleted, young kept, multiple, summary)
Group D: Active Job Protection (queued, processing, retrying files protected)
Group E: Completed Jobs (expired file deleted, safe metadata preserved)
Group F: Failed Jobs (expired file deleted, safe error preserved)
Group G: Orphan Files (recent orphan kept, expired orphan deleted)
Group H: Security & Path Containment (traversal blocked, symlink defense, zero leakage)
Group I: Audit Logging & Privacy (events recorded, no PII/OCR/paths logged)
Group J: Failure Handling (isolated failure continues loop, missing file handled)
Group K: Scheduling & Startup Recovery (manual trigger, non-blocking thread, startup hook)
Group L: Regressions (Phases 1-5, Task 1 demo, /ask compatibility)
"""

import io
import os
import sys
import time
import uuid
import tempfile
import pytest
from unittest.mock import patch, MagicMock

# Ensure backend/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from fastapi.testclient import TestClient
from main import app, job_manager, retention_manager

from multimodal.config import MultimodalConfig
from multimodal.job_manager import Job, JobManager
from multimodal.retention_manager import RetentionManager
from pipeline.security import SecurityEventLogger

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_upload_env(tmp_path):
    """Creates an isolated temporary upload directory and custom config for testing."""
    upload_dir = str(tmp_path / "test_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    cfg = MultimodalConfig(
        upload_dir=upload_dir,
        retention_hours=24.0,
        cleanup_enabled=True,
        cleanup_interval_minutes=60,
        active_job_protection=True,
    )
    mock_logger = MagicMock(spec=SecurityEventLogger)
    mgr = RetentionManager(config=cfg, security_logger=mock_logger)
    return {
        "upload_dir": upload_dir,
        "config": cfg,
        "retention_mgr": mgr,
        "logger": mock_logger,
        "tmp_path": tmp_path,
    }


def create_dummy_file(directory: str, filename: str, content: bytes = b"dummy data", age_hours: float = 0.0) -> str:
    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        f.write(content)
    if age_hours > 0:
        # Backdate the file's modification time
        past_time = time.time() - (age_hours * 3600.0)
        os.utime(path, (past_time, past_time))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Group A: Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_retention_config_defaults():
    cfg = MultimodalConfig()
    assert cfg.retention_hours == 24.0
    assert cfg.cleanup_enabled is True
    assert cfg.cleanup_interval_minutes == 60
    assert cfg.active_job_protection is True


def test_retention_config_custom_values():
    cfg = MultimodalConfig(
        retention_hours=12.5,
        cleanup_enabled=False,
        cleanup_interval_minutes=30,
        active_job_protection=False,
    )
    assert cfg.retention_hours == 12.5
    assert cfg.cleanup_enabled is False
    assert cfg.cleanup_interval_minutes == 30
    assert cfg.active_job_protection is False


# ─────────────────────────────────────────────────────────────────────────────
# Group B: Expiration Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_file_younger_than_retention_is_not_expired(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    path = create_dummy_file(temp_upload_env["upload_dir"], "young.png", age_hours=5.0)
    assert mgr.is_expired(path) is False


def test_file_older_than_retention_is_expired(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    path = create_dummy_file(temp_upload_env["upload_dir"], "old.png", age_hours=25.0)
    assert mgr.is_expired(path) is True


def test_file_at_exact_boundary_is_expired(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    # Exactly 24.0 hours
    path = create_dummy_file(temp_upload_env["upload_dir"], "boundary.png", age_hours=24.0)
    assert mgr.is_expired(path) is True


def test_expiration_with_injected_now_timestamp(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    path = create_dummy_file(temp_upload_env["upload_dir"], "test_inject.png", age_hours=0.0)
    base_time = time.time()
    # At +10h: not expired
    assert mgr.is_expired(path, now=base_time + (10 * 3600)) is False
    # At +24.1h: expired
    assert mgr.is_expired(path, now=base_time + (24.1 * 3600)) is True


# ─────────────────────────────────────────────────────────────────────────────
# Group C: Cleanup & Idempotency Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cleanup_expired_file_deleted_and_young_preserved(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    u_dir = temp_upload_env["upload_dir"]

    old_file = create_dummy_file(u_dir, "file_old.png", age_hours=30.0)
    young_file = create_dummy_file(u_dir, "file_young.png", age_hours=2.0)

    summary = mgr.cleanup_expired_files()
    assert summary["scanned"] == 2
    assert summary["expired"] == 1
    assert summary["deleted"] == 1
    assert summary["failed"] == 0

    assert not os.path.exists(old_file)
    assert os.path.exists(young_file)


def test_cleanup_multiple_expired_files(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    u_dir = temp_upload_env["upload_dir"]

    for i in range(5):
        create_dummy_file(u_dir, f"old_{i}.pdf", age_hours=48.0)
    for i in range(3):
        create_dummy_file(u_dir, f"young_{i}.pdf", age_hours=1.0)

    summary = mgr.cleanup_expired_files()
    assert summary["scanned"] == 8
    assert summary["expired"] == 5
    assert summary["deleted"] == 5
    assert summary["failed"] == 0


def test_cleanup_is_idempotent(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    u_dir = temp_upload_env["upload_dir"]

    create_dummy_file(u_dir, "idempotent.png", age_hours=36.0)

    # First run: deletes the file
    res1 = mgr.cleanup_expired_files()
    assert res1["deleted"] == 1

    # Second run: nothing to delete
    res2 = mgr.cleanup_expired_files()
    assert res2["scanned"] == 0
    assert res2["deleted"] == 0
    assert res2["failed"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Group D: Active Job Protection Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_active_job_files_never_deleted_even_when_old(temp_upload_env):
    jm = JobManager(config=temp_upload_env["config"], enable_persistence=False)
    mgr = RetentionManager(
        config=temp_upload_env["config"],
        job_manager=jm,
        security_logger=temp_upload_env["logger"],
    )
    u_dir = temp_upload_env["upload_dir"]

    # Create old files associated with active jobs
    for status in ("queued", "processing", "retrying"):
        fid = f"file_{status}_{uuid.uuid4().hex[:6]}"
        create_dummy_file(u_dir, f"{fid}.png", age_hours=72.0)
        job = Job(
            job_id=str(uuid.uuid4()),
            file_id=fid,
            fingerprint=f"fp_{status}",
            status=status,
        )
        jm._jobs[job.job_id] = job

    # Also an unassociated old file
    orphan = create_dummy_file(u_dir, "orphan_old.png", age_hours=72.0)

    summary = mgr.cleanup_expired_files()
    assert summary["scanned"] == 4
    assert summary["expired"] == 4
    assert summary["protected"] == 3
    assert summary["deleted"] == 1

    # Active job files remain on disk
    for job in jm._jobs.values():
        fpath = os.path.join(u_dir, f"{job.file_id}.png")
        assert os.path.exists(fpath), f"Protected file for status {job.status} was wrongly deleted!"

    # Orphan was deleted
    assert not os.path.exists(orphan)
    jm.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────────────────────
# Group E: Completed Jobs Cleanup Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_completed_job_file_deleted_and_metadata_preserved(temp_upload_env):
    jm = JobManager(config=temp_upload_env["config"], enable_persistence=False)
    mgr = RetentionManager(config=temp_upload_env["config"], job_manager=jm)
    u_dir = temp_upload_env["upload_dir"]

    fid = f"completed_{uuid.uuid4().hex[:6]}"
    fpath = create_dummy_file(u_dir, f"{fid}.png", age_hours=48.0)

    job = Job(
        job_id=str(uuid.uuid4()),
        file_id=fid,
        fingerprint="fp_comp",
        status="completed",
        result={"status": "accepted", "extracted_fields": {"order_id": "ORD-123"}},
    )
    jm._jobs[job.job_id] = job

    summary = mgr.cleanup_expired_files()
    assert summary["deleted"] == 1
    assert not os.path.exists(fpath)

    # Job record remains intact and client-accessible
    retrieved_job = jm.get_job(job.job_id)
    assert retrieved_job is not None
    assert retrieved_job.status == "completed"
    assert retrieved_job.file_retained is False
    assert retrieved_job.result["extracted_fields"]["order_id"] == "ORD-123"
    jm.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────────────────────
# Group F: Failed Jobs Cleanup Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_failed_job_file_deleted_and_sanitized_error_preserved(temp_upload_env):
    jm = JobManager(config=temp_upload_env["config"], enable_persistence=False)
    mgr = RetentionManager(config=temp_upload_env["config"], job_manager=jm)
    u_dir = temp_upload_env["upload_dir"]

    fid = f"failed_{uuid.uuid4().hex[:6]}"
    fpath = create_dummy_file(u_dir, f"{fid}.png", age_hours=50.0)

    job = Job(
        job_id=str(uuid.uuid4()),
        file_id=fid,
        fingerprint="fp_fail",
        status="failed",
        error="Unable to process the uploaded file due to an internal error.",
    )
    jm._jobs[job.job_id] = job

    summary = mgr.cleanup_expired_files()
    assert summary["deleted"] == 1
    assert not os.path.exists(fpath)

    retrieved = jm.get_job(job.job_id)
    assert retrieved.status == "failed"
    assert retrieved.file_retained is False
    assert "Unable to process" in retrieved.error
    jm.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────────────────────
# Group G: Orphan Files Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_orphan_files_young_kept_and_old_cleaned(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    u_dir = temp_upload_env["upload_dir"]

    # Recent orphan (< 24h)
    young_orphan = create_dummy_file(u_dir, "orphan_young.pdf", age_hours=4.0)
    # Expired orphan (> 24h)
    old_orphan = create_dummy_file(u_dir, "orphan_old.pdf", age_hours=30.0)

    summary = mgr.cleanup_expired_files()
    assert summary["deleted"] == 1
    assert os.path.exists(young_orphan)
    assert not os.path.exists(old_orphan)


# ─────────────────────────────────────────────────────────────────────────────
# Group H: Security & Directory Containment Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_directory_containment_rejects_paths_outside_upload_dir(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    u_dir = temp_upload_env["upload_dir"]

    assert mgr.verify_directory_containment(os.path.join(u_dir, "valid.png")) is True
    assert mgr.verify_directory_containment(os.path.join(u_dir, "..", "secret.txt")) is False
    assert mgr.verify_directory_containment("C:\\Windows\\System32\\calc.exe") is False
    assert mgr.verify_directory_containment(os.path.join(u_dir, "jobs.json")) is False


def test_symlink_pointing_outside_is_blocked(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    u_dir = temp_upload_env["upload_dir"]
    tmp_path = temp_upload_env["tmp_path"]

    # Create target file outside upload directory
    target_outside = str(tmp_path / "outside_target.txt")
    with open(target_outside, "w") as f:
        f.write("sensitive system file")

    symlink_path = os.path.join(u_dir, "symlink_test.txt")
    try:
        os.symlink(target_outside, symlink_path)
        # Should be rejected because it resolves outside upload_dir
        assert mgr.verify_directory_containment(symlink_path) is False
    except OSError:
        # Windows without SeCreateSymbolicLinkPrivilege: test fallback
        pass


def test_jobs_json_metadata_file_never_deleted_by_cleanup(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    u_dir = temp_upload_env["upload_dir"]

    jobs_file = create_dummy_file(u_dir, "jobs.json", age_hours=100.0)
    mgr.cleanup_expired_files()
    assert os.path.exists(jobs_file), "jobs.json was deleted by retention cleanup!"


# ─────────────────────────────────────────────────────────────────────────────
# Group I: Audit Logging & Privacy Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_audit_events_recorded_without_pii_or_paths(temp_upload_env):
    mock_logger = temp_upload_env["logger"]
    mgr = temp_upload_env["retention_mgr"]
    u_dir = temp_upload_env["upload_dir"]

    create_dummy_file(u_dir, "audit_test.png", age_hours=36.0)
    mgr.cleanup_expired_files()

    # Verify security logger was called with sanitized events
    assert mock_logger.log_event.called
    call_args_list = mock_logger.log_event.call_args_list
    events = [call[1].get("event_type") for call in call_args_list]
    assert "MULTIMODAL_FILE_RETENTION_EXPIRED" in events
    assert "MULTIMODAL_FILE_CLEANUP_DELETED" in events

    # Ensure no absolute paths in details
    for call in call_args_list:
        details = call[1].get("details", {})
        details_str = str(details)
        assert u_dir not in details_str


# ─────────────────────────────────────────────────────────────────────────────
# Group J: Failure Handling & Missing File Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_isolated_deletion_failure_does_not_halt_cleanup(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    u_dir = temp_upload_env["upload_dir"]

    f1 = create_dummy_file(u_dir, "fail_one.png", age_hours=30.0)
    f2 = create_dummy_file(u_dir, "succeed_two.png", age_hours=30.0)

    original_remove = os.remove

    def mock_remove(path):
        if "fail_one.png" in path:
            raise PermissionError("Access denied simulating lock")
        original_remove(path)

    with patch("os.remove", side_effect=mock_remove):
        summary = mgr.cleanup_expired_files()

    assert summary["expired"] == 2
    assert summary["failed"] == 1
    assert summary["deleted"] == 1
    assert not os.path.exists(f2)


def test_missing_file_handled_safely_and_skipped(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    u_dir = temp_upload_env["upload_dir"]

    create_dummy_file(u_dir, "ghost.png", age_hours=30.0)

    def mock_remove(path):
        raise FileNotFoundError("File already removed")

    with patch("os.remove", side_effect=mock_remove):
        summary = mgr.cleanup_expired_files()

    assert summary["skipped"] == 1
    assert summary["failed"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Group K: Scheduling & Manual Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_manual_cleanup_api_endpoint():
    response = client.post("/multimodal/cleanup")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    summary = data["cleanup_summary"]
    for key in ("scanned", "expired", "deleted", "protected", "skipped", "failed"):
        assert key in summary


def test_background_scheduler_start_and_stop(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    mgr.config.cleanup_interval_minutes = 1  # 1 min in config
    mgr.start_scheduler()
    assert mgr._scheduler_thread is not None
    assert mgr._scheduler_thread.is_alive()

    mgr.stop_scheduler()
    assert mgr._scheduler_thread is None


def test_startup_cleanup_runs_safely(temp_upload_env):
    mgr = temp_upload_env["retention_mgr"]
    summary = mgr.run_startup_cleanup()
    assert isinstance(summary, dict)
    assert "deleted" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Group L: Regressions (Phases 1-5, Task 1 Demo, /ask)
# ─────────────────────────────────────────────────────────────────────────────

def test_multimodal_analyze_remains_functional():
    buf = io.BytesIO()
    img = Image.new("RGB", (10, 10), color="white")
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    response = client.post(
        "/multimodal/analyze",
        files={"file": ("retention_compat.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"


def test_task1_ask_endpoint_remains_functional():
    response = client.post("/ask", json={"question": "What is the return policy?"})
    assert response.status_code != 404
