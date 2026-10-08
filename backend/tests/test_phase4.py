import os
import json
import pytest
from datetime import datetime, timezone, timedelta
from reportlab.pdfgen import canvas
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

from pipeline.config import PipelineConfig
from pipeline.document_processor import DocumentProcessor
from pipeline.vector_store_manager import VectorStoreManager
from pipeline.quality_evaluator import QualityEvaluator
from pipeline.scheduler import KnowledgeBaseScheduler, PermanentPipelineError


class SimulatedClock:
    """Controllable simulated clock for testing without sleep."""
    def __init__(self, dt: datetime):
        self.current_dt = dt

    def now(self) -> datetime:
        return self.current_dt

    def advance(self, minutes: int = 0, hours: int = 0, seconds: int = 0):
        self.current_dt += timedelta(minutes=minutes, hours=hours, seconds=seconds)

    def set_time(self, hour: int, minute: int):
        self.current_dt = self.current_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)


def create_pdf(filepath: str, text: str = "Sample content.") -> None:
    c = canvas.Canvas(filepath)
    c.drawString(100, 750, text)
    c.save()


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings(size=10)


@pytest.fixture
def phase4_env(tmp_path, fake_embeddings):
    """Isolated environment with base FAISS, DocumentProcessor, and Scheduler."""
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    quarantine_dir = kb_dir / "quarantine"
    versions_dir = tmp_path / "versions"
    base_dir = tmp_path / "base_faiss"
    base_dir.mkdir()
    registry_file = tmp_path / "registry.json"
    state_file = tmp_path / "scheduler_state.json"
    reports_dir = versions_dir / "evaluation_reports"

    # Seed base index with initial knowledge
    base_db = FAISS.from_texts(
        ["Initial refund policy notes.", "Mac VM guide for Power BI."],
        embedding=fake_embeddings
    )
    base_db.save_local(str(base_dir))

    config = PipelineConfig(
        kb_dir=str(kb_dir),
        quarantine_dir=str(quarantine_dir),
        registry_file=str(registry_file),
        versions_dir=str(versions_dir),
        versions_metadata_file=str(versions_dir / "versions.json"),
        active_version_file=str(versions_dir / "active_version.json"),
        evaluation_reports_dir=str(reports_dir),
        base_faiss_dir=str(base_dir),
        scheduler_state_file=str(state_file),
        maintenance_window_start="02:00",
        maintenance_window_end="04:00",
        retry_delays_minutes=(15, 30, 60),
        schedule_interval_minutes=60,
        health_check_delay_seconds=0
    )

    doc_processor = DocumentProcessor(config=config)
    vsm = VectorStoreManager(config=config, embeddings=fake_embeddings)
    vsm.init_base_version()  # Creates v1

    benchmark = [
        {"id": "q1", "question": "What is the refund policy?", "expected_keywords": ["refund", "policy"]}
    ]
    evaluator = QualityEvaluator(
        config=config,
        vector_store_manager=vsm,
        embeddings=fake_embeddings,
        benchmark_cases=benchmark
    )

    # Initial time set to 03:00 (inside maintenance window)
    initial_dt = datetime(2026, 9, 9, 3, 0, tzinfo=timezone.utc)
    clock = SimulatedClock(initial_dt)

    scheduler = KnowledgeBaseScheduler(
        config=config,
        document_processor=doc_processor,
        vector_store_manager=vsm,
        quality_evaluator=evaluator,
        time_provider=clock.now
    )

    return scheduler, clock, config, kb_dir, vsm, fake_embeddings


def test_scheduler_detects_no_changes(phase4_env):
    """Test 1: When no new files exist, scheduler stops without creating a version."""
    scheduler, _, _, _, vsm, _ = phase4_env
    res = scheduler.run_once()

    assert res["status"] == "no_changes"
    assert res["active_version"] == "v1"
    assert vsm.get_active_version() == "v1"
    # No candidate v2 should exist
    assert vsm.get_version_metadata("v2") is None


def test_scheduler_detects_new_document_and_creates_candidate(phase4_env):
    """Test 2 & 3: New document triggers candidate creation and quality evaluation."""
    scheduler, clock, _, kb_dir, vsm, _ = phase4_env
    clock.set_time(3, 0)  # Inside maintenance window

    # Add a good document
    create_pdf(os.path.join(kb_dir, "doc1.pdf"), "Our refund policy is 100% money back.")

    res = scheduler.run_once()

    assert res["status"] == "activated"
    assert res["version"] == "v2"
    assert vsm.get_active_version() == "v2"

    meta = vsm.get_version_metadata("v2")
    assert meta["evaluation"]["decision"] == "APPROVED"


def test_quality_rejected_candidate_is_not_activated(phase4_env):
    """Test 4: Candidate that fails quality evaluation is rejected and NOT activated."""
    scheduler, clock, _, kb_dir, vsm, _ = phase4_env
    clock.set_time(3, 0)

    # Set strict benchmark case requiring knowledge absent from candidate
    scheduler.evaluator.benchmark_cases = [
        {"id": "q_strict", "question": "Quantum mechanics syllabus", "expected_keywords": ["quantum", "schrodinger"]}
    ]

    create_pdf(os.path.join(kb_dir, "unrelated.pdf"), "Spaghetti bolognese recipe cooking steps.")

    res = scheduler.run_once()

    assert res["status"] == "quality_rejected"
    assert res["version"] == "v2"
    # Active version MUST remain v1
    assert res["active_version"] == "v1"
    assert vsm.get_active_version() == "v1"

    meta = vsm.get_version_metadata("v2")
    assert meta["evaluation"]["decision"] == "REJECTED"


def test_candidate_activates_when_inside_maintenance_window(phase4_env):
    """Test 5: Approved candidate activates immediately when current time is inside window."""
    scheduler, clock, _, kb_dir, vsm, _ = phase4_env
    clock.set_time(2, 30)  # Inside window 02:00 -> 04:00

    create_pdf(os.path.join(kb_dir, "refund.pdf"), "Refund policy terms.")
    res = scheduler.run_once()

    assert res["status"] == "activated"
    assert res["version"] == "v2"
    assert vsm.get_active_version() == "v2"


def test_candidate_waits_when_outside_maintenance_window(phase4_env):
    """Test 6: Approved candidate defers activation when current time is outside window."""
    scheduler, clock, _, kb_dir, vsm, _ = phase4_env
    clock.set_time(10, 0)  # 10:00 is outside window 02:00 -> 04:00

    create_pdf(os.path.join(kb_dir, "refund.pdf"), "Refund policy terms.")
    res = scheduler.run_once()

    assert res["status"] == "waiting_for_maintenance_window"
    assert res["version"] == "v2"
    # Active version must still be v1
    assert vsm.get_active_version() == "v1"

    status = scheduler.get_status()
    assert status["pending_candidate"]["version"] == "v2"


def test_waiting_candidate_activates_when_maintenance_window_opens(phase4_env):
    """Test 7: Queued candidate is activated on subsequent run when maintenance window opens."""
    scheduler, clock, _, kb_dir, vsm, _ = phase4_env
    clock.set_time(14, 0)  # 14:00 (outside window)

    create_pdf(os.path.join(kb_dir, "refund.pdf"), "Refund policy terms.")
    res1 = scheduler.run_once()
    assert res1["status"] == "waiting_for_maintenance_window"
    assert vsm.get_active_version() == "v1"

    # Fast forward clock to 02:15 (maintenance window opens)
    clock.advance(hours=12, minutes=15)
    assert scheduler.is_in_maintenance_window() is True

    res2 = scheduler.run_once()
    assert res2["status"] == "activated"
    assert res2["version"] == "v2"
    assert vsm.get_active_version() == "v2"


def test_maintenance_window_crossing_midnight(phase4_env):
    """Test 8: Midnight-crossing window (22:00 -> 02:00) evaluates boundary conditions correctly."""
    scheduler, clock, config, _, _, _ = phase4_env
    config.maintenance_window_start = "22:00"
    config.maintenance_window_end = "02:00"

    # 21:59 -> Outside
    clock.set_time(21, 59)
    assert scheduler.is_in_maintenance_window() is False

    # 22:00 -> Inside
    clock.set_time(22, 0)
    assert scheduler.is_in_maintenance_window() is True

    # 23:30 -> Inside
    clock.set_time(23, 30)
    assert scheduler.is_in_maintenance_window() is True

    # 01:30 -> Inside
    clock.set_time(1, 30)
    assert scheduler.is_in_maintenance_window() is True

    # 02:00 -> Outside
    clock.set_time(2, 0)
    assert scheduler.is_in_maintenance_window() is False

    # 03:00 -> Outside
    clock.set_time(3, 0)
    assert scheduler.is_in_maintenance_window() is False


def test_standard_maintenance_window(phase4_env):
    """Test 8b: Standard day window (02:00 -> 04:00) boundary tests."""
    scheduler, clock, config, _, _, _ = phase4_env
    config.maintenance_window_start = "02:00"
    config.maintenance_window_end = "04:00"

    clock.set_time(1, 59)
    assert scheduler.is_in_maintenance_window() is False

    clock.set_time(2, 0)
    assert scheduler.is_in_maintenance_window() is True

    clock.set_time(3, 59)
    assert scheduler.is_in_maintenance_window() is True

    clock.set_time(4, 0)
    assert scheduler.is_in_maintenance_window() is False


def test_transient_failure_retry_schedule_15_30_60(phase4_env):
    """Test 9, 10, 11: Transient failure triggers retry after 15m, then 30m, then 60m."""
    scheduler, clock, _, _, _, _ = phase4_env

    # 1. First transient failure at 03:00 -> retry after 15 minutes
    err = OSError("Simulated temporary filesystem lock")
    res1 = scheduler.run_once(simulated_failure=err)
    assert res1["status"] == "retry_scheduled"
    assert res1["attempt"] == 1
    assert res1["delay_minutes"] == 15
    retry_time_1 = datetime.fromisoformat(res1["next_retry_at"])

    # If called before 15m expires, scheduler waits
    clock.advance(minutes=5)
    waiting_res = scheduler.run_once()
    assert waiting_res["status"] == "waiting_for_retry"

    # Advance to 15m mark
    clock.advance(minutes=10)

    # 2. Second transient failure -> retry after 30 minutes
    res2 = scheduler.run_once(simulated_failure=err)
    assert res2["status"] == "retry_scheduled"
    assert res2["attempt"] == 2
    assert res2["delay_minutes"] == 30

    # Advance 30 minutes
    clock.advance(minutes=30)

    # 3. Third transient failure -> retry after 60 minutes
    res3 = scheduler.run_once(simulated_failure=err)
    assert res3["status"] == "retry_scheduled"
    assert res3["attempt"] == 3
    assert res3["delay_minutes"] == 60


def test_retry_exhaustion_marks_operation_as_failed(phase4_env):
    """Test 12 & 13: Exhausting all 3 retries marks operation as failed and preserves active v1."""
    scheduler, clock, _, _, vsm, _ = phase4_env
    err = OSError("Continuous transient failure")

    # Attempt 1 (initial failure -> schedule retry 1 in 15m)
    r1 = scheduler.run_once(simulated_failure=err)
    assert r1["status"] == "retry_scheduled"
    clock.advance(minutes=15)

    # Attempt 2 (retry 1 failure -> schedule retry 2 in 30m)
    r2 = scheduler.run_once(simulated_failure=err)
    assert r2["status"] == "retry_scheduled"
    clock.advance(minutes=30)

    # Attempt 3 (retry 2 failure -> schedule retry 3 in 60m)
    r3 = scheduler.run_once(simulated_failure=err)
    assert r3["status"] == "retry_scheduled"
    clock.advance(minutes=60)

    # Attempt 4 (retry 3 failure -> all 3 retries exhausted -> final failure)
    r4 = scheduler.run_once(simulated_failure=err)
    assert r4["status"] == "failed"
    assert r4["attempts"] == 3
    assert r4["active_version"] == "v1"

    # Active version MUST remain v1
    assert vsm.get_active_version() == "v1"

    # Active retry state is cleared
    status = scheduler.get_status()
    assert status["active_retry"] is None


def test_quality_rejection_does_not_trigger_retry(phase4_env):
    """Test 14: Quality gate rejection is a final decision and does NOT trigger retry."""
    scheduler, clock, _, kb_dir, _, _ = phase4_env
    clock.set_time(3, 0)

    # Set strict benchmark case requiring knowledge absent from candidate
    scheduler.evaluator.benchmark_cases = [
        {"id": "q_strict", "question": "Quantum mechanics syllabus", "expected_keywords": ["quantum", "schrodinger"]}
    ]

    create_pdf(os.path.join(kb_dir, "bad.pdf"), "Unrelated text.")
    res = scheduler.run_once()

    assert res["status"] == "quality_rejected"
    state = scheduler.load_state()
    # No retry must be scheduled
    assert state.get("active_retry") is None


def test_permanent_failures_do_not_trigger_retry(phase4_env):
    """Test 15: Permanent pipeline errors do not trigger retry backoff."""
    scheduler, _, _, _, _, _ = phase4_env

    res = scheduler.run_once(simulated_failure=PermanentPipelineError("Fatal bad config"))

    assert res["status"] == "failed"
    assert res.get("is_permanent") is True
    state = scheduler.load_state()
    assert state.get("active_retry") is None


def test_outside_maintenance_window_does_not_consume_retries(phase4_env):
    """Test 17: Being outside maintenance window is NOT a failure and consumes no retry attempts."""
    scheduler, clock, _, kb_dir, _, _ = phase4_env
    clock.set_time(12, 0)  # Outside window

    create_pdf(os.path.join(kb_dir, "good.pdf"), "Refund policy terms.")
    res = scheduler.run_once()

    assert res["status"] == "waiting_for_maintenance_window"
    state = scheduler.load_state()
    assert state.get("active_retry") is None


def test_scheduler_state_persists_across_recreation(phase4_env):
    """Test 16: Scheduler state survives object instantiation/restart."""
    scheduler, clock, config, kb_dir, vsm, fake_embeddings = phase4_env
    clock.set_time(12, 0)

    create_pdf(os.path.join(kb_dir, "good.pdf"), "Refund policy terms.")
    scheduler.run_once()

    # Recreate scheduler instance pointing to same state file
    new_scheduler = KnowledgeBaseScheduler(
        config=config,
        document_processor=DocumentProcessor(config=config),
        vector_store_manager=vsm,
        quality_evaluator=QualityEvaluator(config=config, vector_store_manager=vsm, embeddings=fake_embeddings),
        time_provider=clock.now
    )

    status = new_scheduler.get_status()
    assert status["pending_candidate"]["version"] == "v2"
    assert status["last_result"]["status"] == "waiting_for_maintenance_window"


def test_scheduler_status_reports_diagnostics(phase4_env):
    """Test 17: get_status() returns all required operational diagnostic fields."""
    scheduler, clock, _, _, _, _ = phase4_env
    status = scheduler.get_status()

    assert "enabled" in status
    assert "current_time" in status
    assert "in_maintenance_window" in status
    assert "maintenance_window" in status
    assert "schedule_interval_minutes" in status
    assert "active_version" in status
    assert status["active_version"] == "v1"


def test_schedule_interval_is_configurable(phase4_env):
    """Test 18: Schedule interval is configurable via PipelineConfig."""
    scheduler, _, config, _, _, _ = phase4_env
    assert config.schedule_interval_minutes == 60

    config.schedule_interval_minutes = 120
    status = scheduler.get_status()
    assert status["schedule_interval_minutes"] == 120


def test_scheduler_disabled_skips_execution(phase4_env):
    """Test 19: When scheduler is disabled, run_once skips execution and returns disabled."""
    scheduler, _, config, kb_dir, _, _ = phase4_env
    config.scheduler_enabled = False

    create_pdf(os.path.join(kb_dir, "doc.pdf"), "Refund policy.")
    res = scheduler.run_once()

    assert res["status"] == "disabled"


def test_force_run_bypasses_disabled_scheduler(phase4_env):
    """Test 20: force_run=True allows manual execution even when scheduler is disabled."""
    scheduler, clock, config, kb_dir, vsm, _ = phase4_env
    config.scheduler_enabled = False
    clock.set_time(3, 0)  # In window

    create_pdf(os.path.join(kb_dir, "doc.pdf"), "Refund policy notes.")
    res = scheduler.run_once(force_run=True)

    assert res["status"] == "activated"
    assert vsm.get_active_version() == "v2"


def test_modified_document_triggers_candidate_update(phase4_env):
    """Test 21: Modifying an existing document triggers candidate creation on subsequent run."""
    scheduler, clock, _, kb_dir, vsm, _ = phase4_env
    clock.set_time(3, 0)

    pdf_file = os.path.join(kb_dir, "policy.pdf")
    create_pdf(pdf_file, "Initial refund policy version 1.")
    res1 = scheduler.run_once()
    assert res1["status"] == "activated"
    assert res1["version"] == "v2"

    # Modify the document
    create_pdf(pdf_file, "Updated refund policy version 2 with expanded coverage.")
    res2 = scheduler.run_once()
    assert res2["status"] == "activated"
    assert res2["version"] == "v3"
    assert vsm.get_active_version() == "v3"


def test_audit_history_records_events(phase4_env):
    """Test 22: Scheduler records an auditable trail of executions in state."""
    scheduler, clock, _, kb_dir, _, _ = phase4_env
    clock.set_time(3, 0)

    create_pdf(os.path.join(kb_dir, "policy.pdf"), "Refund policy notes.")
    scheduler.run_once()

    state = scheduler.load_state()
    history = state.get("audit_history", [])
    assert len(history) > 0
    assert "timestamp" in history[-1]
    assert history[-1]["result"]["status"] == "activated"

