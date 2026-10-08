import os
import json
import time
import pytest
from pathlib import Path
from datetime import datetime, timezone

from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

from pipeline.config import PipelineConfig
from pipeline.security import SecurityEventLogger
from pipeline.monitoring import (
    PipelineMetrics,
    MetricsCollector,
    HealthMonitor,
    HealthStatus,
    PipelineExecutionStatus
)


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings(size=10)


@pytest.fixture
def phase6_env(tmp_path, fake_embeddings):
    """Isolated environment for testing Phase 6 monitoring and health features."""
    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    state_file = tmp_path / "monitoring_state.json"
    sec_log_file = tmp_path / "security_events.json"
    sch_state_file = tmp_path / "scheduler_state.json"
    active_version_file = versions_dir / "active_version.json"
    versions_metadata_file = versions_dir / "versions.json"

    # Setup a healthy v1 version
    v1_dir = versions_dir / "v1"
    v1_dir.mkdir()
    base_db = FAISS.from_texts(["Healthy active content."], embedding=fake_embeddings)
    base_db.save_local(str(v1_dir))

    with open(active_version_file, "w") as f:
        json.dump({"active_version": "v1"}, f)

    with open(versions_metadata_file, "w") as f:
        json.dump({
            "versions": {
                "v1": {
                    "version": "v1",
                    "status": "active",
                    "evaluation": {
                        "status": "approved",
                        "retrieval_score": 0.95,
                        "grounding_score": 0.92
                    }
                }
            }
        }, f)

    config = PipelineConfig(
        versions_dir=str(versions_dir),
        active_version_file=str(active_version_file),
        versions_metadata_file=str(versions_metadata_file),
        monitoring_state_file=str(state_file),
        security_log_file=str(sec_log_file),
        scheduler_state_file=str(sch_state_file),
        max_pipeline_latency_ms=1000,
        max_recent_failures=2,
        metrics_history_limit=5,
        max_allowed_regression=0.05
    )

    collector = MetricsCollector(config=config)
    monitor = HealthMonitor(config=config, metrics_collector=collector)

    return collector, monitor, config, tmp_path


# ------------------------------------------------------------------
# Pipeline Execution & Metrics Tests (Tests 1 - 6)
# ------------------------------------------------------------------
def test_successful_pipeline_run_is_recorded(phase6_env):
    """Test 1: Successful pipeline run is logged and updates success metrics."""
    collector, _, _, _ = phase6_env
    m = PipelineMetrics(
        run_id="run_1",
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        duration_ms=150.0,
        status="success",
        new_documents=2,
        candidate_version="v2",
        activation_status="activated"
    )
    state = collector.record_execution(m)

    assert state["total_runs"] == 1
    assert state["successful_runs"] == 1
    assert state["failed_runs"] == 0
    assert len(state["metrics_history"]) == 1


def test_failed_pipeline_run_is_recorded(phase6_env):
    """Test 2: Failed pipeline execution is tracked and increments failed_runs."""
    collector, _, _, _ = phase6_env
    m = PipelineMetrics(
        run_id="run_fail",
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        duration_ms=80.0,
        status="failed",
        error_message="Filesystem lock error"
    )
    state = collector.record_execution(m)

    assert state["total_runs"] == 1
    assert state["successful_runs"] == 0
    assert state["failed_runs"] == 1
    assert state["metrics_history"][-1]["status"] == "failed"


def test_retry_is_recorded(phase6_env):
    """Test 3: Retry attempt is tracked in metrics and increments retry_count."""
    collector, _, _, _ = phase6_env
    m = PipelineMetrics(
        run_id="run_retry",
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        duration_ms=50.0,
        status="retrying",
        retry_attempt=1
    )
    state = collector.record_execution(m)

    assert state["retry_count"] == 1
    assert state["metrics_history"][-1]["retry_attempt"] == 1


def test_retry_exhaustion_is_recorded(phase6_env):
    """Test 4: Final failure after exhausted retries is recorded."""
    collector, _, _, _ = phase6_env
    m = PipelineMetrics(
        run_id="run_exhausted",
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        duration_ms=90.0,
        status="failed",
        retry_attempt=3,
        error_message="Retry attempts exhausted"
    )
    state = collector.record_execution(m)

    assert state["failed_runs"] == 1
    assert state["retry_count"] == 1
    assert state["metrics_history"][-1]["error_message"] == "Retry attempts exhausted"


def test_maintenance_waiting_is_not_classified_as_failure(phase6_env):
    """Test 5: Waiting for a maintenance window is a normal deferred state, not a failure."""
    collector, _, _, _ = phase6_env
    m = PipelineMetrics(
        run_id="run_wait",
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        duration_ms=60.0,
        status="waiting_for_maintenance_window",
        candidate_version="v2"
    )
    state = collector.record_execution(m)

    assert state["total_runs"] == 1
    assert state["failed_runs"] == 0


def test_quality_rejection_is_recorded(phase6_env):
    """Test 6: Quality gate rejection is tracked separately from transient system failures."""
    collector, _, _, _ = phase6_env
    m = PipelineMetrics(
        run_id="run_rej",
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        duration_ms=75.0,
        status="quality_rejected",
        candidate_version="v2",
        evaluation_status="rejected"
    )
    state = collector.record_execution(m)

    assert state["quality_rejections"] == 1
    assert state["failed_runs"] == 0


# ------------------------------------------------------------------
# Latency & Scores Tests (Tests 7 - 11)
# ------------------------------------------------------------------
def test_latency_is_measured(phase6_env):
    """Test 7: Monotonic timing calculates accurate elapsed milliseconds."""
    collector, _, _, _ = phase6_env
    collector.start_run()
    time.sleep(0.01)  # 10ms
    duration = collector.end_run()

    assert duration > 5.0  # Measured in ms


def test_latency_statistics_are_calculated(phase6_env):
    """Test 8: Average, min, max, and last latency are computed across history."""
    collector, _, _, _ = phase6_env
    for d in [100.0, 200.0, 300.0]:
        collector.record_execution(PipelineMetrics(
            run_id=f"run_{d}",
            start_time="",
            end_time="",
            duration_ms=d,
            status="success"
        ))

    state = collector.load_state()
    assert state["last_latency_ms"] == 300.0
    assert state["min_latency_ms"] == 100.0
    assert state["max_latency_ms"] == 300.0
    assert state["avg_latency_ms"] == 200.0


def test_retrieval_score_is_recorded(phase6_env):
    """Test 9: Candidate retrieval quality score is recorded in metrics."""
    collector, _, _, _ = phase6_env
    m = PipelineMetrics(
        run_id="run_ret",
        start_time="",
        end_time="",
        duration_ms=50.0,
        status="success",
        retrieval_score=0.94
    )
    collector.record_execution(m)
    state = collector.load_state()
    assert state["metrics_history"][-1]["retrieval_score"] == 0.94


def test_grounding_score_is_recorded(phase6_env):
    """Test 10: Candidate grounding score is recorded in metrics."""
    collector, _, _, _ = phase6_env
    m = PipelineMetrics(
        run_id="run_gnd",
        start_time="",
        end_time="",
        duration_ms=50.0,
        status="success",
        grounding_score=0.91
    )
    collector.record_execution(m)
    state = collector.load_state()
    assert state["metrics_history"][-1]["grounding_score"] == 0.91


def test_regression_is_recorded(phase6_env):
    """Test 11: Regression delta is recorded in metrics."""
    collector, _, _, _ = phase6_env
    m = PipelineMetrics(
        run_id="run_reg",
        start_time="",
        end_time="",
        duration_ms=50.0,
        status="success",
        regression_amount=0.03
    )
    collector.record_execution(m)
    state = collector.load_state()
    assert state["metrics_history"][-1]["regression_amount"] == 0.03


# ------------------------------------------------------------------
# Health Status Tests (Tests 12 - 15)
# ------------------------------------------------------------------
def test_active_version_health_is_healthy(phase6_env):
    """Test 12: When active version is intact and FAISS exists, health is HEALTHY."""
    _, monitor, _, _ = phase6_env
    status, reasons = monitor.evaluate_overall_health()

    assert status == HealthStatus.HEALTHY
    assert "healthy" in reasons[0].lower()


def test_invalid_active_version_becomes_unhealthy(phase6_env):
    """Test 13: When active version FAISS index is missing, health becomes UNHEALTHY."""
    _, monitor, config, tmp_path = phase6_env

    # Point active version to nonexistent v99
    with open(config.active_version_file, "w") as f:
        json.dump({"active_version": "v99"}, f)

    status, reasons = monitor.evaluate_overall_health()
    assert status == HealthStatus.UNHEALTHY
    assert any("missing" in r.lower() for r in reasons)


def test_health_check_failure_is_recorded(phase6_env):
    """Test 14: Failed post-activation health check is recorded and triggers an alert."""
    collector, _, _, _ = phase6_env
    collector.record_health_check(version="v2", passed=False, previous_version="v1", rollback_triggered=False)

    state = collector.load_state()
    assert state["last_health_check"]["passed"] is False
    assert any(a["alert_type"] == "HEALTH_CHECK_FAILURE" for a in state["alerts"])


def test_automatic_rollback_event_is_recorded(phase6_env):
    """Test 15: Post-activation health failure with rollback is recorded as an AUTOMATIC_ROLLBACK event."""
    collector, _, _, _ = phase6_env
    collector.record_health_check(version="v2", passed=False, previous_version="v1", rollback_triggered=True)

    state = collector.load_state()
    assert state["rollback_count"] == 1
    assert any(a["alert_type"] == "AUTOMATIC_ROLLBACK" for a in state["alerts"])


# ------------------------------------------------------------------
# System Monitoring & Integration Tests (Tests 16 - 21)
# ------------------------------------------------------------------
def test_scheduler_state_is_monitored(phase6_env):
    """Test 16: HealthMonitor reads Phase 4 scheduler state."""
    _, monitor, config, _ = phase6_env

    with open(config.scheduler_state_file, "w") as f:
        json.dump({
            "enabled": True,
            "last_run_at": "2026-09-10T09:00:00Z",
            "pending_candidate": {"version": "v2"},
            "active_retry": None
        }, f)

    dashboard = monitor.get_dashboard_summary()
    assert dashboard["scheduler_status"]["enabled"] is True
    assert dashboard["scheduler_status"]["pending_candidate"]["version"] == "v2"


def test_security_event_counts_are_monitored(phase6_env):
    """Test 17: Dashboard summary reflects security events recorded by Phase 5."""
    _, monitor, config, _ = phase6_env
    sec_logger = SecurityEventLogger(log_file=config.security_log_file)
    sec_logger.log_event("UNAUTHORIZED_ACCESS", "WARNING", "activate", "DENIED", actor_role="viewer")
    sec_logger.log_event("PII_MASKED", "INFO", "mask", "MASKED")

    dashboard = monitor.get_dashboard_summary()
    assert dashboard["security_event_count"] == 2


def test_dashboard_summary_contains_expected_fields(phase6_env):
    """Test 18: Dashboard summary provides all required fields for monitoring interfaces."""
    _, monitor, _, _ = phase6_env
    summary = monitor.get_dashboard_summary()

    required_fields = [
        "overall_health",
        "active_version",
        "total_runs",
        "successful_runs",
        "failed_runs",
        "retry_count",
        "quality_rejections",
        "rollback_count",
        "average_latency_ms",
        "latest_retrieval_score",
        "latest_grounding_score",
        "security_event_count",
        "scheduler_status",
        "active_alerts"
    ]
    for field in required_fields:
        assert field in summary


def test_monitoring_state_persists_across_recreation(phase6_env):
    """Test 19: State file survives object re-instantiation."""
    collector, _, config, _ = phase6_env
    collector.record_execution(PipelineMetrics("run_persist", "", "", 100.0, "success"))

    # Recreate collector
    new_collector = MetricsCollector(config=config)
    state = new_collector.load_state()
    assert state["total_runs"] == 1
    assert state["metrics_history"][0]["run_id"] == "run_persist"


def test_history_limit_is_respected(phase6_env):
    """Test 20: Metrics history is bounded and never exceeds metrics_history_limit."""
    collector, _, config, _ = phase6_env
    assert config.metrics_history_limit == 5

    for i in range(10):
        collector.record_execution(PipelineMetrics(f"run_{i}", "", "", 10.0, "success"))

    state = collector.load_state()
    assert len(state["metrics_history"]) == 5
    # Oldest runs are pruned
    assert state["metrics_history"][0]["run_id"] == "run_5"
    assert state["metrics_history"][-1]["run_id"] == "run_9"


def test_monitoring_configuration_is_configurable():
    """Test 21: Monitoring thresholds are customizable in PipelineConfig."""
    cfg = PipelineConfig(
        monitoring_enabled=False,
        max_pipeline_latency_ms=2500,
        max_recent_failures=5,
        metrics_history_limit=200
    )
    assert cfg.monitoring_enabled is False
    assert cfg.max_pipeline_latency_ms == 2500
    assert cfg.max_recent_failures == 5
    assert cfg.metrics_history_limit == 200


# ------------------------------------------------------------------
# Alert Conditions & Data Safety Tests (Tests 22 - 25)
# ------------------------------------------------------------------
def test_high_latency_alert_generated(phase6_env):
    """Test 22: HIGH_LATENCY alert is logged when execution duration exceeds threshold."""
    collector, _, config, _ = phase6_env
    assert config.max_pipeline_latency_ms == 1000

    collector.record_execution(PipelineMetrics("run_slow", "", "", 1500.0, "success"))
    state = collector.load_state()
    assert any(a["alert_type"] == "HIGH_LATENCY" for a in state["alerts"])


def test_repeated_failure_alert_generated(phase6_env):
    """Test 23: REPEATED_FAILURES alert is logged after max_recent_failures consecutive failures."""
    collector, _, config, _ = phase6_env
    assert config.max_recent_failures == 2

    collector.record_execution(PipelineMetrics("run_f1", "", "", 50.0, "failed"))
    collector.record_execution(PipelineMetrics("run_f2", "", "", 50.0, "failed"))

    state = collector.load_state()
    assert any(a["alert_type"] == "REPEATED_FAILURES" for a in state["alerts"])


def test_quality_regression_alert_generated(phase6_env):
    """Test 24: QUALITY_REGRESSION alert is generated when candidate regression exceeds limit."""
    collector, _, config, _ = phase6_env
    assert config.max_allowed_regression == 0.05

    collector.record_execution(PipelineMetrics(
        "run_reg_high", "", "", 50.0, "quality_rejected",
        candidate_version="v2",
        regression_amount=0.08
    ))
    state = collector.load_state()
    assert any(a["alert_type"] == "QUALITY_REGRESSION" for a in state["alerts"])


def test_sensitive_information_not_leaked_into_monitoring_state(phase6_env):
    """Test 25: PII and secrets are sanitized before persisting to monitoring_state.json."""
    collector, _, config, _ = phase6_env
    collector.record_execution(PipelineMetrics(
        run_id="run_leak_test",
        start_time="",
        end_time="",
        duration_ms=50.0,
        status="failed",
        error_message="User support@customer.com failed with key AIzaSyFakeSecretKey1234567890123456"
    ))

    with open(config.monitoring_state_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert "support@customer.com" not in content
    assert "AIzaSyFakeSecretKey1234567890123456" not in content
    assert "[EMAIL_REDACTED]" in content
    assert "[SECRET_REDACTED]" in content
