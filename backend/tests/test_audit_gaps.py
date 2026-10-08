import os
import json
import pytest
from datetime import datetime, timezone

from pipeline.config import PipelineConfig
from pipeline.monitoring import (
    PipelineMetrics,
    MetricsCollector,
    HealthMonitor,
    calculate_application_confidence
)


@pytest.fixture
def audit_env(tmp_path):
    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    state_file = tmp_path / "monitoring_state.json"
    sec_log_file = tmp_path / "security_events.json"
    active_version_file = versions_dir / "active_version.json"
    versions_metadata_file = versions_dir / "versions.json"

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
                        "retrieval_score": 0.90,
                        "grounding_score": 0.80,
                        "response_confidence": 0.85
                    }
                }
            }
        }, f)

    config = PipelineConfig(
        versions_dir=str(versions_dir),
        versions_metadata_file=str(versions_metadata_file),
        active_version_file=str(active_version_file),
        monitoring_state_file=str(state_file),
        security_log_file=str(sec_log_file)
    )

    collector = MetricsCollector(config=config)
    monitor = HealthMonitor(config=config, metrics_collector=collector)
    return collector, monitor, config, tmp_path


# ------------------------------------------------------------------
# Requirement A: Explicit Confidence Monitoring (Tests 1 - 4)
# ------------------------------------------------------------------
def test_confidence_metric_creation(audit_env):
    """Test 1: Explicit response_confidence metric is created and derived."""
    collector, monitor, _, _ = audit_env

    # 1. Directly specified confidence in PipelineMetrics
    m1 = PipelineMetrics(
        run_id="run_1",
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        duration_ms=100.0,
        status="success",
        response_confidence=0.88
    )
    assert m1.response_confidence == 0.88

    # 2. Derived deterministically from retrieval and grounding evidence
    m2 = PipelineMetrics(
        run_id="run_2",
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        duration_ms=120.0,
        status="success",
        retrieval_score=0.90,
        grounding_score=0.80
    )
    assert m2.response_confidence == 0.85

    # 3. Query-level confidence recorded via MetricsCollector
    conf = collector.record_confidence(0.92, query="What is the refund window?")
    assert conf == 0.92


def test_confidence_range_validation():
    """Test 2: Confidence metric is normalized and bounded between 0.0 and 1.0."""
    # Standard helper
    assert calculate_application_confidence(1.0, 1.0) == 1.0
    assert calculate_application_confidence(0.0, 0.0) == 0.0
    assert calculate_application_confidence(None, None) is None

    # Clamping out-of-range inputs
    m_high = PipelineMetrics(
        run_id="high", start_time="", end_time="", duration_ms=10.0, status="success",
        response_confidence=1.45
    )
    assert m_high.response_confidence == 1.0

    m_low = PipelineMetrics(
        run_id="low", start_time="", end_time="", duration_ms=10.0, status="success",
        response_confidence=-0.35
    )
    assert m_low.response_confidence == 0.0


def test_confidence_persistence(audit_env):
    """Test 3: Confidence metric is persisted to monitoring state JSON on disk."""
    collector, _, config, _ = audit_env

    m = PipelineMetrics(
        run_id="run_conf",
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        duration_ms=95.0,
        status="success",
        retrieval_score=0.82,
        grounding_score=0.78,
        response_confidence=0.80
    )
    collector.record_execution(m)

    with open(config.monitoring_state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("last_response_confidence") == 0.80
    history = data.get("metrics_history", [])
    assert len(history) == 1
    assert history[0].get("response_confidence") == 0.80


def test_monitoring_summary_includes_confidence(audit_env):
    """Test 4: HealthMonitor dashboard summary includes confidence metrics."""
    _, monitor, _, _ = audit_env
    summary = monitor.get_dashboard_summary()

    assert "response_confidence" in summary
    assert "latest_response_confidence" in summary
    assert summary["response_confidence"] == 0.85
    assert summary["latest_response_confidence"] == 0.85
    assert summary["latest_retrieval_score"] == 0.90
    assert summary["latest_grounding_score"] == 0.80


# ------------------------------------------------------------------
# Requirement B: Escalation Monitoring (Tests 5 - 10)
# ------------------------------------------------------------------
def test_escalation_event_recording(audit_env):
    """Test 5: An escalation event is recorded with expected metadata structure."""
    collector, _, _, _ = audit_env

    event = collector.record_escalation(
        reason="Customer requested live agent support",
        severity="high",
        session_id="sess_42",
        metadata={"user_tier": "premium"}
    )

    assert "escalation_id" in event
    assert event["escalation_count"] == 1
    assert event["reason"] == "Customer requested live agent support"
    assert event["severity"] == "high"
    assert event["session_id"] == "sess_42"
    assert "timestamp" in event


def test_escalation_count(audit_env):
    """Test 6: Total escalation count increments monotonically."""
    collector, _, _, _ = audit_env
    assert collector.load_state().get("total_escalations", 0) == 0

    collector.record_escalation("Reason 1")
    assert collector.load_state()["total_escalations"] == 1

    collector.record_escalation("Reason 2")
    assert collector.load_state()["total_escalations"] == 2

    collector.record_escalation("Reason 3")
    assert collector.load_state()["total_escalations"] == 3


def test_escalation_reason_persistence(audit_env):
    """Test 7: Escalation reasons are persisted in state history."""
    collector, _, config, _ = audit_env

    collector.record_escalation("Complex billing refund dispute")
    with open(config.monitoring_state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    escalations = data.get("escalations", [])
    assert len(escalations) == 1
    assert escalations[0]["reason"] == "Complex billing refund dispute"


def test_escalation_persistence_across_reload(audit_env):
    """Test 8: Escalations survive recreation of MetricsCollector instance."""
    collector, _, config, _ = audit_env
    collector.record_escalation("Issue requiring human supervisor", severity="critical")

    # Create fresh collector with same config
    new_collector = MetricsCollector(config=config)
    state = new_collector.load_state()

    assert state["total_escalations"] == 1
    assert len(state["escalations"]) == 1
    assert state["escalations"][0]["reason"] == "Issue requiring human supervisor"
    assert state["escalations"][0]["severity"] == "critical"


def test_masked_escalation_data(audit_env):
    """Test 9: PII and secrets in escalation reasons and metadata are sanitized."""
    collector, _, config, _ = audit_env

    raw_reason = "User alice@example.com (phone 9876543210) reported leaked key sk-live-99887766554433221100"
    collector.record_escalation(
        reason=raw_reason,
        metadata={"email": "admin@secret.org", "token": "Bearer secret_token_xyz"}
    )

    with open(config.monitoring_state_file, "r", encoding="utf-8") as f:
        raw_json = f.read()

    assert "alice@example.com" not in raw_json
    assert "9876543210" not in raw_json
    assert "sk-live-99887766554433221100" not in raw_json
    assert "[EMAIL_REDACTED]" in raw_json
    assert "[PHONE_REDACTED]" in raw_json
    assert "[API_KEY_REDACTED]" in raw_json


def test_monitoring_summary_includes_escalation_count(audit_env):
    """Test 10: HealthMonitor dashboard summary reports total_escalations."""
    collector, monitor, _, _ = audit_env

    # Initial state: 0 escalations
    summary = monitor.get_dashboard_summary()
    assert summary.get("total_escalations") == 0

    # Record 2 escalations
    collector.record_escalation("Escalation 1")
    collector.record_escalation("Escalation 2")

    summary = monitor.get_dashboard_summary()
    assert summary.get("total_escalations") == 2

