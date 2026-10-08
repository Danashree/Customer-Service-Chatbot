import os
from dataclasses import dataclass
from typing import List

# Base paths
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(PIPELINE_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

DEFAULT_KB_DIR = os.path.join(PROJECT_ROOT, "knowledge_base")
DEFAULT_QUARANTINE_DIR = os.path.join(DEFAULT_KB_DIR, "quarantine")
DEFAULT_REGISTRY_FILE = os.path.join(PIPELINE_DIR, "registry.json")
DEFAULT_QUARANTINE_REPORT = os.path.join(DEFAULT_QUARANTINE_DIR, "quarantine_report.json")
DEFAULT_VERSIONS_DIR = os.path.join(BACKEND_DIR, "versions")
DEFAULT_VERSIONS_METADATA = os.path.join(DEFAULT_VERSIONS_DIR, "versions.json")
DEFAULT_ACTIVE_VERSION_FILE = os.path.join(DEFAULT_VERSIONS_DIR, "active_version.json")
DEFAULT_BASE_FAISS_DIR = os.path.join(BACKEND_DIR, "faiss_index")
DEFAULT_HEALTH_CHECK_DELAY_SECONDS = 300  # 5 minutes
DEFAULT_EVALUATION_REPORTS_DIR = os.path.join(DEFAULT_VERSIONS_DIR, "evaluation_reports")

# Phase 3 Quality Gate Thresholds
DEFAULT_MIN_RETRIEVAL_SCORE = 0.70
DEFAULT_MIN_GROUNDING_SCORE = 0.70
DEFAULT_MAX_ALLOWED_REGRESSION = 0.05

# Phase 4 Scheduler, Retry, and Maintenance Window Configuration
DEFAULT_SCHEDULER_ENABLED = True
DEFAULT_SCHEDULE_INTERVAL_MINUTES = 60
DEFAULT_MAINTENANCE_WINDOW_START = "02:00"
DEFAULT_MAINTENANCE_WINDOW_END = "04:00"
DEFAULT_RETRY_DELAYS_MINUTES = (15, 30, 60)
DEFAULT_SCHEDULER_STATE_FILE = os.path.join(PIPELINE_DIR, "scheduler_state.json")

# Phase 5 Security & Data Protection Configuration
DEFAULT_SECURITY_ENABLED = True
DEFAULT_PII_MASKING_ENABLED = True
DEFAULT_PROMPT_INJECTION_DETECTION_ENABLED = True
DEFAULT_MAX_DOCUMENT_SIZE_MB = 20
DEFAULT_ALLOWED_DOCUMENT_EXTENSIONS = (".pdf",)
DEFAULT_SECURITY_LOG_FILE = os.path.join(PIPELINE_DIR, "security_events.json")
DEFAULT_USER_ROLE = "viewer"

# Phase 6 Monitoring & Health Configuration
DEFAULT_MONITORING_ENABLED = True
DEFAULT_MAX_PIPELINE_LATENCY_MS = 10000
DEFAULT_MAX_RECENT_FAILURES = 3
DEFAULT_HEALTH_CHECK_FAILURE_THRESHOLD = 1
DEFAULT_METRICS_HISTORY_LIMIT = 100
DEFAULT_MONITORING_STATE_FILE = os.path.join(PIPELINE_DIR, "monitoring_state.json")

@dataclass
class PipelineConfig:
    kb_dir: str = DEFAULT_KB_DIR
    quarantine_dir: str = DEFAULT_QUARANTINE_DIR
    registry_file: str = DEFAULT_REGISTRY_FILE
    quarantine_report_file: str = DEFAULT_QUARANTINE_REPORT
    supported_extensions: tuple = (".pdf",)
    chunk_size: int = 800
    chunk_overlap: int = 150
    versions_dir: str = DEFAULT_VERSIONS_DIR
    versions_metadata_file: str = DEFAULT_VERSIONS_METADATA
    active_version_file: str = DEFAULT_ACTIVE_VERSION_FILE
    base_faiss_dir: str = DEFAULT_BASE_FAISS_DIR
    health_check_delay_seconds: int = DEFAULT_HEALTH_CHECK_DELAY_SECONDS
    evaluation_reports_dir: str = DEFAULT_EVALUATION_REPORTS_DIR
    min_retrieval_score: float = DEFAULT_MIN_RETRIEVAL_SCORE
    min_grounding_score: float = DEFAULT_MIN_GROUNDING_SCORE
    max_allowed_regression: float = DEFAULT_MAX_ALLOWED_REGRESSION
    scheduler_enabled: bool = DEFAULT_SCHEDULER_ENABLED
    schedule_interval_minutes: int = DEFAULT_SCHEDULE_INTERVAL_MINUTES
    maintenance_window_start: str = DEFAULT_MAINTENANCE_WINDOW_START
    maintenance_window_end: str = DEFAULT_MAINTENANCE_WINDOW_END
    retry_delays_minutes: tuple = DEFAULT_RETRY_DELAYS_MINUTES
    scheduler_state_file: str = DEFAULT_SCHEDULER_STATE_FILE
    security_enabled: bool = DEFAULT_SECURITY_ENABLED
    pii_masking_enabled: bool = DEFAULT_PII_MASKING_ENABLED
    prompt_injection_detection_enabled: bool = DEFAULT_PROMPT_INJECTION_DETECTION_ENABLED
    max_document_size_mb: int = DEFAULT_MAX_DOCUMENT_SIZE_MB
    allowed_document_extensions: tuple = DEFAULT_ALLOWED_DOCUMENT_EXTENSIONS
    security_log_file: str = DEFAULT_SECURITY_LOG_FILE
    default_role: str = DEFAULT_USER_ROLE
    monitoring_enabled: bool = DEFAULT_MONITORING_ENABLED
    max_pipeline_latency_ms: int = DEFAULT_MAX_PIPELINE_LATENCY_MS
    max_recent_failures: int = DEFAULT_MAX_RECENT_FAILURES
    health_check_failure_threshold: int = DEFAULT_HEALTH_CHECK_FAILURE_THRESHOLD
    metrics_history_limit: int = DEFAULT_METRICS_HISTORY_LIMIT
    monitoring_state_file: str = DEFAULT_MONITORING_STATE_FILE

    def __post_init__(self):
        # Normalize paths
        self.kb_dir = os.path.abspath(self.kb_dir)
        self.quarantine_dir = os.path.abspath(self.quarantine_dir)
        self.registry_file = os.path.abspath(self.registry_file)
        self.quarantine_report_file = os.path.abspath(self.quarantine_report_file)
        self.versions_dir = os.path.abspath(self.versions_dir)
        self.versions_metadata_file = os.path.abspath(self.versions_metadata_file)
        self.active_version_file = os.path.abspath(self.active_version_file)
        self.base_faiss_dir = os.path.abspath(self.base_faiss_dir)
        self.evaluation_reports_dir = os.path.abspath(self.evaluation_reports_dir)
        self.scheduler_state_file = os.path.abspath(self.scheduler_state_file)
        self.security_log_file = os.path.abspath(self.security_log_file)
        self.monitoring_state_file = os.path.abspath(self.monitoring_state_file)
