import os
from dataclasses import dataclass
from typing import Tuple, Optional

MULTIMODAL_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(MULTIMODAL_DIR)
DEFAULT_UPLOAD_DIR = os.path.join(BACKEND_DIR, 'temp_uploads')

DEFAULT_MAX_FILE_SIZE_MB = 10
DEFAULT_ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.pdf')
DEFAULT_ALLOWED_MIME_TYPES = (
    'image/png',
    'image/jpeg',
    'image/webp',
    'application/pdf',
)
DEFAULT_RETENTION_HOURS = 24
DEFAULT_PROCESSING_THRESHOLD_SECONDS = 30.0
DEFAULT_MAX_BACKGROUND_RETRIES = 3
DEFAULT_BACKGROUND_RETRY_DELAYS = (1.0, 2.0, 5.0)
DEFAULT_CLEANUP_ENABLED = True
DEFAULT_CLEANUP_INTERVAL_MINUTES = 60
DEFAULT_ACTIVE_JOB_PROTECTION = True

MIME_EXTENSION_MAP = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.pdf': 'application/pdf',
}


@dataclass
class MultimodalConfig:
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    allowed_extensions: Tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS
    allowed_mime_types: Tuple[str, ...] = DEFAULT_ALLOWED_MIME_TYPES
    upload_dir: str = DEFAULT_UPLOAD_DIR
    retention_hours: float = DEFAULT_RETENTION_HOURS
    security_log_file: Optional[str] = None
    multimodal_processing_threshold_seconds: float = DEFAULT_PROCESSING_THRESHOLD_SECONDS
    max_background_retries: int = DEFAULT_MAX_BACKGROUND_RETRIES
    background_retry_delays: Tuple[float, ...] = DEFAULT_BACKGROUND_RETRY_DELAYS
    cleanup_enabled: bool = DEFAULT_CLEANUP_ENABLED
    cleanup_interval_minutes: int = DEFAULT_CLEANUP_INTERVAL_MINUTES
    active_job_protection: bool = DEFAULT_ACTIVE_JOB_PROTECTION

    def __post_init__(self):
        self.upload_dir = os.path.abspath(self.upload_dir)
        os.makedirs(self.upload_dir, exist_ok=True)
