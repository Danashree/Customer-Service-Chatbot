"""
Pipeline package for Task 1: Knowledge-base update and monitoring pipeline.
Phase 1: Ingestion, Deduplication, and Quarantine.
"""

from .document_processor import DocumentProcessor, IngestionResult
from .vector_store_manager import VectorStoreManager
from .quality_evaluator import QualityEvaluator
from .scheduler import KnowledgeBaseScheduler, PermanentPipelineError
from .config import PipelineConfig
from .security import (
    AccessController,
    UnauthorizedAccessError,
    PIIMasker,
    PromptInjectionDetector,
    FileSafetyValidator,
    SecurityEventLogger,
    SecurityException,
    UnsafeFileError,
)
from .monitoring import (
    PipelineMetrics,
    MetricsCollector,
    HealthMonitor,
    HealthStatus,
    PipelineExecutionStatus,
)

__all__ = [
    "DocumentProcessor",
    "IngestionResult",
    "VectorStoreManager",
    "QualityEvaluator",
    "KnowledgeBaseScheduler",
    "PermanentPipelineError",
    "PipelineConfig",
    "AccessController",
    "UnauthorizedAccessError",
    "PIIMasker",
    "PromptInjectionDetector",
    "FileSafetyValidator",
    "SecurityEventLogger",
    "SecurityException",
    "UnsafeFileError",
    "PipelineMetrics",
    "MetricsCollector",
    "HealthMonitor",
    "HealthStatus",
    "PipelineExecutionStatus",
]
