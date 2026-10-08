"""
Task 3 Phase 4: Duplicate & Grouping Configuration.
Central, single source of truth for duplicate detection and relationship thresholds.
"""

from pydantic import BaseModel, Field
import threading
from typing import Optional


class DuplicateConfig(BaseModel):
    duplicate_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    possible_duplicate_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    related_threshold: float = Field(default=0.45, ge=0.0, le=1.0)


_config_lock = threading.Lock()
_global_config = DuplicateConfig()


def get_duplicate_config() -> DuplicateConfig:
    with _config_lock:
        return _global_config.model_copy()


def set_duplicate_config(
    duplicate_threshold: Optional[float] = None,
    possible_duplicate_threshold: Optional[float] = None,
    related_threshold: Optional[float] = None,
) -> DuplicateConfig:
    global _global_config
    with _config_lock:
        current = _global_config.model_dump()
        if duplicate_threshold is not None:
            current["duplicate_threshold"] = duplicate_threshold
        if possible_duplicate_threshold is not None:
            current["possible_duplicate_threshold"] = possible_duplicate_threshold
        if related_threshold is not None:
            current["related_threshold"] = related_threshold
        _global_config = DuplicateConfig(**current)
        return _global_config.model_copy()


def reset_duplicate_config() -> DuplicateConfig:
    global _global_config
    with _config_lock:
        _global_config = DuplicateConfig()
        return _global_config.model_copy()
