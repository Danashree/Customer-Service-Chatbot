"""
Task 3 Phase 2: Runtime-configurable SLA Configuration.

SLAConfig is a module-level singleton that is safe to update at runtime.
New tickets always use the current configuration.
Existing ticket SLA timestamps are not retroactively changed;
they carry their originally-computed sla_due_at and sla_warning_at values.

Usage:
    from tickets.sla_config import get_sla_config, update_sla_config
    cfg = get_sla_config()
    update_sla_config(sla_hours_by_priority={"HIGH": 6})
"""

from __future__ import annotations

import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .business_hours import BusinessHoursConfig
from .priority_enums import Priority

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# SLAConfig
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SLAConfig:
    """
    Runtime-configurable SLA rules.

    sla_hours_by_priority: maps Priority value → SLA duration in business hours.
    warning_threshold:     fraction of SLA at which WARNING state is triggered (default 0.75).
    business_hours:        BusinessHoursConfig controlling working-day windows.
    """

    sla_hours_by_priority: Dict[str, float] = field(default_factory=lambda: {
        Priority.LOW.value: 24.0,
        Priority.MEDIUM.value: 12.0,
        Priority.HIGH.value: 8.0,
        Priority.CRITICAL.value: 4.0,
    })
    warning_threshold: float = 0.75   # 75%
    business_hours: BusinessHoursConfig = field(default_factory=BusinessHoursConfig)

    def get_sla_hours(self, priority: str) -> float:
        """Return the configured SLA hours for a priority level. Falls back to MEDIUM."""
        return self.sla_hours_by_priority.get(priority, 12.0)

    def clone(self) -> "SLAConfig":
        return SLAConfig(
            sla_hours_by_priority=dict(self.sla_hours_by_priority),
            warning_threshold=self.warning_threshold,
            business_hours=self.business_hours.clone(),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton — thread-safe
# ──────────────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_singleton: SLAConfig = SLAConfig()


def get_sla_config() -> SLAConfig:
    """Return the current (possibly updated) SLA configuration."""
    with _lock:
        return _singleton


def update_sla_config(
    sla_hours_by_priority: Optional[Dict[str, float]] = None,
    warning_threshold: Optional[float] = None,
    work_start: Optional[int] = None,
    work_end: Optional[int] = None,
    working_weekdays: Optional[List[int]] = None,
    holidays: Optional[List[str]] = None,
) -> SLAConfig:
    """
    Partially update the SLA configuration at runtime.
    Only the provided keyword arguments are changed; everything else is preserved.

    Returns the new configuration snapshot.

    Behavior for existing tickets:
        Existing tickets keep their original sla_started_at, sla_due_at, and
        sla_warning_at timestamps. Calling POST /tickets/{id}/sla/check after
        a config change will re-evaluate the breach/warning state against the
        STORED timestamps (not recalculate them), unless the application
        explicitly recalculates (via POST /tickets/{id}/priority).
    """
    global _singleton
    with _lock:
        new_cfg = _singleton.clone()

        if sla_hours_by_priority is not None:
            new_cfg.sla_hours_by_priority.update(sla_hours_by_priority)
        if warning_threshold is not None:
            new_cfg.warning_threshold = float(warning_threshold)
        if work_start is not None:
            new_cfg.business_hours.work_start = int(work_start)
        if work_end is not None:
            new_cfg.business_hours.work_end = int(work_end)
        if working_weekdays is not None:
            new_cfg.business_hours.working_weekdays = list(working_weekdays)
        if holidays is not None:
            new_cfg.business_hours.holidays = list(holidays)

        _singleton = new_cfg
        logger.info("SLA configuration updated: %s", new_cfg)
        return new_cfg


def reset_sla_config() -> SLAConfig:
    """Reset SLA configuration to factory defaults. Useful in tests."""
    global _singleton
    with _lock:
        _singleton = SLAConfig()
        return _singleton
