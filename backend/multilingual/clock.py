"""
Task 6 Phase 3 — Clock Abstraction
==================================
Provides injectable time sources for production (SystemClock)
and deterministic simulated-time testing (SimulatedClock).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional


class Clock(ABC):
    """Abstract base class for time providers."""

    @abstractmethod
    def now(self) -> float:
        """Returns current epoch timestamp in seconds."""
        pass


class SystemClock(Clock):
    """Production clock returning real system time."""

    def now(self) -> float:
        return time.time()


class SimulatedClock(Clock):
    """Deterministic simulated clock for testing expiry and restoration boundaries."""

    def __init__(self, start_time: Optional[float] = None):
        self._current_time: float = start_time if start_time is not None else 1774000000.0

    def now(self) -> float:
        return self._current_time

    def set_time(self, timestamp: float) -> None:
        """Sets the clock to an absolute timestamp."""
        self._current_time = float(timestamp)

    def advance(
        self,
        seconds: float = 0.0,
        minutes: float = 0.0,
        hours: float = 0.0,
        days: float = 0.0,
    ) -> float:
        """Advances simulated time by specified duration and returns new timestamp."""
        delta = seconds + (minutes * 60.0) + (hours * 3600.0) + (days * 86400.0)
        self._current_time += delta
        return self._current_time
