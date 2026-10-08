"""
Task 3 Phase 2: Business Hours Configuration and Calculation Utilities.

All SLA time calculations use business time (not wall-clock time), correctly:
  - excluding weekends (configurable)
  - excluding holidays (configurable list of "YYYY-MM-DD" strings)
  - excluding non-working hours (configurable start/end)
  - crossing multiple business days
  - handling SLA start outside business hours (snaps forward)

Public API:
  BusinessHoursConfig   — mutable dataclass; can be updated at runtime
  is_business_day()     — check date
  is_business_time()    — check datetime
  next_business_moment() — snap forward to next valid business moment
  calculate_business_minutes() — count business minutes in [start, end]
  add_business_minutes() — add N business minutes to a datetime
  add_business_hours()  — add N business hours to a datetime
  calculate_sla_due_at() — compute SLA deadline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class BusinessHoursConfig:
    """
    Runtime-configurable business hours definition.
    All fields may be updated at any time; the change affects
    future calculations immediately (existing timestamps are preserved).

    working_weekdays: list of ISO weekday numbers (Mon=0 … Sun=6).
    holidays: ISO date strings "YYYY-MM-DD" to exclude.
    work_start / work_end: hour-of-day integers in 24h format (e.g. 9, 18).
    """

    work_start: int = 9          # 09:00 local (naive datetime comparison)
    work_end: int = 18           # 18:00
    working_weekdays: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon–Fri
    holidays: List[str] = field(default_factory=list)   # ["2025-12-25", ...]

    # ── derived helpers ────────────────────────────────────────────────────

    @property
    def _holiday_set(self) -> Set[date]:
        result: Set[date] = set()
        for s in self.holidays:
            try:
                result.add(date.fromisoformat(s))
            except ValueError:
                logger.warning("Invalid holiday date string ignored: %r", s)
        return result

    @property
    def business_minutes_per_day(self) -> int:
        return (self.work_end - self.work_start) * 60

    def clone(self) -> "BusinessHoursConfig":
        return BusinessHoursConfig(
            work_start=self.work_start,
            work_end=self.work_end,
            working_weekdays=list(self.working_weekdays),
            holidays=list(self.holidays),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Core predicates
# ──────────────────────────────────────────────────────────────────────────────

def is_business_day(d: date, cfg: BusinessHoursConfig) -> bool:
    """Return True if `d` is a working weekday and not a holiday."""
    if d.weekday() not in cfg.working_weekdays:
        return False
    if d in cfg._holiday_set:
        return False
    return True


def is_business_time(dt: datetime, cfg: BusinessHoursConfig) -> bool:
    """
    Return True if `dt` falls within working hours on a business day.
    Comparison is on the naive hour (UTC-aware datetimes are converted to UTC first;
    callers should pass timezone-consistent datetimes).
    """
    if not is_business_day(dt.date(), cfg):
        return False
    hour = dt.hour + dt.minute / 60
    return cfg.work_start <= hour < cfg.work_end


# ──────────────────────────────────────────────────────────────────────────────
# Navigation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _next_calendar_day_start(dt: datetime) -> datetime:
    """Advance to midnight of the next calendar day, preserving tzinfo."""
    next_day = dt.date() + timedelta(days=1)
    return datetime(next_day.year, next_day.month, next_day.day,
                    tzinfo=dt.tzinfo)


def _day_work_start(dt: datetime, cfg: BusinessHoursConfig) -> datetime:
    """Return the work_start moment on `dt`'s date."""
    return dt.replace(hour=cfg.work_start, minute=0, second=0, microsecond=0)


def _day_work_end(dt: datetime, cfg: BusinessHoursConfig) -> datetime:
    """Return the work_end moment on `dt`'s date."""
    if cfg.work_end >= 24:
        next_day = dt.date() + timedelta(days=1)
        return datetime(next_day.year, next_day.month, next_day.day,
                        tzinfo=dt.tzinfo)
    return dt.replace(hour=cfg.work_end, minute=0, second=0, microsecond=0)


def next_business_moment(dt: datetime, cfg: BusinessHoursConfig) -> datetime:
    """
    If `dt` is already within business hours on a business day, return it unchanged.
    Otherwise, snap forward to the start of the next valid business period:
      - if before work_start today and today is a business day → today at work_start
      - if after work_end today, or today is not a business day → next business day at work_start
    """
    current = dt
    # Safety limit: up to 365 days
    for _ in range(365):
        if not is_business_day(current.date(), cfg):
            current = _next_calendar_day_start(current)
            continue

        day_start = _day_work_start(current, cfg)
        day_end = _day_work_end(current, cfg)

        if current < day_start:
            return day_start
        if current >= day_end:
            current = _next_calendar_day_start(current)
            continue
        # Within business hours
        return current

    # Fallback (should never happen in practice)
    raise RuntimeError("Could not find next business moment within 365 days")


# ──────────────────────────────────────────────────────────────────────────────
# Business-time arithmetic
# ──────────────────────────────────────────────────────────────────────────────

def calculate_business_minutes(
    start: datetime, end: datetime, cfg: BusinessHoursConfig
) -> int:
    """
    Count the number of business minutes in the half-open interval [start, end).
    Excludes weekends, holidays, and non-working hours.
    Returns 0 if start >= end.

    Algorithm:
      Iterate day by day, computing the intersection of [start, end) with
      [day_work_start, day_work_end) for each business day.
    """
    # Align timezones if one is aware and the other is naive
    if start.tzinfo is not None and end.tzinfo is None:
        end = end.replace(tzinfo=start.tzinfo)
    elif start.tzinfo is None and end.tzinfo is not None:
        start = start.replace(tzinfo=end.tzinfo)

    if start >= end:
        return 0

    total = 0
    current_day_start = datetime(
        start.date().year, start.date().month, start.date().day,
        tzinfo=start.tzinfo
    )

    end_date = end.date()

    while current_day_start.date() <= end_date:
        d = current_day_start.date()
        if is_business_day(d, cfg):
            day_ws = _day_work_start(current_day_start, cfg)
            day_we = _day_work_end(current_day_start, cfg)

            # Effective window = intersection of business window and [start, end)
            eff_start = max(start, day_ws)
            eff_end = min(end, day_we)

            if eff_end > eff_start:
                total += int((eff_end - eff_start).total_seconds() / 60)

        current_day_start = _next_calendar_day_start(current_day_start)


    return total


def add_business_minutes(
    start: datetime, minutes: int, cfg: BusinessHoursConfig
) -> datetime:
    """
    Return the datetime that is exactly `minutes` business minutes after `start`.
    Handles weekends, holidays, and non-working hours correctly.

    If `start` is outside business hours it is first snapped forward to the
    next valid business moment, then counting begins.
    """
    if minutes <= 0:
        return start

    current = next_business_moment(start, cfg)
    remaining = minutes

    for _ in range(365 * 2):   # safety cap: 2 years
        if remaining <= 0:
            break

        if not is_business_day(current.date(), cfg):
            current = _next_calendar_day_start(current)
            current = next_business_moment(current, cfg)
            continue

        day_end = _day_work_end(current, cfg)
        minutes_left_today = int((day_end - current).total_seconds() / 60)

        if remaining <= minutes_left_today:
            current = current + timedelta(minutes=remaining)
            remaining = 0
        else:
            remaining -= minutes_left_today
            current = _next_calendar_day_start(current)
            current = next_business_moment(current, cfg)

    return current


def add_business_hours(
    start: datetime, hours: float, cfg: BusinessHoursConfig
) -> datetime:
    """Convenience wrapper: add `hours` business hours to `start`."""
    return add_business_minutes(start, int(hours * 60), cfg)


def calculate_sla_due_at(
    sla_started_at: datetime,
    sla_hours: float,
    cfg: BusinessHoursConfig,
) -> datetime:
    """
    Compute the SLA due datetime.

    If sla_started_at falls outside business hours, counting starts from
    the next valid business moment (e.g. a ticket raised Friday night
    starts counting from Monday 09:00).
    """
    return add_business_hours(sla_started_at, sla_hours, cfg)


def calculate_sla_warning_at(
    sla_started_at: datetime,
    sla_hours: float,
    cfg: BusinessHoursConfig,
    warning_threshold: float = 0.75,
) -> datetime:
    """Return the datetime at which the SLA warning threshold is hit."""
    warning_hours = sla_hours * warning_threshold
    return add_business_hours(sla_started_at, warning_hours, cfg)
