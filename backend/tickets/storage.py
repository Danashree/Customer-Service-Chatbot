"""
Task 3 Phase 1: JSON-based ticket persistence.

Atomic writes via os.replace() with a .tmp intermediary — same pattern as
VectorStoreManager._atomic_write_json in backend/pipeline/vector_store_manager.py.
Thread-safe via threading.Lock().
"""

import json
import os
import threading
import logging
from typing import Dict, List, Optional

from .models import TicketBase, TicketStatus

logger = logging.getLogger(__name__)


class TicketStorage:
    """
    Persists tickets as JSON keyed by ticket_id.
    Default store path: <tickets package dir>/tickets_store.json
    Pass a custom path (e.g. a tmp_path in tests) to isolate storage.
    """

    def __init__(self, store_path: Optional[str] = None) -> None:
        if store_path is None:
            store_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "tickets_store.json"
            )
        self._store_path = store_path
        self._lock = threading.Lock()
        self._ensure_store()

    # ─── internal helpers ────────────────────────────────────────────────────

    def _ensure_store(self) -> None:
        """Create an empty store file if it does not yet exist."""
        if not os.path.exists(self._store_path):
            self._write({})

    def _read(self) -> Dict[str, dict]:
        try:
            with open(self._store_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: Dict[str, dict]) -> None:
        """Atomic write: write to .tmp then os.replace() into place."""
        tmp_path = self._store_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self._store_path)

    # ─── public API ──────────────────────────────────────────────────────────

    def create(self, ticket: TicketBase) -> TicketBase:
        """Persist a new ticket. Raises ValueError if ticket_id already exists."""
        with self._lock:
            data = self._read()
            if ticket.ticket_id in data:
                raise ValueError(f"Ticket {ticket.ticket_id!r} already exists.")
            data[ticket.ticket_id] = ticket.model_dump()
            self._write(data)
            logger.info("Ticket created: %s", ticket.ticket_id)
        return ticket

    def get(self, ticket_id: str) -> Optional[TicketBase]:
        """Return the ticket or None if not found."""
        with self._lock:
            data = self._read()
            raw = data.get(ticket_id)
        if raw is None:
            return None
        return TicketBase(**raw)

    def update_status(
        self,
        ticket_id: str,
        status: TicketStatus,
        note: Optional[str] = None,
    ) -> Optional[TicketBase]:
        """
        Update ticket status (and optionally append a note to unresolved_reason).
        Returns the updated ticket, or None if not found.
        """
        with self._lock:
            data = self._read()
            if ticket_id not in data:
                return None
            data[ticket_id]["status"] = status.value
            if note:
                existing = data[ticket_id].get("unresolved_reason") or ""
                data[ticket_id]["unresolved_reason"] = (
                    f"{existing} | {note}" if existing else note
                )
            self._write(data)
            logger.info("Ticket %s status → %s", ticket_id, status.value)
            return TicketBase(**data[ticket_id])

    def update(self, ticket: TicketBase) -> TicketBase:
        """Update an existing ticket with new data. Overwrites stored dict."""
        with self._lock:
            data = self._read()
            data[ticket.ticket_id] = ticket.model_dump()
            self._write(data)
            logger.info("Ticket %s updated", ticket.ticket_id)
        return ticket

    def update_fields(self, ticket_id: str, fields: dict) -> Optional[TicketBase]:
        """Update specific fields of an existing ticket."""
        with self._lock:
            data = self._read()
            if ticket_id not in data:
                return None
            data[ticket_id].update(fields)
            self._write(data)
            logger.info("Ticket %s fields updated: %s", ticket_id, list(fields.keys()))
            return TicketBase(**data[ticket_id])

    def list_all(self) -> List[TicketBase]:
        """Return all stored tickets, newest-first (by created_at)."""
        with self._lock:
            data = self._read()
        tickets = [TicketBase(**v) for v in data.values()]
        tickets.sort(key=lambda t: t.created_at, reverse=True)
        return tickets
