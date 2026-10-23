"""In-memory/SQLite ticketing + warm human handoff mocks.

This is deliberately an offline mock: no Zendesk/Salesforce/Intercom calls and
no network access. It exists to exercise tenant isolation, tool-call
trajectories, compensation, and pause/resume semantics in tests and evals.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TENANT = "default"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    category TEXT,
    queue TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    priority TEXT NOT NULL DEFAULT 'normal',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_tenant ON tickets(tenant_id);
CREATE TABLE IF NOT EXISTS notifications (
    notification_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS handoffs (
    handoff_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    state_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'paused',
    created_at REAL NOT NULL,
    resumed_at REAL
);
"""


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]
    seq: int = 0
    result: dict[str, Any] = field(default_factory=dict)


class TicketingSystem:
    """Tenant-scoped ticket store with create/classify/route/update/notify."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def create_ticket(
        self,
        tenant_id: str,
        customer_id: str,
        subject: str,
        category: str | None = None,
        priority: str = "normal",
        *,
        ticket_id: str | None = None,
    ) -> dict[str, Any]:
        ticket_id = ticket_id or f"T-{uuid.uuid4().hex[:10].upper()}"
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO tickets (ticket_id, tenant_id, customer_id, subject, category, "
                "status, priority, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    ticket_id,
                    tenant_id,
                    customer_id,
                    subject,
                    category,
                    "open",
                    priority,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return self.get_ticket(tenant_id, ticket_id) or {}

    def get_ticket(self, tenant_id: str, ticket_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tickets WHERE tenant_id=? AND ticket_id=?",
                (tenant_id, ticket_id),
            ).fetchone()
        return self._row(row)

    def list_tickets(self, tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tickets WHERE tenant_id=? ORDER BY created_at DESC LIMIT ?",
                (tenant_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def classify(self, tenant_id: str, ticket_id: str, category: str) -> dict[str, Any] | None:
        return self._update(tenant_id, ticket_id, category=category)

    def route(self, tenant_id: str, ticket_id: str, queue: str) -> dict[str, Any] | None:
        return self._update(tenant_id, ticket_id, queue=queue)

    def update_status(self, tenant_id: str, ticket_id: str, status: str) -> dict[str, Any] | None:
        return self._update(tenant_id, ticket_id, status=status)

    def notify(self, tenant_id: str, ticket_id: str, channel: str = "sms") -> dict[str, Any]:
        ticket = self.get_ticket(tenant_id, ticket_id)
        if ticket is None:
            return {"ok": False, "error": "ticket_not_found"}
        notification_id = f"N-{uuid.uuid4().hex[:10].upper()}"
        payload = {
            "ticket_id": ticket_id,
            "status": ticket["status"],
            "queue": ticket["queue"],
        }
        with self._lock:
            self._conn.execute(
                "INSERT INTO notifications (notification_id, tenant_id, ticket_id, channel, "
                "payload, created_at) VALUES (?,?,?,?,?,?)",
                (notification_id, tenant_id, ticket_id, channel, json.dumps(payload), time.time()),
            )
            self._conn.commit()
        return {"ok": True, "notification_id": notification_id, "channel": channel}

    def notifications(self, tenant_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM notifications WHERE tenant_id=? ORDER BY created_at",
                (tenant_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # --- compensation helpers -------------------------------------------
    def undo_create(self, tenant_id: str, ticket_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM tickets WHERE tenant_id=? AND ticket_id=?", (tenant_id, ticket_id)
            )
            self._conn.commit()

    def undo_status(self, tenant_id: str, ticket_id: str, previous: str) -> None:
        self._update(tenant_id, ticket_id, status=previous)

    def undo_route(self, tenant_id: str, ticket_id: str, previous: str | None) -> None:
        self._update(tenant_id, ticket_id, queue=previous)

    def _update(self, tenant_id: str, ticket_id: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {k: v for k, v in fields.items() if k in {"category", "queue", "status"}}
        if not allowed:
            return self.get_ticket(tenant_id, ticket_id)
        sets = ", ".join(f"{k}=?" for k in allowed)
        values = list(allowed.values()) + [time.time(), tenant_id, ticket_id]
        with self._lock:
            self._conn.execute(
                f"UPDATE tickets SET {sets}, updated_at=? WHERE tenant_id=? AND ticket_id=?",
                values,
            )
            self._conn.commit()
        return self.get_ticket(tenant_id, ticket_id)


class HandoffService:
    """Pause an agent turn, persist state, issue a token, then resume it."""

    def __init__(self, db_path: str = ":memory:", ttl_seconds: int = 3600) -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.ttl_seconds = ttl_seconds
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def pause(
        self,
        tenant_id: str,
        conversation_id: str,
        state: dict[str, Any],
        reason: str = "caller_requested_human",
    ) -> dict[str, Any]:
        handoff_id = f"H-{uuid.uuid4().hex[:12].upper()}"
        token = _sign_token(
            {
                "handoff_id": handoff_id,
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "exp": int(time.time()) + self.ttl_seconds,
            }
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO handoffs (handoff_id, tenant_id, conversation_id, reason, "
                "state_json, status, created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    handoff_id,
                    tenant_id,
                    conversation_id,
                    reason,
                    json.dumps(state, default=str),
                    "paused",
                    time.time(),
                ),
            )
            self._conn.commit()
        return {"handoff_id": handoff_id, "handoff_token": token, "status": "paused"}

    def resume(self, token: str) -> dict[str, Any]:
        claims = _verify_token(token)
        if claims is None:
            return {"ok": False, "error": "invalid_or_expired_token"}
        handoff_id = claims["handoff_id"]
        tenant_id = claims["tenant_id"]
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM handoffs WHERE handoff_id=? AND tenant_id=?",
                (handoff_id, tenant_id),
            ).fetchone()
            if row is None:
                return {"ok": False, "error": "not_found"}
            if row["status"] != "paused":
                return {"ok": False, "error": f"already_{row['status']}"}
            self._conn.execute(
                "UPDATE handoffs SET status='resumed', resumed_at=? WHERE handoff_id=?",
                (time.time(), handoff_id),
            )
            self._conn.commit()
        return {
            "ok": True,
            "handoff_id": handoff_id,
            "tenant_id": tenant_id,
            "conversation_id": row["conversation_id"],
            "reason": row["reason"],
            "state": json.loads(row["state_json"]),
            "status": "resumed",
        }

    def get(self, tenant_id: str, handoff_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM handoffs WHERE tenant_id=? AND handoff_id=?",
                (tenant_id, handoff_id),
            ).fetchone()
        return dict(row) if row else None


def _sign_token(payload: dict[str, Any]) -> str:
    import jwt

    from core.config import settings

    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _verify_token(token: str) -> dict[str, Any] | None:
    import jwt

    from core.config import settings

    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        return None
