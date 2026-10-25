"""Memory + trace persistence for the LangGraph supervisor."""
from __future__ import annotations

import json
from typing import Any

from core.db import execute, query
from core.logging import get_logger

log = get_logger("agents.memory")


def start_trace(
    trace_id: str,
    question: str,
    user_id: int | None = None,
    conversation_id: int | None = None,
    tenant_id: str = "default",
) -> None:
    try:
        execute(
            "INSERT INTO agent_traces (trace_id, user_id, question, status, tenant_id) "
            "VALUES (%s, %s, %s, 'running', %s) ON CONFLICT (trace_id) DO NOTHING",
            (trace_id, user_id, question, tenant_id),
        )
    except Exception as exc:  # pragma: no cover - persistence best effort
        log.warning("start_trace_failed", error=str(exc))


def record_step(
    trace_id: str,
    seq: int,
    node: str,
    input_obj: Any,
    output_obj: Any,
    latency_ms: int,
) -> None:
    try:
        execute(
            "INSERT INTO agent_steps (trace_id, seq, node, input_json, output_json, latency_ms) "
            "VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s) ON CONFLICT (trace_id, seq) DO NOTHING",
            (
                trace_id,
                seq,
                node,
                json.dumps(input_obj, default=str),
                json.dumps(output_obj, default=str),
                latency_ms,
            ),
        )
    except Exception as exc:  # pragma: no cover - persistence best effort
        log.warning("record_step_failed", error=str(exc))


def finish_trace(
    trace_id: str,
    answer: str,
    status: str,
    total_steps: int,
    latency_ms: int,
    tokens_used: int | None = None,
    guardrail_blocked: bool = False,
) -> None:
    try:
        execute(
            "UPDATE agent_traces "
            "SET answer=%s, status=%s, total_steps=%s, latency_ms=%s, "
            "tokens_used=%s, guardrail_blocked=%s, finished_at=now() "
            "WHERE trace_id=%s",
            (answer, status, total_steps, latency_ms, tokens_used, guardrail_blocked, trace_id),
        )
    except Exception as exc:  # pragma: no cover - persistence best effort
        log.warning("finish_trace_failed", error=str(exc))


def save_message(
    conversation_id: int,
    role: str,
    content: str,
    trace_id: str | None = None,
    tenant_id: str = "default",
) -> None:
    try:
        execute(
            "INSERT INTO messages (conversation_id, role, content, trace_id, tenant_id) "
            "VALUES (%s, %s, %s, %s, %s)",
            (conversation_id, role, content, trace_id, tenant_id),
        )
    except Exception as exc:  # pragma: no cover - persistence best effort
        log.warning("save_message_failed", error=str(exc))


def load_history(
    conversation_id: int, limit: int = 8, tenant_id: str = "default"
) -> list[dict[str, Any]]:
    try:
        rows = query(
            "SELECT role, content FROM messages "
            "WHERE conversation_id=%s AND tenant_id=%s "
            "ORDER BY message_id DESC LIMIT %s",
            (conversation_id, tenant_id, limit),
        )
    except Exception as exc:  # pragma: no cover - persistence best effort
        log.warning("load_history_failed", error=str(exc))
        return []
    return list(reversed([{"role": r["role"], "content": r["content"]} for r in rows]))
