"""Shared tool-result contract for the agent graph.

Every tool returns the same envelope so LangGraph nodes can route failures to
the critic without ad-hoc branching:

    {
      "ok": bool,
      "data": Any,          # tool payload when ok, else None
      "error_code": str|None,   # one of ErrorCode, else None
      "error_msg": str|None,
      "trace_id": str|None,     # inherited from the agent run
    }
"""
from __future__ import annotations

import enum
from typing import Any

from core.logging import get_trace_id


class ErrorCode(enum.StrEnum):
    INPUT_INVALID = "INPUT_INVALID"          # bad/missing args from the planner
    SQL_REJECTED = "SQL_REJECTED"            # guardrail/validator blocked the query
    NO_RESULTS = "NO_RESULTS"                # ran fine but returned zero rows
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"    # DB / model / LLM call timed out
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"  # forecast service down or artifact missing
    TOOL_ERROR = "TOOL_ERROR"                # unexpected internal failure


def ok(data: Any, trace_id: str | None = None) -> dict[str, Any]:
    return {
        "ok": True,
        "data": data,
        "error_code": None,
        "error_msg": None,
        "trace_id": trace_id or get_trace_id(),
    }


def err(code: ErrorCode, msg: str, trace_id: str | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error_code": code.value if isinstance(code, ErrorCode) else str(code),
        "error_msg": msg,
        "trace_id": trace_id or get_trace_id(),
    }


def is_ok(result: dict[str, Any]) -> bool:
    return bool(result.get("ok"))


def error_code_of(result: dict[str, Any]) -> ErrorCode | None:
    raw = result.get("error_code")
    if raw is None:
        return None
    try:
        return ErrorCode(raw)
    except ValueError:
        return ErrorCode.TOOL_ERROR
