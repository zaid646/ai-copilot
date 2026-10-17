"""Tool: run_sql — guarded, read-only analytical SQL over the warehouse."""
from __future__ import annotations

import datetime as _dt
import time
from decimal import Decimal
from typing import Any

import psycopg

from core.db import connection
from core.logging import get_logger
from guardrails.sql_guard import validate_sql
from observability.metrics import TOOL_CALLS, TOOL_LATENCY
from tools.contract import ErrorCode, err, ok

log = get_logger("tools.sql")

STATEMENT_TIMEOUT_MS = 6000


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    return value


def _rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: to_jsonable(v) for k, v in row.items()} for row in rows]


def run_sql(sql: str, params: list[Any] | None = None, timeout_ms: int = STATEMENT_TIMEOUT_MS) -> dict[str, Any]:
    """Execute a single read-only SELECT and return the rows."""
    start = time.perf_counter()
    okv, normalized, reason = validate_sql(sql)
    if not okv:
        TOOL_CALLS.labels(tool="run_sql", status="rejected").inc()
        log.warning("sql rejected", reason=reason)
        return err(ErrorCode.SQL_REJECTED, reason or "invalid query")

    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
                cur.execute(normalized, params or [])
                rows = cur.fetchall() if cur.description else []
    except psycopg.errors.QueryCanceled:
        TOOL_CALLS.labels(tool="run_sql", status="timeout").inc()
        return err(ErrorCode.UPSTREAM_TIMEOUT, f"query exceeded {timeout_ms}ms")
    except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
        TOOL_CALLS.labels(tool="run_sql", status="error").inc()
        return err(ErrorCode.UPSTREAM_TIMEOUT, f"database unavailable: {exc}")
    except psycopg.Error as exc:
        TOOL_CALLS.labels(tool="run_sql", status="error").inc()
        return err(ErrorCode.SQL_REJECTED, f"query failed: {exc}")
    finally:
        TOOL_LATENCY.labels(tool="run_sql").observe(time.perf_counter() - start)

    if not rows:
        TOOL_CALLS.labels(tool="run_sql", status="empty").inc()
        return err(ErrorCode.NO_RESULTS, "query returned no rows")

    TOOL_CALLS.labels(tool="run_sql", status="success").inc()
    return ok({"row_count": len(rows), "rows": _rows(rows), "sql": normalized})
