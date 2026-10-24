"""Tool registry + dispatcher.

The planner sees `tool_schemas()`; the graph executes via `dispatch()`, which
guarantees every result is a contract envelope even if a tool raises.
"""
from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from core.logging import get_logger
from tools.contract import ErrorCode, err
from tools.forecast import forecast_demand
from tools.graphrag import graphrag_search
from tools.kpi import compute_kpi
from tools.sql_tool import run_sql

log = get_logger("tools.registry")

_TOOL_TRACE: ContextVar[list[dict[str, Any]] | None] = ContextVar("tool_trace", default=None)


def reset_tool_trace() -> None:
    _TOOL_TRACE.set([])


def current_tool_trace() -> list[dict[str, Any]]:
    trace = _TOOL_TRACE.get()
    return list(trace or [])


def _record_tool(name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
    trace = _TOOL_TRACE.get()
    if trace is None:
        return
    trace.append(
        {
            "seq": len(trace),
            "tool": name,
            "args": dict(args or {}),
            "ok": bool(result.get("ok", False)),
            "error_code": result.get("error_code"),
        }
    )

TOOLS: dict[str, dict[str, Any]] = {
    "run_sql": {
        "fn": run_sql,
        "description": (
            "Run a single read-only analytical SQL SELECT against the warehouse "
            "(tables: fact_sales, fact_finance, fact_customer, dim_region, dim_product, "
            "dim_date, agg_daily_kpi). Use for ad-hoc breakdowns."
        ),
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A read-only SELECT query"}},
            "required": ["sql"],
        },
    },
    "compute_kpi": {
        "fn": compute_kpi,
        "description": (
            "Deterministic KPI rollup (revenue, units, orders, gross margin, AOV) for a "
            "region and quarter, with the previous quarter and deltas. Preferred over raw SQL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "enum": ["WEST", "EAST", "CENTRAL", "SOUTH"]},
                "quarter": {"type": "string", "description": "e.g. '2026Q3'"},
                "category": {
                    "type": "string",
                    "enum": ["Beverages", "Dairy", "Household", "Produce", "Snacks"],
                },
            },
            "required": ["region", "quarter"],
        },
    },
    "graphrag_search": {
        "fn": graphrag_search,
        "description": (
            "Semantic search over company policy PDFs (finance + retail). Use to ground "
            "recommendations in stated policy. Returns chunks with source and page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                "source_filter": {
                    "type": "string",
                    "description": (
                        "Restrict retrieval to one source document. Use 'Retail' for the "
                        "retail/produce manual or 'Finance' for the budget/accounting manual."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    "forecast_demand": {
        "fn": forecast_demand,
        "description": (
            "Weekly revenue forecast from the ML model service for a (region, category). "
            "Use to project next weeks and quantify a gap versus target."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "enum": ["WEST", "EAST", "CENTRAL", "SOUTH"]},
                "category": {
                    "type": "string",
                    "enum": ["Beverages", "Dairy", "Household", "Produce", "Snacks"],
                },
                "horizon_weeks": {"type": "integer", "minimum": 1, "maximum": 12},
            },
            "required": ["region", "category"],
        },
    },
}


def get_tool(name: str) -> Callable[..., dict[str, Any]]:
    return TOOLS[name]["fn"]


def tool_names() -> list[str]:
    return list(TOOLS)


def tool_schemas() -> list[dict[str, Any]]:
    """Compact schema list for the planner prompt."""
    return [
        {"name": name, "description": spec["description"], "parameters": spec["parameters"]}
        for name, spec in TOOLS.items()
    ]


def dispatch(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = TOOLS.get(name)
    if spec is None:
        result = err(ErrorCode.INPUT_INVALID, f"unknown tool {name!r}; valid: {tool_names()}")
        _record_tool(name, args or {}, result)
        return result
    try:
        result = spec["fn"](**(args or {}))
    except TypeError as exc:
        result = err(ErrorCode.INPUT_INVALID, f"bad arguments for {name}: {exc}")
    except Exception as exc:  # noqa: BLE001 - never leak a raw crash to the graph
        log.error("tool crashed", tool=name, error=str(exc))
        result = err(ErrorCode.TOOL_ERROR, f"{name} crashed: {type(exc).__name__}: {exc}")
    _record_tool(name, args or {}, result)
    return result
