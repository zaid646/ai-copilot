"""Tool: compute_kpi — deterministic KPI rollup for a region (and optional category).

Returns current-quarter metrics plus the previous quarter and deltas, so the
agent can reason about "what changed" without writing SQL itself.
"""
from __future__ import annotations

import re
import time
from typing import Any

from core.db import query_one
from core.logging import get_logger
from observability.metrics import TOOL_CALLS, TOOL_LATENCY
from tools.contract import ErrorCode, err, ok

log = get_logger("tools.kpi")

_QUARTER_RE = re.compile(r"^\s*(\d{4})\s*[-/]?\s*[Qq]\s*([1-4])\s*$")

_METRIC_SQL = """
SELECT
    COALESCE(sum(f.revenue), 0)               AS revenue,
    COALESCE(sum(f.quantity), 0)              AS units,
    COALESCE(count(DISTINCT f.order_id), 0)   AS orders,
    COALESCE(sum(f.revenue - f.cost), 0)      AS gross_margin
FROM fact_sales f
JOIN dim_date d    ON f.date_id = d.date_id
JOIN dim_product p ON f.product_id = p.product_id
WHERE f.region_id = %s AND d.year = %s AND d.quarter = %s
  AND (%s::text IS NULL OR p.category = %s)
"""


def _prev_quarter(year: int, quarter: int) -> tuple[int, int]:
    return (year - 1, 4) if quarter == 1 else (year, quarter - 1)


def _with_derived(m: dict[str, Any]) -> dict[str, Any]:
    revenue = float(m["revenue"] or 0)
    orders = int(m["orders"] or 0)
    out = {
        "revenue": round(revenue, 2),
        "units": int(m["units"] or 0),
        "orders": orders,
        "gross_margin": round(float(m["gross_margin"] or 0), 2),
        "avg_order_value": round(revenue / orders, 2) if orders else 0.0,
    }
    return out


def _delta(cur: dict[str, Any], prev: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("revenue", "units", "orders", "gross_margin", "avg_order_value"):
        c, p = float(cur[k]), float(prev[k])
        out[k] = {
            "abs": round(c - p, 2),
            "pct": round((c - p) / p * 100, 2) if p else None,
        }
    return out


def compute_kpi(region: str, quarter: str, category: str | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        m = _QUARTER_RE.match(quarter or "")
        if not m:
            return err(ErrorCode.INPUT_INVALID, f"quarter must look like '2026Q3', got {quarter!r}")
        year, q = int(m.group(1)), int(m.group(2))

        region_id = (region or "").strip().upper()
        if not region_id:
            return err(ErrorCode.INPUT_INVALID, "region is required")
        exists = query_one("SELECT 1 AS one FROM dim_region WHERE region_id = %s", (region_id,))
        if not exists:
            known = [r["region_id"] for r in _all_regions()]
            return err(ErrorCode.INPUT_INVALID, f"unknown region {region!r}; known: {known}")

        cat = category or None
        cur_row = query_one(_METRIC_SQL, (region_id, year, q, cat, cat))
        if cur_row is None:
            return err(ErrorCode.NO_RESULTS, "no data for the requested slice")

        cur = _with_derived(cur_row)
        if cur["orders"] == 0:
            return err(ErrorCode.NO_RESULTS, f"no sales for {region_id} {year}Q{q}"
                       + (f" category={cat}" if cat else ""))

        py, pq = _prev_quarter(year, q)
        prev_row = query_one(_METRIC_SQL, (region_id, py, pq, cat, cat))
        prev = _with_derived(prev_row) if prev_row else _with_derived(
            {"revenue": 0, "units": 0, "orders": 0, "gross_margin": 0}
        )

        TOOL_CALLS.labels(tool="compute_kpi", status="success").inc()
        return ok({
            "region": region_id,
            "category": cat,
            "quarter": f"{year}Q{q}",
            "previous_quarter": f"{py}Q{pq}",
            "current": cur,
            "previous": prev,
            "delta": _delta(cur, prev),
        })
    finally:
        TOOL_LATENCY.labels(tool="compute_kpi").observe(time.perf_counter() - start)


def _all_regions() -> list[dict[str, Any]]:
    from core.db import query
    return query("SELECT region_id FROM dim_region ORDER BY region_id")
