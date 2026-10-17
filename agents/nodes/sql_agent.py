"""SQL agent node: deterministic KPI first + optional breakdown SQL."""
from __future__ import annotations

from typing import Any

from agents.nodes.base import llm_json
from agents.tracelib import traced
from tools.dispatcher import dispatch


@traced("sql_agent")
def sql_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    step = state.get("current_step", {}) or {}
    task = step.get("task", state.get("question", ""))
    args = step.get("args", {}) or {}

    evidence = dict(state.get("evidence", {}))
    scratchpad = list(state.get("scratchpad", []))
    notes = {"node": "sql_agent", "kpi": None, "breakdown": None}

    region = args.get("region")
    quarter = args.get("quarter")
    category = args.get("category") if args.get("category") else None

    # compute_kpi if region+quarter provided
    if region and quarter:
        try:
            kpi = dispatch("compute_kpi", {"region": region, "quarter": quarter, "category": category})
            evidence["kpi"] = kpi
            notes["kpi"] = "ok" if kpi.get("ok") else kpi.get("error_code")
        except Exception as exc:
            evidence["kpi_error"] = str(exc)
            notes["kpi"] = "error"

    # breakdown SQL if task implies breakdown
    if any(word in (task or "").lower() for word in ("breakdown", "by ", "driver", "category", "channel", "week")):
        try:
            bd = llm_json(
                "sql_agent",
                {
                    "question": state.get("question", ""),
                    "task": task,
                    "args": {"region": region, "quarter": quarter, "category": category},
                },
            )
            sql = bd.get("breakdown_sql") if isinstance(bd, dict) else None
            if sql:
                res = dispatch("run_sql", {"sql": sql})
                evidence["breakdown"] = res
                notes["breakdown"] = "ok" if res.get("ok") else res.get("error_code")
        except Exception as exc:
            evidence["breakdown_error"] = str(exc)
            notes["breakdown"] = "error"

    scratchpad.append(notes)
    return {"evidence": evidence, "scratchpad": scratchpad}
