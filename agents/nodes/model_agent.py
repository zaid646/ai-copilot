"""Model agent node (forecast)."""
from __future__ import annotations

from typing import Any

from agents.nodes.base import llm_json
from agents.tracelib import traced
from tools.dispatcher import dispatch


def _top_category_from_evidence(evidence: dict[str, Any]) -> str | None:
    breakdown = evidence.get("breakdown")
    if breakdown and breakdown.get("ok"):
        rows = (breakdown.get("data") or {}).get("rows") or []
        # pick category with largest negative revenue delta? or largest drop magnitude — not trivial
        # but for demo: pick first category present or look for "revenue"
        if rows:
            r = rows[0]
            for k in ("category", "category_id"):
                if k in r:
                    return r[k]
    return None


@traced("model_agent")
def model_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    step = state.get("current_step", {}) or {}
    task = step.get("task", state.get("question", ""))
    args = step.get("args", {}) or {}

    evidence = dict(state.get("evidence", {}))
    scratchpad = list(state.get("scratchpad", []))
    notes = {"node": "model_agent", "status": None}

    try:
        spec = llm_json(
            "model_agent",
            {
                "question": state.get("question", ""),
                "task": task,
                "args": args,
                "evidence": {
                    "kpi": evidence.get("kpi"),
                    "breakdown": evidence.get("breakdown") if evidence.get("breakdown") else None,
                    "docs_count": (evidence.get("docs") or {}).get("count"),
                },
            },
        )
        region = spec.get("region") or args.get("region")
        category = spec.get("category") or args.get("category") or _top_category_from_evidence(evidence)
        horizon_weeks = int(spec.get("horizon_weeks") or args.get("horizon_weeks") or 4)
        if region and category:
            res = dispatch("forecast_demand", {"region": region, "category": category, "horizon_weeks": horizon_weeks})
            evidence["forecast"] = res
            notes["status"] = "ok" if res.get("ok") else res.get("error_code")
            notes["region"] = region
            notes["category"] = category
            notes["horizon_weeks"] = horizon_weeks
        else:
            notes["status"] = "skipped (no region/category)"
    except Exception as exc:
        evidence["forecast_error"] = str(exc)
        notes["status"] = "error"

    scratchpad.append(notes)
    return {"evidence": evidence, "scratchpad": scratchpad}
