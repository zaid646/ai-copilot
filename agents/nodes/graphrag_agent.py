"""GraphRAG agent node."""
from __future__ import annotations

from typing import Any

from agents.nodes.base import llm_json
from agents.tracelib import traced
from tools.dispatcher import dispatch


@traced("graphrag_agent")
def graphrag_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    step = state.get("current_step", {}) or {}
    task = step.get("task", state.get("question", ""))
    args = step.get("args", {}) or {}

    evidence = dict(state.get("evidence", {}))
    scratchpad = list(state.get("scratchpad", []))
    notes = {"node": "graphrag_agent", "query": None, "hits": 0}

    try:
        spec = llm_json(
            "graphrag_agent",
            {
                "question": state.get("question", ""),
                "task": task,
                "args": args,
            },
        )
        query = spec.get("query") or args.get("query") or task or state.get("question", "")
        top_k = int(spec.get("top_k") or args.get("top_k") or 5)
        res = dispatch("graphrag_search", {"query": query, "top_k": top_k})
        if res.get("ok"):
            data = res.get("data", {}) or {}
            evidence["docs"] = {
                "results": data.get("results", []),
                "count": data.get("count"),
                "strategy": data.get("strategy"),
            }
            notes["hits"] = evidence["docs"]["count"] or len(evidence["docs"]["results"])
            notes["query"] = query
        else:
            evidence["docs_error"] = res.get("error_msg")
            notes["hits"] = 0
    except Exception as exc:
        evidence["docs_error"] = str(exc)
        notes["hits"] = 0

    scratchpad.append(notes)
    return {"evidence": evidence, "scratchpad": scratchpad}
