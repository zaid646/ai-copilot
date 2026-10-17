"""Planner node."""
from __future__ import annotations

from typing import Any

from agents.nodes.base import llm_json
from agents.tracelib import traced


@traced("planner")
def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    question = state.get("question", "")
    history = state.get("history", [])
    data = llm_json(
        "planner",
        {
            "question": question,
            "history": history,
        },
    )
    steps = data.get("steps", []) if isinstance(data, dict) else []
    valid = []
    for s in steps[:5]:
        agent = s.get("agent")
        if agent in {"sql_agent", "graphrag_agent", "model_agent"}:
            valid.append(
                {
                    "agent": agent,
                    "task": s.get("task", question),
                    "args": s.get("args", {}) or {},
                }
            )
    return {
        "plan": valid,
        "plan_idx": 0,
        "scratchpad": list(state.get("scratchpad", []))
        + [{"node": "planner", "steps": valid}],
        "next_agent": "",
    }
