"""Planner node."""
from __future__ import annotations

from typing import Any

from agents.domain import classify_domain, classify_intent
from agents.nodes.base import llm_json
from agents.tracelib import traced
from voice.speech_style import inject as inject_voice_style


@traced("planner")
def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    question = state.get("question", "")
    history = state.get("history", [])
    # Deterministic domain/intent up front so the supervisor and retrieval
    # nodes agree without an extra LLM round-trip.
    domain = classify_domain(question)
    intent = classify_intent(question)
    data = llm_json(
        "planner",
        {
            "question": question,
            "history": history,
            "domain": domain,
            "intent": intent,
            "voice_directive": inject_voice_style("") if state.get("voice_mode") else "",
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
        "domain": domain,
        "intent": intent,
        "plan": valid,
        "plan_idx": 0,
        "scratchpad": list(state.get("scratchpad", []))
        + [{"node": "planner", "steps": valid}],
        "next_agent": "",
    }
