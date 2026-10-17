"""Synthesis agent."""
from __future__ import annotations

from typing import Any

from agents.nodes.base import _load_prompt
from agents.tracelib import traced
from core.llm import chat_json


@traced("action_agent")
def action_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    context = {
        "question": state.get("question", ""),
        "evidence": state.get("evidence", {}),
        "history": state.get("history", [])[-3:],
    }
    prompt = _load_prompt("action_agent", context)
    try:
        out = chat_json("Synthesize a grounded answer. Be precise with numbers and cite policy.", prompt)
        return {
            "answer": out.get("answer", ""),
            "citations": out.get("citations", []) or [],
            "scratchpad": list(state.get("scratchpad", []))
            + [{"node": "action_agent", "out_len": len(out.get("answer", ""))}],
        }
    except Exception as exc:
        return {
            "answer": f"**Answer synthesis failed:** {exc}",
            "citations": [],
            "scratchpad": list(state.get("scratchpad", []))
            + [{"node": "action_agent", "error": str(exc)}],
            "status": "failed",
        }
