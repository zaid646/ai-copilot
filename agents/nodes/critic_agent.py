"""Critic agent."""
from __future__ import annotations

from typing import Any

from agents.nodes.base import _load_prompt
from agents.tracelib import traced
from core.llm import chat_json


@traced("critic_agent")
def critic_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    evidence = state.get("evidence", {}) or {}
    docs = evidence.get("docs") or {}
    docs_results = docs.get("results", []) if isinstance(docs, dict) else []
    context = {
        "evidence": evidence,
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []) or [],
        "critique": state.get("critique", {}) or {},
    }
    prompt = _load_prompt("critic_agent", context)
    try:
        out = chat_json("Act as a strict, grounded critic. No hand-waving.", prompt)
    except Exception as exc:
        out = {"ok": False, "issues": [str(exc)], "decision": "revise", "instructions": "Fix grounding."}

    decision = out.get("decision", "revise") if isinstance(out, dict) else "revise"
    iteration = int(state.get("iteration", 0))
    max_it = int(state.get("max_iterations", 8))

    # Deterministic checks
    issues = list(out.get("issues", [])) if isinstance(out, dict) else []
    if docs_results and not (state.get("citations") or []):
        issues.append("Missing citations despite available policy docs")
        decision = "revise" if iteration < max_it - 1 else "pass"  # don't loop forever
    if not state.get("answer"):
        issues.append("Empty answer")
        decision = "revise"

    out.update({"issues": issues, "decision": decision})

    scratchpad = list(state.get("scratchpad", []))
    scratchpad.append({"node": "critic_agent", "decision": decision, "issues": issues})

    if decision == "pass" or decision == "fail":
        return {
            "critique": out,
            "status": "completed" if decision == "pass" else "blocked",
            "scratchpad": scratchpad,
            "next_agent": "END",
        }
    if iteration < max_it:
        return {
            "critique": out,
            "status": "running",
            "iteration": iteration + 1,
            "scratchpad": scratchpad,
            "next_agent": "action_agent",
        }
    return {
        "critique": out,
        "status": "completed",
        "scratchpad": scratchpad,
        "next_agent": "END",
    }
