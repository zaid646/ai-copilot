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
        # Determine whether the answer is SQL-grounded or document-grounded so
        # the egress guardrail only enforces citations for retrieval answers.
        evidence = state.get("evidence", {}) or {}
        evidence_keys = set(evidence.get("evidence_keys", []) or [])
        sql_keys = {"kpi", "breakdown", "forecast", "model"}
        docs = evidence.get("docs") or {}
        docs_results = docs.get("results", []) if isinstance(docs, dict) else []
        if evidence_keys.intersection(sql_keys):
            answer_source = "sql"
        elif evidence_keys or docs_results:
            answer_source = "retrieval"
        else:
            answer_source = "sql"
        return {
            "answer": out.get("answer", ""),
            "citations": out.get("citations", []) or [],
            "answer_source": answer_source,
            "scratchpad": list(state.get("scratchpad", []))
            + [{"node": "action_agent", "out_len": len(out.get("answer", "")), "source": answer_source}],
        }
    except Exception as exc:
        return {
            "answer": f"**Answer synthesis failed:** {exc}",
            "citations": [],
            "scratchpad": list(state.get("scratchpad", []))
            + [{"node": "action_agent", "error": str(exc)}],
            "status": "failed",
        }
