"""Supervisor node and routing."""
from __future__ import annotations

from typing import Any

from langgraph.graph import END

from agents.domain import classify_domain, source_filter_for
from agents.tracelib import traced


@traced("supervisor")
def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
    plan = state.get("plan") or []
    idx = int(state.get("plan_idx", 0))
    iteration = int(state.get("iteration", 0)) + 1
    max_it = int(state.get("max_iterations", 8))
    # Domain routing: classify the question before picking tools. Persist so
    # downstream nodes (graphrag source filter, critic checks) can rely on it.
    domain = state.get("domain") or classify_domain(state.get("question", ""))
    if idx >= len(plan) or iteration > max_it:
        return {
            "next_agent": "action_agent",
            "iteration": iteration,
            "plan_idx": idx,
            "domain": domain,
        }
    step = plan[idx]
    agent = step.get("agent")
    allowed = {"sql_agent", "graphrag_agent", "model_agent"}
    if agent not in allowed:
        agent = "action_agent"
        return {"next_agent": agent, "iteration": iteration, "plan_idx": idx, "domain": domain}
    # Ensure retrieval steps carry a source filter aligned to the domain.
    if agent == "graphrag_agent":
        args = dict(step.get("args", {}) or {})
        args.setdefault("source_filter", source_filter_for(domain))
        step = {**step, "args": args}
    return {
        "next_agent": agent,
        "current_step": step,
        "plan_idx": idx + 1,
        "iteration": iteration,
        "domain": domain,
    }


def route_from_supervisor(state: dict[str, Any]) -> str:
    na = state.get("next_agent") or "action_agent"
    if na in ("sql_agent", "graphrag_agent", "model_agent", "action_agent"):
        return na
    return "action_agent"


def route_from_critic(state: dict[str, Any]) -> str:
    na = state.get("next_agent") or END
    if na == "action_agent":
        return "action_agent"
    if na == "END":
        return END
    return END
