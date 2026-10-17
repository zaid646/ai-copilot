"""Supervisor node and routing."""
from __future__ import annotations

from typing import Any

from langgraph.graph import END

from agents.tracelib import traced


@traced("supervisor")
def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
    plan = state.get("plan") or []
    idx = int(state.get("plan_idx", 0))
    iteration = int(state.get("iteration", 0)) + 1
    max_it = int(state.get("max_iterations", 8))
    if idx >= len(plan) or iteration > max_it:
        return {
            "next_agent": "action_agent",
            "iteration": iteration,
            "plan_idx": idx,
        }
    step = plan[idx]
    agent = step.get("agent")
    allowed = {"sql_agent", "graphrag_agent", "model_agent"}
    if agent not in allowed:
        agent = "action_agent"
        return {"next_agent": agent, "iteration": iteration, "plan_idx": idx}
    return {
        "next_agent": agent,
        "current_step": step,
        "plan_idx": idx + 1,
        "iteration": iteration,
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
