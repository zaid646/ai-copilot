"""LangGraph supervisor graph."""
from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.memory import finish_trace, load_history, save_message, start_trace
from agents.nodes.action_agent import action_agent_node
from agents.nodes.critic_agent import critic_agent_node
from agents.nodes.graphrag_agent import graphrag_agent_node
from agents.nodes.model_agent import model_agent_node
from agents.nodes.planner import planner_node
from agents.nodes.sql_agent import sql_agent_node
from agents.nodes.supervisor import (
    route_from_critic,
    route_from_supervisor,
    supervisor_node,
)
from agents.state import AgentState
from core.logging import get_logger, set_trace_id
from tools.registry import current_tool_trace, reset_tool_trace

log = get_logger("agents.graph")
_CHECKPOINTER = MemorySaver()


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("sql_agent", sql_agent_node)
    g.add_node("graphrag_agent", graphrag_agent_node)
    g.add_node("model_agent", model_agent_node)
    g.add_node("action_agent", action_agent_node)
    g.add_node("critic_agent", critic_agent_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "sql_agent": "sql_agent",
            "graphrag_agent": "graphrag_agent",
            "model_agent": "model_agent",
            "action_agent": "action_agent",
        },
    )
    g.add_edge("sql_agent", "supervisor")
    g.add_edge("graphrag_agent", "supervisor")
    g.add_edge("model_agent", "supervisor")
    g.add_edge("action_agent", "critic_agent")
    g.add_conditional_edges(
        "critic_agent",
        route_from_critic,
        {
            "action_agent": "action_agent",
            END: END,
        },
    )
    return g.compile(checkpointer=_CHECKPOINTER)


APP = build_graph()


def run_agent(
    question: str,
    user_id: int | None = None,
    conversation_id: int | None = None,
    history: list[dict[str, Any]] | None = None,
    max_iterations: int = 8,
    trace_id: str | None = None,
    tenant_id: str = "default",
    voice_mode: bool = False,
) -> dict[str, Any]:
    tid = trace_id or uuid4().hex
    set_trace_id(tid)
    start = time.perf_counter()
    reset_tool_trace()

    hist = history
    if hist is None and conversation_id is not None:
        try:
            hist = load_history(conversation_id)
        except Exception:
            hist = []

    try:
        start_trace(tid, question, user_id, conversation_id, tenant_id)
    except Exception:
        pass

    init: dict[str, Any] = {
        "trace_id": tid,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "tenant_id": tenant_id,
        "voice_mode": voice_mode,
        "question": question,
        "history": hist or [],
        "plan": [],
        "plan_idx": 0,
        "current_step": {},
        "scratchpad": [],
        "evidence": {},
        "answer": "",
        "citations": [],
        "critique": {},
        "next_agent": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "status": "running",
        "error": None,
        "steps": [],
    }

    final = APP.invoke(
        init,
        config={
            "configurable": {"thread_id": tid},
            "recursion_limit": 4 * max_iterations + 8,
        },
    )

    dt_ms = int((time.perf_counter() - start) * 1000)
    tokens = None
    try:
        finish_trace(
            tid,
            final.get("answer", ""),
            final.get("status") or "completed",
            len(final.get("steps", [])),
            dt_ms,
            tokens,
            guardrail_blocked=False,
        )
    except Exception:
        pass

    if conversation_id is not None:
        try:
            save_message(conversation_id, "user", question, tid, tenant_id)
            save_message(conversation_id, "assistant", final.get("answer", ""), tid, tenant_id)
        except Exception:
            pass

    final["tool_calls"] = current_tool_trace()
    return final
