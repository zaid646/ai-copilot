"""Shared LangGraph state for the copilot supervisor.

`AgentState` flows through every node. Nodes return *partial* updates (LangGraph
merges them). Keys split into three groups:

  - identity/IO      : trace_id, question, history, answer
  - working memory   : plan, current_step, scratchpad, evidence
  - control/telemetry: next_agent, iteration, status, steps

`evidence` is the shared blackboard: sql_agent writes headline KPIs + a
breakdown, graphrag_agent writes policy chunks, model_agent writes a forecast.
The action_agent synthesizes from `evidence`; the critic_agent verifies it.
"""
from __future__ import annotations

from typing import Any, TypedDict


class PlanStep(TypedDict, total=False):
    agent: str  # sql_agent | graphrag_agent | model_agent
    task: str  # natural-language instruction for the agent
    args: dict[str, Any]  # suggested arguments (region, quarter, category, query...)


class AgentState(TypedDict, total=False):
    # --- identity / IO ---
    trace_id: str
    user_id: int | None
    conversation_id: int | None
    question: str
    history: list[dict[str, Any]]  # prior turns: [{"role","content"}]
    answer: str
    citations: list[dict[str, Any]]
    tenant_id: str
    voice_mode: bool

    # --- planning / working memory ---
    domain: str  # retail | finance (drives retrieval source filter)
    intent: str  # free-form intent label, e.g. "sales_drop", "forecast"
    plan: list[PlanStep]
    plan_idx: int
    current_step: PlanStep
    scratchpad: list[dict[str, Any]]  # per-node notes
    evidence: dict[str, Any]  # {"kpi":..., "breakdown":..., "docs":..., "forecast":...}

    # --- control / telemetry ---
    next_agent: str
    iteration: int
    max_iterations: int
    status: str  # running | completed | failed | blocked
    error: str | None
    critique: dict[str, Any]
    steps: list[dict[str, Any]]  # node-transition trace (mirrored to Postgres)
