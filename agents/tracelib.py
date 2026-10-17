"""Tracing decorator and helpers for recording agent graph transitions."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agents.memory import record_step
from observability.metrics import AGENT_STEP_LATENCY, AGENT_STEPS

if TYPE_CHECKING:
    from typing import Any


def traced(node_name: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Decorate a node function to record timing, metrics, and trace persistence.

    The wrapped node must accept and return partial AgentState dicts.
    """
    def deco(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            t0 = time.perf_counter()
            out = fn(state)
            dt_s = time.perf_counter() - t0
            lat_ms = int(dt_s * 1000)
            out = dict(out or {})
            steps = list(state.get("steps", []))
            seq = len(steps)
            # summarize inputs/outputs to keep payloads small
            input_obj = {
                "question": state.get("question"),
                "current_step": state.get("current_step"),
                "plan_idx": state.get("plan_idx"),
                "next_agent": state.get("next_agent"),
            }
            output_obj = {
                "next_agent": out.get("next_agent"),
                "evidence_keys": list((out.get("evidence") or {}).keys()),
                "scratchpad_len": len(out.get("scratchpad") or (state.get("scratchpad") or [])),
                "status": out.get("status"),
                "iteration": out.get("iteration"),
            }
            # persist trace row
            tid = state.get("trace_id")
            if tid:
                try:
                    record_step(tid, seq, node_name, input_obj, output_obj, lat_ms)
                except Exception:  # pragma: no cover
                    pass
            # update metrics
            AGENT_STEPS.labels(node=node_name).inc()
            AGENT_STEP_LATENCY.labels(node=node_name).observe(dt_s)
            out["steps"] = steps + [
                {
                    "seq": seq,
                    "node": node_name,
                    "latency_ms": lat_ms,
                    "next_agent": out.get("next_agent"),
                }
            ]
            return out
        return wrapper
    return deco
