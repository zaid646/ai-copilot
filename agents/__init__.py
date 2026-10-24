"""Agent graph package.

Keep this module import-light: voice-only paths (support agent) must not pull
in LangGraph and the full analytics stack.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_graph", "run_agent"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from agents.graph import build_graph, run_agent

        return {"build_graph": build_graph, "run_agent": run_agent}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
