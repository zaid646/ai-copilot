"""Bridge from the voice session to an agent backend.

``support`` (default for voice) uses the deterministic support agent plus the
SQLite ticketing mock. ``analytics`` calls the existing LangGraph supervisor
with ``voice_mode=True``; it requires the normal Postgres/Neo4j/LLM stack.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from agents.voice_support import SupportAgent
from integrations.ticketing import TicketingSystem


@dataclass
class AgentResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    backend: str = "support"
    refused: bool = False


class AgentBackend(Protocol):
    def respond(self, text: str, *, tenant_id: str, session_id: str) -> AgentResponse: ...


class SupportBackend:
    def __init__(self, ticketing: TicketingSystem | None = None) -> None:
        self.agent = SupportAgent(ticketing)

    def respond(self, text: str, *, tenant_id: str, session_id: str) -> AgentResponse:
        reply = self.agent.handle_text(text, tenant_id, session_id)
        return AgentResponse(
            text=reply.text,
            tool_calls=reply.tool_calls,
            backend="support",
            refused=reply.refused,
        )


class AnalyticsBackend:
    """Runs the existing supervisor graph; voice style is injected in voice mode."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def respond(self, text: str, *, tenant_id: str, session_id: str) -> AgentResponse:
        from agents.graph import run_agent

        with self._lock:
            final = run_agent(
                text,
                tenant_id=tenant_id,
                voice_mode=True,
                max_iterations=4,
            )
        return AgentResponse(
            text=final.get("answer", "") or "I couldn't produce an answer.",
            tool_calls=final.get("tool_calls", []) or [],
            backend="analytics",
        )


def build_backend(name: str) -> AgentBackend:
    if name == "analytics":
        return AnalyticsBackend()
    return SupportBackend()
