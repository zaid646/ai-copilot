"""Bridge from the voice session to an agent backend.

``llm`` (default for voice) runs the LLM-planned support agent with the SQLite
ticketing mock. ``analytics`` calls the existing LangGraph supervisor with
``voice_mode=True``; it requires the normal Postgres/Neo4j/LLM stack. The
``support`` backend is the deterministic regex agent, kept only as an offline
unit-test mock and explicit fallback.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from agents.llm_support import LLMSupportAgent
from agents.voice_support import SupportAgent
from integrations.ticketing import TicketingSystem
from voice.config import VoiceSettings


@dataclass
class AgentResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    backend: str = "support"
    refused: bool = False


class AgentBackend(Protocol):
    def respond(self, text: str, *, tenant_id: str, session_id: str) -> AgentResponse: ...


class SupportBackend:
    """Deterministic regex support agent; offline test/fallback backend only."""

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


class LLMSupportBackend:
    """Real LLM-planned support agent with deterministic confirmation guards."""

    def __init__(
        self,
        ticketing: TicketingSystem | None = None,
        *,
        model: str | None = None,
        seed: int = 42,
        use_cache: bool = False,
        settings: VoiceSettings | None = None,
    ) -> None:
        settings = settings or VoiceSettings.from_env()
        self.agent = LLMSupportAgent(
            ticketing,
            model=model or settings.voice_llm_model,
            seed=seed,
            use_cache=use_cache,
            provider=settings.voice_llm_provider,
            base_url=settings.voice_llm_base_url,
        )

    def respond(self, text: str, *, tenant_id: str, session_id: str) -> AgentResponse:
        reply = self.agent.handle_text(text, tenant_id, session_id)
        return AgentResponse(
            text=reply.text,
            tool_calls=reply.tool_calls,
            backend="llm",
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
    if name in {"llm", "support_llm"}:
        return LLMSupportBackend()
    if name in {"support", "regex"}:
        return SupportBackend()
    raise ValueError(f"unknown voice agent backend: {name}")
