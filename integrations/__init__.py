"""Mock integrations used by the voice agent and handoff path (no paid APIs)."""

from integrations.ticketing import HandoffService, TicketingSystem

__all__ = ["HandoffService", "TicketingSystem"]
