"""Warm handoff pause/resume + ticketing tenant isolation."""
from __future__ import annotations

from integrations.ticketing import HandoffService, TicketingSystem


def test_handoff_pause_then_resume_restores_state():
    svc = HandoffService(ttl_seconds=60)
    state = {"transcript": "I need a human", "turn": 3, "pending": {"action": "refund"}}
    paused = svc.pause("tenant-a", "conv-9", state, reason="caller_requested_human")
    assert paused["status"] == "paused"
    assert paused["handoff_token"]

    # A different tenant cannot resume or read the handoff.
    assert svc.get("tenant-b", paused["handoff_id"]) is None

    resumed = svc.resume(paused["handoff_token"])
    assert resumed["ok"] is True
    assert resumed["tenant_id"] == "tenant-a"
    assert resumed["state"] == state
    assert resumed["status"] == "resumed"

    # Tokens are single-use.
    again = svc.resume(paused["handoff_token"])
    assert again["ok"] is False
    assert again["error"] in {"already_resumed", "not_found"}


def test_invalid_handoff_token_is_rejected():
    svc = HandoffService()
    assert svc.resume("not-a-token")["ok"] is False


def test_ticketing_is_tenant_scoped():
    tickets = TicketingSystem()
    ticket = tickets.create_ticket("tenant-a", "cust-1", "Broken item")
    assert tickets.get_ticket("tenant-a", ticket["ticket_id"]) is not None
    assert tickets.get_ticket("tenant-b", ticket["ticket_id"]) is None
    assert tickets.list_tickets("tenant-b") == []
    tickets.classify("tenant-a", ticket["ticket_id"], "complaint")
    tickets.route("tenant-a", ticket["ticket_id"], "support")
    tickets.notify("tenant-a", ticket["ticket_id"])
    assert len(tickets.notifications("tenant-a")) == 1
    assert tickets.notifications("tenant-b") == []
