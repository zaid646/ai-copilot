"""Offline tests for the LLM-planned support agent (LLM mocked)."""
from __future__ import annotations

from typing import Any

from agents.llm_support import LLMSupportAgent
from integrations.ticketing import TicketingSystem


def plan(
    action: str,
    *,
    ticket_id: str | None = None,
    order_id: str | None = None,
    amount: float | None = None,
    date: str | None = None,
    reply: str = "Okay.",
    confirmed: bool = False,
    refused: bool = False,
) -> dict[str, Any]:
    return {
        "action": action,
        "args": {
            "ticket_id": ticket_id,
            "order_id": order_id,
            "amount": amount,
            "date": date,
        },
        "reply": reply,
        "confirmed": confirmed,
        "refused": refused,
    }


def test_llm_refund_requires_confirmation_then_executes(monkeypatch):
    responses = iter(
        [
            plan("refund", order_id="B123", amount=30, reply="Confirm the $30 refund?"),
            plan("refund", order_id="B123", amount=30, confirmed=True),
        ]
    )
    calls: list[dict[str, Any]] = []

    def fake_chat_json(system, user, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr("agents.llm_support.chat_json", fake_chat_json)
    ticketing = TicketingSystem()
    agent = LLMSupportAgent(ticketing, model="fake-model")

    first = agent.handle_text("Refund order B123 for thirty dollars.", "t1", "s1")
    assert "confirm" in first.text.lower()
    assert first.tool_calls == []
    assert "B123" in first.text

    second = agent.handle_text("Yes, confirm the refund.", "t1", "s1")
    assert second.text.startswith("Refund request logged")
    assert [c["tool"] for c in second.tool_calls] == [
        "create_ticket",
        "classify_ticket",
        "route_ticket",
        "notify_customer",
    ]
    assert second.tool_calls[0]["args"]["category"] == "refund"
    assert second.tool_calls[1]["args"]["category"] == "billing"
    assert second.tool_calls[2]["args"]["queue"] == "billing"
    assert ticketing.notifications("t1")
    assert calls[0]["model"] == "fake-model"
    assert calls[0]["use_cache"] is False


def test_llm_correction_preserves_amount_and_requires_reconfirmation(monkeypatch):
    responses = iter(
        [
            plan("refund", order_id="B123", amount=30),
            plan("refund", order_id="B124", amount=50),
            plan("refund", order_id="B124", amount=30, confirmed=True),
        ]
    )
    monkeypatch.setattr("agents.llm_support.chat_json", lambda *a, **k: next(responses))
    ticketing = TicketingSystem()
    agent = LLMSupportAgent(ticketing)

    agent.handle_text("Refund order B123 for thirty dollars.", "t1", "s1")
    corrected = agent.handle_text("No, I said B124.", "t1", "s1")
    assert "confirm" in corrected.text.lower()
    assert "$30.00" in corrected.text
    assert "B124" in corrected.text

    done = agent.handle_text("Yes, confirm the refund.", "t1", "s1")
    subject = done.tool_calls[0]["args"]["subject"]
    assert "B124" in subject and "$30.00" in subject


def test_llm_status_looks_up_ticket(monkeypatch):
    ticketing = TicketingSystem()
    ticketing.create_ticket("t1", "cust-001", "Printer issue", ticket_id="T-123456")
    monkeypatch.setattr(
        "agents.llm_support.chat_json",
        lambda *a, **k: plan("status", ticket_id="T-123456"),
    )
    agent = LLMSupportAgent(ticketing)

    reply = agent.handle_text("What is the status of ticket T-123456?", "t1", "s1")
    assert "T-123456" in reply.text
    assert "open" in reply.text
    assert reply.tool_calls[0]["tool"] == "lookup_ticket"
    assert reply.tool_calls[0]["args"]["ticket_id"] == "T-123456"


def test_llm_out_of_scope_refuses_without_tools(monkeypatch):
    monkeypatch.setattr(
        "agents.llm_support.chat_json",
        lambda *a, **k: plan(
            "out_of_scope",
            reply="I can't help with weather, but I can help with a refund.",
            refused=True,
        ),
    )
    agent = LLMSupportAgent(TicketingSystem())

    reply = agent.handle_text("What's the weather in Tokyo?", "t1", "s1")
    assert reply.refused is True
    assert "can't help" in reply.text
    assert reply.tool_calls == []
