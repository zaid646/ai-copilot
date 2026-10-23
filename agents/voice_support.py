"""Deterministic support agent used by the voice pipeline's offline mode.

Why this exists
---------------
The analytics supervisor is the right agent for the enterprise copilot, but the
voice scenarios (booking, complaint, refund, handoff, code-switching) need a
small support domain plus a ticket system. This module is a real, rule-based
agent (not an LLM stub): every reply is derived from the caller text and every
write goes through :class:`integrations.ticketing.TicketingSystem`.

It deliberately records structured tool calls so trajectory evaluation can score
tool choice, argument validity and call order without parsing prose.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from integrations.ticketing import TicketingSystem
from voice.speech_style import (
    confirmation_prompt,
    is_affirmative,
    is_correction,
    is_negative,
)

_TICKET_RE = re.compile(r"\b(T-[A-Z0-9]{6,12})\b", re.IGNORECASE)
_ORDER_RE = re.compile(r"\b(?:order|pedido)?\s*([A-Z]\d{3,6}|ORD-\d{3,6})\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"(?<![A-Za-z0-9])\$?\s*(\d{1,5}(?:\.\d{1,2})?)\s*(?:dollars|usd|euros|bucks)?", re.IGNORECASE)
_SPLIT_ORDER_RE = re.compile(r"\b([A-Za-z])\s*\$?\s*(\d{3,6})\b")
_TICKET_SPLIT_RE = re.compile(r"\bT\s*[- ]?\s*([A-Za-z]{2,})\s+(\d{3,6})\b", re.IGNORECASE)
_TICKET_ASR_RE = re.compile(r"\bT-[A-Z]{2,}(\d{6})\b", re.IGNORECASE)
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_OUT_OF_SCOPE = (
    "weather",
    "medical",
    "diagnose",
    "legal advice",
    "stock price",
    "crypto",
    "lottery",
    "joke",
    "meaning of life",
    "clima",
    "médico",
    "abogado",
)
_COMPLAINT_WORDS = ("complaint", "complain", "broken", "damaged", "late", "missing", "wrong item", "queja")
_REFUND_WORDS = ("refund", "money back", "reembolso", "return my money")
_STATUS_WORDS = ("status", "where is", "track", "estado", "dónde está", "donde esta")
_BOOKING_WORDS = ("book", "booking", "schedule", "appointment", "reserve", "reservar", "cita")


@dataclass
class SupportSession:
    session_id: str
    tenant_id: str
    customer_id: str = "cust-001"
    history: list[dict[str, str]] = field(default_factory=list)
    pending: dict[str, Any] | None = None
    ticket_ids: list[str] = field(default_factory=list)
    last_ticket_id: str | None = None
    turn: int = 0


@dataclass
class AgentReply:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, Any] | None = None
    language: str = "en"
    refused: bool = False


class SupportAgent:
    def __init__(self, ticketing: TicketingSystem | None = None) -> None:
        self.ticketing = ticketing or TicketingSystem()
        self.sessions: dict[str, SupportSession] = {}

    def session(self, session_id: str, tenant_id: str) -> SupportSession:
        key = f"{tenant_id}:{session_id}"
        if key not in self.sessions:
            self.sessions[key] = SupportSession(session_id=session_id, tenant_id=tenant_id)
        return self.sessions[key]

    # ------------------------------------------------------------------
    def handle_text(self, text: str, tenant_id: str, session_id: str) -> AgentReply:
        session = self.session(session_id, tenant_id)
        session.turn += 1
        text = _TICKET_SPLIT_RE.sub(r"T-\1\2", text or "")
        text = _TICKET_ASR_RE.sub(r"T-\1", text or "")
        text = _THOUSANDS_RE.sub("", text or "")
        text = _SPLIT_ORDER_RE.sub(r"\1\2", text or "")
        session.history.append({"role": "user", "content": text})
        low = text.lower().strip()

        if is_correction(low) and session.pending:
            reply = self._apply_correction(session, text)
        elif session.pending and is_affirmative(low):
            reply = self._execute_pending(session)
        elif session.pending and is_negative(low):
            session.pending = None
            reply = AgentReply("Okay, I won't make that change. What would you like to do instead?")
        elif any(word in low for word in _OUT_OF_SCOPE):
            reply = AgentReply(
                "I can't help with that, but I can help with bookings, complaints, "
                "refunds, or ticket status. Which one do you need?",
                refused=True,
            )
        elif any(word in low for word in _REFUND_WORDS):
            reply = self._refund(session, text)
        elif any(word in low for word in _COMPLAINT_WORDS):
            reply = self._complaint(session, text)
        elif any(word in low for word in _STATUS_WORDS):
            reply = self._status(session, text)
        elif any(word in low for word in _BOOKING_WORDS):
            reply = self._booking(session, text)
        elif _TICKET_RE.search(text):
            reply = self._status(session, text)
        else:
            reply = AgentReply(
                "I can help with a booking, a complaint, a refund, or a ticket status. "
                "Which would you like?"
            )
        session.history.append({"role": "assistant", "content": reply.text})
        return reply

    # ------------------------------------------------------------------
    def _status(self, session: SupportSession, text: str) -> AgentReply:
        match = _TICKET_RE.search(text)
        ticket_id = match.group(1).upper() if match else session.last_ticket_id
        if not ticket_id:
            return AgentReply("What is the ticket ID? It starts with T-.")
        ticket = self.ticketing.get_ticket(session.tenant_id, ticket_id)
        calls = [{"tool": "lookup_ticket", "args": {"ticket_id": ticket_id}, "seq": 0}]
        if ticket is None:
            calls[0]["result"] = {"found": False}
            return AgentReply(
                f"I couldn't find {ticket_id} in this account. Could you check the ID?",
                tool_calls=calls,
            )
        status = ticket.get("status", "open")
        queue = ticket.get("queue") or "the general queue"
        calls[0]["result"] = {"ticket_id": ticket_id, "status": status, "queue": queue}
        return AgentReply(
            f"Ticket {ticket_id} is {status} and is with {queue}. Anything else?",
            tool_calls=calls,
        )

    def _booking(self, session: SupportSession, text: str) -> AgentReply:
        date = _extract_date(text)
        if not date:
            return AgentReply("What day and time should I book?")
        session.pending = {"action": "booking", "date": date, "subject": f"Booking request for {date}"}
        return AgentReply(
            confirmation_prompt(f"a booking for {date}"),
            pending=session.pending,
        )

    def _complaint(self, session: SupportSession, text: str) -> AgentReply:
        order = _ORDER_RE.search(text)
        subject = "Customer complaint"
        if order:
            subject = f"Complaint about order {order.group(1).upper()}"
        session.pending = {"action": "complaint", "subject": subject, "order_id": order.group(1).upper() if order else None}
        return AgentReply(
            confirmation_prompt(f"registering a complaint: {subject}"),
            pending=session.pending,
        )

    def _refund(self, session: SupportSession, text: str) -> AgentReply:
        order = _ORDER_RE.search(text)
        amounts = [float(m.group(1)) for m in _AMOUNT_RE.finditer(text) if m.group(1)]
        amount = amounts[-1] if amounts else None
        if not order:
            return AgentReply("Which order is the refund for? For example, A123.")
        if amount is None:
            return AgentReply("How much should the refund be?")
        session.pending = {
            "action": "refund",
            "order_id": order.group(1).upper(),
            "amount": amount,
            "subject": f"Refund for order {order.group(1).upper()} (${amount:.2f})",
        }
        return AgentReply(
            confirmation_prompt(f"a ${amount:.2f} refund for order {order.group(1).upper()}"),
            pending=session.pending,
        )

    def _apply_correction(self, session: SupportSession, text: str) -> AgentReply:
        pending = dict(session.pending or {})
        order = _ORDER_RE.search(text)
        amounts = [float(m.group(1)) for m in _AMOUNT_RE.finditer(text) if m.group(1)]
        if order:
            pending["order_id"] = order.group(1).upper()
            pending["subject"] = f"{pending.get('action', 'request')} for order {order.group(1).upper()}"
        if amounts and pending.get("action") == "refund":
            pending["amount"] = amounts[-1]
            pending["subject"] = (
                f"Refund for order {pending.get('order_id', 'unknown')} (${amounts[-1]:.2f})"
            )
        date = _extract_date(text)
        if date and pending.get("action") == "booking":
            pending["date"] = date
            pending["subject"] = f"Booking request for {date}"
        session.pending = pending
        summary = _summarize(pending)
        return AgentReply(f"Thanks — corrected. {confirmation_prompt(summary)}", pending=pending)

    def _execute_pending(self, session: SupportSession) -> AgentReply:
        pending = dict(session.pending or {})
        session.pending = None
        action = pending.get("action")
        calls: list[dict[str, Any]] = []

        ticket = self.ticketing.create_ticket(
            session.tenant_id,
            session.customer_id,
            pending.get("subject", "Voice request"),
            category=action,
            priority="high" if action == "complaint" else "normal",
        )
        calls.append(
            {
                "tool": "create_ticket",
                "args": {
                    "subject": pending.get("subject"),
                    "category": action,
                    "priority": "high" if action == "complaint" else "normal",
                },
                "seq": 0,
            }
        )
        ticket_id = ticket["ticket_id"]
        calls[0]["result"] = {"ticket_id": ticket_id}
        session.ticket_ids.append(ticket_id)
        session.last_ticket_id = ticket_id

        category = {
            "refund": "billing",
            "complaint": "complaint",
            "booking": "booking",
        }.get(action, "general")
        self.ticketing.classify(session.tenant_id, ticket_id, category)
        calls.append(
            {"tool": "classify_ticket", "args": {"ticket_id": ticket_id, "category": category}, "seq": 1}
        )

        queue = {"refund": "billing", "complaint": "support", "booking": "scheduling"}.get(
            action, "general"
        )
        self.ticketing.route(session.tenant_id, ticket_id, queue)
        calls.append(
            {"tool": "route_ticket", "args": {"ticket_id": ticket_id, "queue": queue}, "seq": 2}
        )

        self.ticketing.notify(session.tenant_id, ticket_id, "sms")
        calls.append(
            {"tool": "notify_customer", "args": {"ticket_id": ticket_id, "channel": "sms"}, "seq": 3}
        )

        if action == "refund":
            text = (
                f"Refund request logged as {ticket_id}. It's routed to billing and "
                "you'll get a text when it's reviewed."
            )
        elif action == "complaint":
            text = f"Complaint registered as {ticket_id} and routed to support. We'll text you updates."
        else:
            text = f"Booked. Your confirmation is {ticket_id}; it's with scheduling."
        return AgentReply(text, tool_calls=calls)


def _extract_date(text: str) -> str | None:
    low = text.lower()
    m = re.search(
        r"\b(?:on\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"tomorrow|today|next week|january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\b(?:\s+(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?",
        low,
    )
    if m:
        return m.group(0).replace("on ", "").strip()
    m = re.search(r"\b\d{4}-\d{2}-\d{2}\b", low)
    return m.group(0) if m else None


def _summarize(pending: dict[str, Any]) -> str:
    action = pending.get("action", "request")
    if action == "refund":
        return f"a ${float(pending.get('amount', 0)):.2f} refund for order {pending.get('order_id')}"
    if action == "booking":
        return f"a booking for {pending.get('date')}"
    return f"a complaint: {pending.get('subject')}"


def new_session_id() -> str:
    return f"S-{uuid.uuid4().hex[:8]}"
