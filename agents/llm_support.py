"""LLM-planned support agent for the voice pipeline.

The LLM chooses the caller's intent and slot values from the live conversation
state. Deterministic code owns the safety boundary: it validates arguments,
requires explicit confirmation before any write, executes tenant-scoped
ticketing tools, and records the structured trajectory used by the eval.

The regex :mod:`agents.voice_support` agent remains available as an offline
unit-test mock, but it is not the default voice backend.
"""
from __future__ import annotations

import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.llm import chat_json
from core.logging import get_logger
from integrations.ticketing import TicketingSystem
from voice.speech_style import confirmation_prompt, is_affirmative, is_correction, is_negative

log = get_logger("agents.llm_support")

DEFAULT_MODEL = os.getenv("VOICE_AGENT_MODEL", "ornith-1.5-9b")

_TICKET_RE = re.compile(r"\b(T-[A-Z0-9]{6,12})\b", re.IGNORECASE)
_ORDER_RE = re.compile(r"\b(?:order|pedido)?\s*([A-Z]\d{3,6}|ORD-\d{3,6})\b", re.IGNORECASE)
_TICKET_SPACE_RE = re.compile(r"\bT\s*[- ]?\s*(\d{6,12})\b", re.IGNORECASE)
_TICKET_SPLIT_RE = re.compile(r"\bT\s*[- ]?\s*([A-Za-z]{2,})\s+(\d{3,6})\b", re.IGNORECASE)
_TICKET_ASR_RE = re.compile(r"\bT-([A-Za-z]{2,})(\d{6})\b", re.IGNORECASE)
_SPLIT_ORDER_RE = re.compile(r"\b([A-Za-z])\s*\$?\s*(\d{3,6})\b")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_AMOUNT_HINT_RE = re.compile(
    r"(\$|\b(?:dollars?|usd|euros?|bucks?|cents?)\b)", re.IGNORECASE
)

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["status", "refund", "complaint", "booking", "out_of_scope", "other"],
        },
        "args": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": ["string", "null"]},
                "order_id": {"type": ["string", "null"]},
                "amount": {"type": ["number", "null"]},
                "date": {"type": ["string", "null"]},
            },
            "required": ["ticket_id", "order_id", "amount", "date"],
            "additionalProperties": False,
        },
        "reply": {"type": "string"},
        "confirmed": {"type": "boolean"},
        "refused": {"type": "boolean"},
    },
    "required": ["action", "args", "reply", "confirmed", "refused"],
    "additionalProperties": False,
}

_SYSTEM = """You are the planner for a customer-support voice agent. You DO have
these tools and may act on the caller's behalf:
- lookup_ticket(ticket_id) for ticket status.
- create_ticket, classify_ticket, route_ticket, notify_customer for refunds,
  complaints, and bookings.

Rules:
1. status: action=status and args.ticket_id comes from the caller.
2. refund/complaint/booking: action is that request. Fill order_id, amount, or
   date only when the caller stated it. Ask a short confirmation question and
   set confirmed=false.
3. If the state says awaiting_confirmation=true and the caller clearly says yes,
   repeat the same action/args and set confirmed=true. Do not invent values.
4. If the caller corrects one value, change only that value and keep
   confirmed=false.
5. Weather, medical/legal advice, and other unrelated requests: action=
   out_of_scope, refused=true, and refuse in one short sentence.
6. If no action applies, action=other and give a short helpful reply.
7. reply is one short spoken sentence. Keep ticket IDs and amounts exact.
"""


@dataclass
class SupportSession:
    session_id: str
    tenant_id: str
    customer_id: str = "cust-001"
    history: list[dict[str, str]] = field(default_factory=list)
    pending: dict[str, Any] | None = None
    last_ticket_id: str | None = None
    turn: int = 0


@dataclass
class AgentReply:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, Any] | None = None
    refused: bool = False


class LLMSupportAgent:
    """Thin LLM-tool agent over the ticketing domain."""

    def __init__(
        self,
        ticketing: TicketingSystem | None = None,
        *,
        model: str | None = None,
        seed: int = 42,
        use_cache: bool = False,
    ) -> None:
        self.ticketing = ticketing or TicketingSystem()
        self.model = model or DEFAULT_MODEL
        self.seed = seed
        self.use_cache = use_cache
        self._lock = threading.RLock()
        self.sessions: dict[str, SupportSession] = {}

    def session(self, session_id: str, tenant_id: str) -> SupportSession:
        key = f"{tenant_id}:{session_id}"
        with self._lock:
            if key not in self.sessions:
                self.sessions[key] = SupportSession(session_id=session_id, tenant_id=tenant_id)
            return self.sessions[key]

    def handle_text(self, text: str, tenant_id: str, session_id: str) -> AgentReply:
        session = self.session(session_id, tenant_id)
        session.turn += 1
        text = _normalize(text or "")
        plan = self._plan(session, text)
        try:
            reply = self._dispatch(session, text, plan)
        except Exception as exc:  # defensive: one bad turn must not kill the call
            log.warning("llm support dispatch failed", error=str(exc))
            reply = AgentReply("Sorry, I couldn't complete that request. Please try again.")
        session.history.append({"role": "user", "content": text})
        session.history.append({"role": "assistant", "content": reply.text})
        return reply

    # ------------------------------------------------------------------
    def _plan(self, session: SupportSession, text: str) -> dict[str, Any]:
        state = {
            "pending": session.pending,
            "awaiting_confirmation": session.pending is not None,
            "last_ticket_id": session.last_ticket_id,
        }
        recent = session.history[-6:]
        prompt = (
            f"STATE:\n{state}\n\n"
            f"RECENT TURNS:\n{recent or '(none)'}\n\n"
            f"CALLER: {text}"
        )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return chat_json(
                    _SYSTEM,
                    prompt,
                    temperature=0.0 if attempt == 0 else 0.1,
                    think=False,
                    seed=self.seed + attempt,
                    model=self.model,
                    schema=_PLAN_SCHEMA if attempt == 0 else None,
                    max_tokens=400,
                    use_cache=self.use_cache,
                )
            except Exception as exc:
                last_error = exc
                log.warning(
                    "llm support plan attempt failed",
                    attempt=attempt + 1,
                    error=str(exc),
                )
        log.warning("llm support plan failed", error=str(last_error))
        return {
            "action": "other",
            "args": {},
            "reply": "Sorry, I couldn't understand that. Could you say it again?",
            "confirmed": False,
            "refused": False,
        }

    def _dispatch(self, session: SupportSession, text: str, plan: dict[str, Any]) -> AgentReply:
        action = str(plan.get("action") or "other").lower()
        args = plan.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        refused = bool(plan.get("refused")) or action == "out_of_scope"
        if refused:
            return AgentReply(
                text=(
                    "I can't help with that, but I can help with refunds, "
                    "complaints, bookings, or ticket status."
                ),
                refused=True,
            )

        if action == "status":
            return self._status(session, args, text)

        if action in {"refund", "complaint", "booking"}:
            return self._write(session, action, args, text, plan)

        if session.pending and self._is_confirmation(plan, text):
            return self._execute_pending(session)

        if session.pending and is_negative(text) and not is_correction(text):
            session.pending = None
            return AgentReply("Okay, I won't make that change. What would you like to do instead?")

        return AgentReply(
            text=str(plan.get("reply") or "I can help with a refund, a complaint, a booking, or a ticket status.")
        )

    # ------------------------------------------------------------------
    def _status(self, session: SupportSession, args: dict[str, Any], text: str) -> AgentReply:
        ticket_id = _ticket(args.get("ticket_id")) or _extract_ticket(text) or session.last_ticket_id
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

    def _write(
        self,
        session: SupportSession,
        action: str,
        args: dict[str, Any],
        text: str,
        plan: dict[str, Any],
    ) -> AgentReply:
        pending = _merge_pending(session.pending, action, args, text)
        if not pending:
            return AgentReply(_missing_slot(action, pending))

        if session.pending and self._is_confirmation(plan, text):
            changed = _material_change(session.pending, pending)
            session.pending = pending
            if changed:
                return AgentReply(
                    confirmation_prompt(_summarize(pending)),
                    pending=dict(pending),
                )
            return self._execute_pending(session)

        session.pending = pending
        return AgentReply(
            confirmation_prompt(_summarize(pending)),
            pending=dict(pending),
        )

    @staticmethod
    def _is_confirmation(plan: dict[str, Any], text: str) -> bool:
        if is_correction(text) or is_negative(text):
            return False
        return bool(plan.get("confirmed")) or is_affirmative(text)

    def _execute_pending(self, session: SupportSession) -> AgentReply:
        pending = dict(session.pending or {})
        session.pending = None
        action = str(pending.get("action") or "other")
        category = {"refund": "billing", "complaint": "complaint", "booking": "booking"}.get(
            action, "general"
        )
        queue = {"refund": "billing", "complaint": "support", "booking": "scheduling"}.get(
            action, "general"
        )
        priority = "high" if action == "complaint" else "normal"
        subject = str(pending.get("subject") or "Voice request")

        ticket = self.ticketing.create_ticket(
            session.tenant_id,
            session.customer_id,
            subject,
            category=action,
            priority=priority,
        )
        ticket_id = str(ticket.get("ticket_id") or "")
        calls: list[dict[str, Any]] = [
            {
                "tool": "create_ticket",
                "args": {"subject": subject, "category": action, "priority": priority},
                "seq": 0,
                "result": {"ticket_id": ticket_id},
            }
        ]
        session.last_ticket_id = ticket_id

        self.ticketing.classify(session.tenant_id, ticket_id, category)
        calls.append(
            {
                "tool": "classify_ticket",
                "args": {"ticket_id": ticket_id, "category": category},
                "seq": 1,
            }
        )
        self.ticketing.route(session.tenant_id, ticket_id, queue)
        calls.append(
            {
                "tool": "route_ticket",
                "args": {"ticket_id": ticket_id, "queue": queue},
                "seq": 2,
            }
        )
        self.ticketing.notify(session.tenant_id, ticket_id, "sms")
        calls.append(
            {
                "tool": "notify_customer",
                "args": {"ticket_id": ticket_id, "channel": "sms"},
                "seq": 3,
            }
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


def _normalize(text: str) -> str:
    text = _THOUSANDS_RE.sub("", text)
    text = _TICKET_SPACE_RE.sub(r"T-\1", text)
    text = _TICKET_SPLIT_RE.sub(r"T-\1\2", text)
    text = _TICKET_ASR_RE.sub(r"T-\1", text)
    return _SPLIT_ORDER_RE.sub(r"\1\2", text)


def _ticket(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).upper().strip()
    match = _TICKET_RE.search(raw)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d{6,12}", raw):
        return f"T-{raw}"
    return None


def _extract_ticket(text: str) -> str | None:
    match = _TICKET_RE.search(text)
    return match.group(1).upper() if match else None


def _order(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).upper().strip()
    match = _ORDER_RE.search(raw)
    if match:
        return match.group(1).upper()
    if re.fullmatch(r"ORD-\d{3,6}", raw):
        return raw
    return None


def _extract_order(text: str) -> str | None:
    match = _ORDER_RE.search(text)
    return match.group(1).upper() if match else None


def _amount(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 2) if number >= 0 else None


def _merge_pending(
    old: dict[str, Any] | None, action: str, args: dict[str, Any], text: str
) -> dict[str, Any] | None:
    base: dict[str, Any] = dict(old) if old and old.get("action") == action else {}
    order = _order(args.get("order_id")) or _extract_order(text)
    amount = _amount(args.get("amount"))
    date = str(args.get("date")).strip() if args.get("date") else None

    if order:
        base["order_id"] = order
    if amount is not None:
        base["amount"] = amount
    if date:
        base["date"] = date

    if action == "refund":
        if (
            old
            and old.get("action") == "refund"
            and old.get("amount") is not None
            and not _AMOUNT_HINT_RE.search(text)
        ):
            base["amount"] = old["amount"]
        order_id = base.get("order_id")
        amount_value = base.get("amount")
        if not order_id or amount_value is None:
            return None
        base["subject"] = f"Refund for order {order_id} (${float(amount_value):.2f})"
    elif action == "complaint":
        order_id = base.get("order_id")
        base["subject"] = f"Complaint about order {order_id}" if order_id else "Customer complaint"
    elif action == "booking":
        booking_date = base.get("date")
        if not booking_date:
            return None
        base["subject"] = f"Booking request for {booking_date}"

    base["action"] = action
    return base


def _material_change(old: dict[str, Any], new: dict[str, Any]) -> bool:
    for key in ("order_id", "amount", "date"):
        if old.get(key) != new.get(key):
            return True
    return False


def _missing_slot(action: str, pending: dict[str, Any] | None) -> str:
    if pending is None:
        if action == "refund":
            return "Which order is the refund for, and how much should it be?"
        if action == "booking":
            return "What day and time should I book?"
    return "Could you give me the missing details?"


def _summarize(pending: dict[str, Any]) -> str:
    action = pending.get("action", "request")
    if action == "refund":
        return f"a ${float(pending.get('amount', 0)):.2f} refund for order {pending.get('order_id')}"
    if action == "booking":
        return f"a booking for {pending.get('date')}"
    return f"a complaint about order {pending.get('order_id')}"


def new_session_id() -> str:
    return f"S-{uuid.uuid4().hex[:8]}"
