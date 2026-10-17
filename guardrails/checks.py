"""Ingress/egress guardrail stubs + audit logging.

These are intentionally lightweight stubs that establish the *contract* and the
audit trail (the `guardrail_events` table) so the full Phase 6 policy engine can
drop in behind the same interface:

    verdict = ingress_check(question, trace_id)   # {action, rule, reason}
    verdict = egress_check(answer, trace_id)

`action` is one of: passed | redacted | blocked. Every call is recorded to
`guardrail_events` and mirrored into Prometheus counters.
"""
from __future__ import annotations

import re
from typing import Any

from core.db import execute
from core.logging import get_logger
from observability.metrics import GUARDRAIL_BLOCKS, GUARDRAIL_EVENTS

log = get_logger("guardrails.checks")

# Prompt-injection / jailbreak style phrases (ingress).
_INJECTION_PATTERNS = (
    re.compile(r"ignore (all )?previous instructions", re.I),
    re.compile(r"disregard (the )?(system|above)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"reveal (your )?(system )?prompt", re.I),
    re.compile(r"print (the )?(api|secret|password|token)", re.I),
)

# Secrets / PII that must never leave in an answer (egress).
_SECRET_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
)

_MAX_QUESTION = 4000


def log_event(
    trace_id: str | None,
    direction: str,
    rule: str,
    action: str,
    reason: str = "",
) -> None:
    """Persist a guardrail decision to `guardrail_events` and metrics."""
    try:
        GUARDRAIL_EVENTS.labels(direction=direction, action=action).inc()
        if action == "blocked":
            GUARDRAIL_BLOCKS.labels(direction=direction, rule=rule).inc()
    except Exception:  # pragma: no cover - metrics best effort
        pass
    try:
        execute(
            "INSERT INTO guardrail_events (trace_id, direction, rule, action, reason) "
            "VALUES (%s, %s, %s, %s, %s)",
            (trace_id, direction, rule, action, reason),
        )
    except Exception as exc:  # pragma: no cover - audit best effort
        log.warning("guardrail_log_failed", error=str(exc))


def ingress_check(question: str, trace_id: str | None = None) -> dict[str, Any]:
    """Validate an incoming question. Returns a verdict dict."""
    q = (question or "").strip()
    if not q:
        verdict = {"action": "blocked", "rule": "empty_question", "reason": "question is empty"}
    elif len(q) > _MAX_QUESTION:
        verdict = {
            "action": "blocked",
            "rule": "max_length",
            "reason": f"question exceeds {_MAX_QUESTION} chars",
        }
    else:
        hit = next((p for p in _INJECTION_PATTERNS if p.search(q)), None)
        if hit is not None:
            verdict = {
                "action": "blocked",
                "rule": "prompt_injection",
                "reason": "possible prompt-injection / jailbreak pattern",
            }
        else:
            verdict = {"action": "passed", "rule": "ingress_default", "reason": ""}
    log_event(trace_id, "ingress", verdict["rule"], verdict["action"], verdict["reason"])
    return verdict


def egress_check(answer: str, trace_id: str | None = None) -> dict[str, Any]:
    """Validate an outgoing answer; redact secrets if present."""
    a = answer or ""
    for pat in _SECRET_PATTERNS:
        if pat.search(a):
            redacted = pat.sub("[REDACTED]", a)
            verdict = {
                "action": "redacted",
                "rule": "secret_redaction",
                "reason": "sensitive token pattern redacted from answer",
                "redacted_answer": redacted,
            }
            log_event(trace_id, "egress", verdict["rule"], verdict["action"], verdict["reason"])
            return verdict
    verdict = {"action": "passed", "rule": "egress_default", "reason": ""}
    log_event(trace_id, "egress", verdict["rule"], verdict["action"], verdict["reason"])
    return verdict
