"""Spoken-dialogue style rules injected into voice-mode agent prompts.

These are deliberately short and composable so the same rules apply to the
analytics LangGraph supervisor and to the offline support agent used by the
simulated-caller eval.
"""
from __future__ import annotations

import re

VOICE_SYSTEM_PROMPT = """\
You are speaking to a caller on a live voice line.

Voice rules (non-negotiable):
1. SHORT TURNS: answer in one or two short sentences. Never read a list or a
   table aloud. Offer the detail instead of dumping it.
2. CONFIRM BEFORE ACTION: before creating a ticket, issuing a refund, changing a
   booking, or any write action, restate the key arguments (ID, amount, date)
   and ask for a yes/no confirmation.
3. MISHEARING RECOVERY: if a value is ambiguous or the caller corrects you,
   say "I heard X — is that right?" and wait. Never guess a number or an ID.
4. NO FILLER: do not say "As an AI" or describe internal tools. Speak the result.
5. ONE QUESTION AT A TIME: collect missing slots in priority order.
6. SPELL-BACK: read back confirmation codes and ticket IDs digit by digit.
"""

CONFIRMATION_TEMPLATE = (
    "Just to confirm: {summary}. Should I go ahead?"
)

MISHEAR_TEMPLATE = "I heard {heard!r} — is that right?"

_CORRECTION_PATTERNS = (
    r"\bno[, ]+i said\b",
    r"\bno[, ]+not\b",
    r"\bthat'?s (wrong|not right|incorrect)\b",
    r"\bcorrection\b",
    r"\bi meant\b",
    r"\bactually\b",
    r"\bperd[oó]n\b",
    r"\bno[, ]+era\b",
)


def inject(base_prompt: str) -> str:
    """Append the voice style block to a base system prompt exactly once."""
    if not base_prompt:
        return VOICE_SYSTEM_PROMPT
    if "Voice rules (non-negotiable)" in base_prompt:
        return base_prompt
    return f"{base_prompt.rstrip()}\n\n{VOICE_SYSTEM_PROMPT}"


def is_correction(text: str) -> bool:
    low = (text or "").lower().strip()
    return any(re.search(pattern, low) for pattern in _CORRECTION_PATTERNS)


def confirmation_prompt(summary: str) -> str:
    return CONFIRMATION_TEMPLATE.format(summary=summary)


def mishear_prompt(heard: str) -> str:
    return MISHEAR_TEMPLATE.format(heard=heard)


def is_affirmative(text: str) -> bool:
    return bool(re.search(r"\b(yes|yeah|yep|correct|right|confirm|go ahead|sí|si)\b", (text or "").lower()))


def is_negative(text: str) -> bool:
    return bool(re.search(r"\b(no|nope|wrong|not right|cancel|stop)\b", (text or "").lower()))
