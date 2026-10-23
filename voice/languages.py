"""Language routing for ASR -> Piper voice selection.

Whisper supplies a language id + probability. We map supported languages to a
Piper voice and fall back to the configured default voice when the language is
unsupported or the utterance looks code-switched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_LANGUAGE = "en"
DEFAULT_VOICE = "en_US-lessac-medium"

LANG_TO_VOICE: dict[str, str] = {
    "en": "en_US-lessac-medium",
    "es": "es_ES-sharvard-medium",
}

# Small cue lists used only to *flag* code-switching; the ASR text remains the
# source of truth for what the caller said.
_SPANISH_CUES = {
    "hola",
    "gracias",
    "por",
    "favor",
    "quiero",
    "necesito",
    "reembolso",
    "pedido",
    "ayuda",
    "queja",
    "pero",
    "tambien",
    "también",
}
_ENGLISH_CUES = {"the", "please", "refund", "order", "ticket", "help", "but", "also", "and"}


@dataclass
class LanguageDecision:
    language: str
    voice: str
    confidence: float
    code_switched: bool = False
    reason: str = "detected"
    cues: list[str] = field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-záéíóúñü]+", (text or "").lower()))


def detect_language(text: str, whisper_language: str | None, confidence: float = 1.0) -> str:
    """Return the routed language id (whisper's id when confident)."""
    lang = (whisper_language or "").lower().split("-")[0]
    if lang and confidence >= 0.55 and lang in LANG_TO_VOICE:
        return lang
    toks = _tokens(text)
    es_hits = len(toks & _SPANISH_CUES)
    en_hits = len(toks & _ENGLISH_CUES)
    if es_hits > en_hits:
        return "es"
    if lang:
        return lang
    return DEFAULT_LANGUAGE


def is_code_switched(text: str, language: str) -> bool:
    toks = _tokens(text)
    if language == "en":
        return bool(toks & _SPANISH_CUES)
    if language == "es":
        return bool(toks & _ENGLISH_CUES)
    return False


def route_voice(
    text: str,
    whisper_language: str | None = None,
    confidence: float = 1.0,
    default_voice: str = DEFAULT_VOICE,
) -> LanguageDecision:
    language = detect_language(text, whisper_language, confidence)
    voice = LANG_TO_VOICE.get(language, default_voice)
    switched = is_code_switched(text, language)
    reason = "whisper" if language == (whisper_language or "").lower().split("-")[0] else "lexical"
    if switched:
        # A mixed utterance remains intelligible in the default voice; routing it
        # to a single non-default voice would mispronounce the other language.
        voice = default_voice
        reason = "code_switch_fallback"
    cues = sorted(_tokens(text) & (_SPANISH_CUES | _ENGLISH_CUES))
    return LanguageDecision(
        language=language,
        voice=voice,
        confidence=confidence,
        code_switched=switched,
        reason=reason,
        cues=cues,
    )
