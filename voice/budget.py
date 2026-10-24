"""Per-hop voice latency telemetry + Prometheus emission.

Hop definitions (all measured in milliseconds):

* ``asr_ms``     end-of-speech -> final transcript
* ``llm_ms``     transcript -> agent response text
* ``tts_ms``     agent text -> first synthesized audio frame
* ``network_ms`` first audio frame produced -> first frame handed to the socket
* ``e2e_ms``     end-of-speech -> first agent audio frame sent

``e2e_ms`` is the number compared with the configured budget (default 1000 ms).
The start-of-turn timestamp is captured when VAD detects end-of-speech, not when
the caller started talking, so the budget measures responsiveness after the
user stops speaking.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_client import Counter, Histogram

_BUCKETS = (25, 50, 100, 150, 250, 375, 500, 750, 1000, 1500, 2000, 3000, 5000)

VOICE_ASR_MS = Histogram(
    "voice_asr_ms", "Voice ASR latency (end-of-speech to final transcript, ms)", buckets=_BUCKETS
)
VOICE_LLM_MS = Histogram(
    "voice_llm_ms", "Voice agent latency (transcript to response text, ms)", buckets=_BUCKETS
)
VOICE_TTS_MS = Histogram(
    "voice_tts_ms", "Voice TTS first-audio latency (agent text to first frame, ms)", buckets=_BUCKETS
)
VOICE_NETWORK_MS = Histogram(
    "voice_network_ms", "Voice first-audio socket send latency (ms)", buckets=_BUCKETS
)
VOICE_E2E_MS = Histogram(
    "voice_e2e_ms",
    "Voice end-of-speech to first agent audio (ms)",
    buckets=_BUCKETS,
)
VOICE_BARGEIN_TOTAL = Counter(
    "voice_bargein_total", "Barge-in events detected while agent audio was playing"
)
VOICE_TURNS_TOTAL = Counter(
    "voice_turns_total", "Voice turns processed", ["outcome"]
)


@dataclass
class TurnBudget:
    """Accumulates one turn's hop timings and emits Prometheus observations."""

    budget_ms: float = 1000.0
    end_of_speech_at: float = field(default_factory=time.perf_counter)
    asr_done_at: float | None = None
    llm_done_at: float | None = None
    tts_first_at: float | None = None
    first_audio_sent_at: float | None = None
    barge_in: bool = False
    cancelled: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    # --- hop markers -----------------------------------------------------
    def mark_asr(self) -> None:
        self.asr_done_at = time.perf_counter()

    def mark_llm(self) -> None:
        self.llm_done_at = time.perf_counter()

    def mark_tts_first(self) -> None:
        self.tts_first_at = time.perf_counter()

    def mark_first_audio_sent(self) -> None:
        self.first_audio_sent_at = time.perf_counter()

    # --- derived ---------------------------------------------------------
    def _ms(self, end: float | None) -> float | None:
        if end is None:
            return None
        return round((end - self.end_of_speech_at) * 1000.0, 2)

    def asr_ms(self) -> float | None:
        return self._ms(self.asr_done_at)

    def llm_ms(self) -> float | None:
        if self.asr_done_at is None or self.llm_done_at is None:
            return None
        return round((self.llm_done_at - self.asr_done_at) * 1000.0, 2)

    def tts_ms(self) -> float | None:
        if self.llm_done_at is None or self.tts_first_at is None:
            return None
        return round((self.tts_first_at - self.llm_done_at) * 1000.0, 2)

    def network_ms(self) -> float | None:
        if self.tts_first_at is None or self.first_audio_sent_at is None:
            return None
        return round((self.first_audio_sent_at - self.tts_first_at) * 1000.0, 2)

    def e2e_ms(self) -> float | None:
        return self._ms(self.first_audio_sent_at)

    def budget_ok(self) -> bool | None:
        value = self.e2e_ms()
        if value is None:
            return None
        return value <= self.budget_ms

    def observe(self, labels: dict[str, str] | None = None) -> dict[str, Any]:
        """Emit hop observations once and return the JSON-serialisable record."""
        del labels  # reserved for per-tenant dashboards; metrics are global today
        values = {
            "asr_ms": self.asr_ms(),
            "llm_ms": self.llm_ms(),
            "tts_ms": self.tts_ms(),
            "network_ms": self.network_ms(),
            "e2e_ms": self.e2e_ms(),
            "budget_ms": self.budget_ms,
            "budget_ok": self.budget_ok(),
            "barge_in": self.barge_in,
            "cancelled": self.cancelled,
        }
        for name, metric in (
            ("asr_ms", VOICE_ASR_MS),
            ("llm_ms", VOICE_LLM_MS),
            ("tts_ms", VOICE_TTS_MS),
            ("network_ms", VOICE_NETWORK_MS),
            ("e2e_ms", VOICE_E2E_MS),
        ):
            value = values[name]
            if value is not None:
                metric.observe(float(value))
        return values


def metric_count(metric: Any) -> float:
    """Sum the observation count for a Histogram (test/debug helper)."""
    total = 0.0
    for family in metric.collect():
        for sample in family.samples:
            if sample.name.endswith("_count"):
                total += float(sample.value)
    return total
