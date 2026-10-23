"""Streaming voice layer for the analytics/support copilot.

The package is intentionally dependency-light at import time: Silero VAD,
faster-whisper and Piper are imported lazily so the CPU-only unit tests and CI
can run with small wheels and deterministic fakes.
"""

from voice.budget import TurnBudget
from voice.config import VoiceSettings

__all__ = ["TurnBudget", "VoiceSettings"]
