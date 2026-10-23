"""Voice runtime settings.

Everything has a CPU-only default. The `large-v3` ASR model is opt-in via
``VOICE_ASR_MODEL`` and is expected to run on a GPU host (see the runbook).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "evals" / "fixtures" / "models"
WAV_DIR = ROOT / "evals" / "fixtures" / "wav"


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class VoiceSettings:
    sample_rate: int = 16000
    frame_ms: int = 32
    asr_model: str = "base.en"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_beam_size: int = 1
    asr_cpu_threads: int = 8
    asr_tail_ms: int = 96
    seed: int = 42
    vad_kind: str = "silero"
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 320
    vad_min_speech_ms: int = 96
    partial_interval_ms: int = 320
    tts_voice: str = "en_US-lessac-medium"
    tts_default_language: str = "en"
    model_dir: Path = MODEL_DIR
    budget_ms: int = 1000
    agent_backend: str = "llm"
    max_turn_seconds: float = 30.0
    partials_enabled: bool = True
    pacing: bool = False
    asr_initial_prompt: str | None = None
    voice_llm_provider: str | None = None
    voice_llm_base_url: str | None = None
    voice_llm_model: str | None = None

    @classmethod
    def from_env(cls) -> VoiceSettings:
        return cls(
            sample_rate=int(os.getenv("VOICE_SAMPLE_RATE", "16000")),
            frame_ms=int(os.getenv("VOICE_FRAME_MS", "32")),
            asr_model=os.getenv("VOICE_ASR_MODEL", "base.en"),
            asr_device=os.getenv("VOICE_ASR_DEVICE", "cpu"),
            asr_compute_type=os.getenv("VOICE_ASR_COMPUTE_TYPE", "int8"),
            asr_beam_size=int(os.getenv("VOICE_ASR_BEAM_SIZE", "1")),
            asr_cpu_threads=int(os.getenv("VOICE_ASR_CPU_THREADS", "8")),
            asr_tail_ms=int(os.getenv("VOICE_ASR_TAIL_MS", "96")),
            seed=int(os.getenv("VOICE_SEED", "42")),
            vad_kind=os.getenv("VOICE_VAD", "silero"),
            vad_threshold=float(os.getenv("VOICE_VAD_THRESHOLD", "0.5")),
            vad_min_silence_ms=int(os.getenv("VOICE_VAD_MIN_SILENCE_MS", "320")),
            vad_min_speech_ms=int(os.getenv("VOICE_VAD_MIN_SPEECH_MS", "96")),
            partial_interval_ms=int(os.getenv("VOICE_PARTIAL_INTERVAL_MS", "320")),
            tts_voice=os.getenv("VOICE_TTS_VOICE", "en_US-lessac-medium"),
            tts_default_language=os.getenv("VOICE_TTS_DEFAULT_LANGUAGE", "en"),
            model_dir=Path(os.getenv("VOICE_MODEL_DIR", str(MODEL_DIR))),
            budget_ms=int(os.getenv("VOICE_BUDGET_MS", "1000")),
            agent_backend=os.getenv("VOICE_AGENT_BACKEND", "llm"),
            max_turn_seconds=float(os.getenv("VOICE_MAX_TURN_SECONDS", "30")),
            partials_enabled=_bool("VOICE_PARTIALS", True),
            pacing=_bool("VOICE_PACING", False),
            asr_initial_prompt=os.getenv("VOICE_ASR_PROMPT") or None,
            voice_llm_provider=os.getenv("VOICE_LLM_PROVIDER") or None,
            voice_llm_base_url=os.getenv("VOICE_LLM_BASE_URL") or None,
            voice_llm_model=os.getenv("VOICE_LLM_MODEL") or None,
        )

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)
