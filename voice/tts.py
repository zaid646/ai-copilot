"""Piper TTS adapter (CPU) with frame-level streaming."""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from voice.asr import resample_linear


class PiperTTS:
    """Loads one or more Piper voices and streams 20 ms PCM frames."""

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        model_dir: str | Path = "evals/fixtures/models",
        sample_rate: int = 22050,
        frame_ms: int = 20,
    ) -> None:
        self.voice = voice
        self.model_dir = Path(model_dir)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._voices: dict[str, Any] = {}

    def _load_voice(self, voice: str) -> Any:
        if voice in self._voices:
            return self._voices[voice]
        from piper import PiperVoice

        model_path = self.model_dir / f"{voice}.onnx"
        config_path = self.model_dir / f"{voice}.onnx.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Piper voice not found: {model_path}")
        loaded = PiperVoice.load(model_path, config_path=config_path)
        self._voices[voice] = loaded
        return loaded

    def synthesize_pcm(self, text: str, voice: str | None = None) -> tuple[bytes, int]:
        """Return the full utterance as 16-bit PCM plus its sample rate."""
        piper_voice = self._load_voice(voice or self.voice)
        chunks = [chunk.audio_int16_bytes for chunk in piper_voice.synthesize(text)]
        if not chunks:
            return b"", int(getattr(piper_voice.config, "sample_rate", self.sample_rate))
        rate = int(getattr(piper_voice.config, "sample_rate", self.sample_rate))
        return b"".join(chunks), rate

    def synthesize_wav_bytes(self, text: str, voice: str | None = None) -> tuple[bytes, int]:
        """Return a valid WAV file (16-bit mono) for fixtures/tests."""
        import io
        import wave

        pcm, rate = self.synthesize_pcm(text, voice)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return buf.getvalue(), rate

    async def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        sample_rate: int | None = None,
        cancel: Any | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield fixed-size PCM frames, synthesizing sentence-by-sentence.

        ``cancel`` is any object with ``is_set()`` (e.g. ``asyncio.Event``). It is
        checked before every frame and between sentences, so a barge-in stops the
        stream within one frame (the frame already being sent is the maximum
        overshoot).
        """
        piper_voice = self._load_voice(voice or self.voice)
        native_rate = int(getattr(piper_voice.config, "sample_rate", self.sample_rate))
        target_rate = sample_rate or native_rate
        frame_bytes = max(2, int(target_rate * self.frame_ms / 1000) * 2)
        sentences = [s.strip() for s in _split_sentences(text) if s.strip()] or [text]
        pending = bytearray()
        for sentence in sentences:
            if cancel is not None and cancel.is_set():
                break
            # Yield as soon as the first Piper chunk is available instead of
            # synthesizing the whole sentence first. This removes the dominant
            # first-audio delay on CPU.
            for chunk in piper_voice.synthesize(sentence):
                if cancel is not None and cancel.is_set():
                    return
                samples = (
                    np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                samples = resample_linear(samples, native_rate, target_rate)
                pending.extend(
                    np.clip(samples * 32768.0, -32768, 32767).astype(np.int16).tobytes()
                )
                while len(pending) >= frame_bytes:
                    yield bytes(pending[:frame_bytes])
                    del pending[:frame_bytes]
        if pending:
            yield bytes(pending)


def _split_sentences(text: str) -> list[str]:
    import re

    return re.split(r"(?<=[.!?])\s+", text or "")


class PiperTTSLatencyMixin:
    """Small helper used by evals to record first-audio latency."""

    @staticmethod
    def first_frame_latency_ms(started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000.0, 2)
