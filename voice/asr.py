"""Streaming ASR adapters.

``FasterWhisperASR`` uses CTranslate2 on CPU (int8) by default. It is imported
lazily, so CI can exercise the pipeline with ``FakeASR`` and no model download.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Transcript:
    text: str
    language: str = "en"
    language_probability: float = 1.0
    latency_ms: float = 0.0
    duration_s: float = 0.0


def pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / float(src_rate)
    n_dst = max(1, int(round(duration * dst_rate)))
    src_t = np.arange(samples.size, dtype=np.float64) / float(src_rate)
    dst_t = np.arange(n_dst, dtype=np.float64) / float(dst_rate)
    return np.interp(dst_t, src_t, samples).astype(np.float32)


class FasterWhisperASR:
    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        download_root: str | Path | None = None,
        initial_prompt: str | None = None,
        seed: int = 42,
        cpu_threads: int = 0,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.download_root = str(download_root) if download_root else None
        self.initial_prompt = initial_prompt
        self.seed = seed
        self.cpu_threads = cpu_threads
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
                cpu_threads=self.cpu_threads,
            )
        return self._model

    def transcribe_pcm(
        self,
        pcm16: bytes,
        *,
        sample_rate: int = 16000,
        language: str | None = None,
        beam_size: int | None = None,
        initial_prompt: str | None = None,
        seed: int | None = None,
    ) -> Transcript:
        # faster-whisper 1.x has no ``seed`` argument on ``transcribe``. We seed
        # the process RNG for any sampling and force temperature=0.0 so decoding
        # uses greedy/beam search without random fallback.
        random.seed(self.seed if seed is None else seed)
        np.random.seed(self.seed if seed is None else seed)
        start = time.perf_counter()
        audio = pcm16_to_float32(pcm16)
        audio = resample_linear(audio, sample_rate, 16000)
        model = self._load()
        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=beam_size or self.beam_size,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            word_timestamps=False,
            initial_prompt=initial_prompt or self.initial_prompt,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text and seg.text.strip())
        return Transcript(
            text=text.strip(),
            language=getattr(info, "language", language or "en") or "en",
            language_probability=float(getattr(info, "language_probability", 1.0) or 1.0),
            latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
            duration_s=len(audio) / 16000.0,
        )
