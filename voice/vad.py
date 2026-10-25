"""Voice activity detection (CPU).

``SileroVAD`` runs the upstream 2 MB ONNX model through onnxruntime. The model
is small enough to run on every 512-sample frame without a GPU. ``EnergyVAD`` is
a deterministic fallback used when the ONNX asset is not present (for example
in unit tests or a fresh clone before ``make voice-models``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

FRAME_SAMPLES_16K = 512  # 32 ms at 16 kHz, the model's expected window


class VAD(Protocol):
    def is_speech(self, pcm16: bytes) -> bool: ...

    def reset(self) -> None: ...


class EnergyVAD:
    """Deterministic RMS gate. Useful for tests and as a last-resort fallback."""

    def __init__(self, threshold: float = 0.012) -> None:
        self.threshold = threshold

    def is_speech(self, pcm16: bytes) -> bool:
        if not pcm16:
            return False
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return False
        return float(np.sqrt(np.mean(samples * samples))) >= self.threshold

    def reset(self) -> None:
        return None


class SileroVAD:
    """Silero VAD v5 ONNX wrapper."""

    def __init__(self, model_path: str | Path, threshold: float = 0.5, sample_rate: int = 16000):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        self.threshold = threshold
        self.sample_rate = sample_rate
        if not self.model_path.exists():
            raise FileNotFoundError(f"Silero VAD ONNX not found: {self.model_path}")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)

    def is_speech(self, pcm16: bytes) -> bool:
        if not pcm16:
            return False
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size < FRAME_SAMPLES_16K:
            samples = np.pad(samples, (0, FRAME_SAMPLES_16K - samples.size))
        elif samples.size > FRAME_SAMPLES_16K:
            samples = samples[:FRAME_SAMPLES_16K]
        # Silero v5 consumes 64 samples of left context + the 512-sample window.
        model_input = np.concatenate([self._context, samples]).reshape(1, -1)
        out, state = self._session.run(
            None,
            {
                "input": model_input,
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        self._state = state
        self._context = samples[-64:].copy()
        return float(np.asarray(out).reshape(-1)[0]) >= self.threshold


def make_vad(kind: str, model_dir: Path, threshold: float = 0.5) -> VAD:
    """Build a VAD, falling back to the deterministic energy gate."""
    if kind == "energy":
        return EnergyVAD()
    try:
        return SileroVAD(model_dir / "silero_vad.onnx", threshold=threshold)
    except Exception:
        return EnergyVAD()
