"""Turn segmentation + streaming turn pipeline.

The session is transport-agnostic: the WebSocket server feeds PCM frames and
supplies two async callbacks (JSON control messages and binary audio frames).
That makes barge-in and budget metrics testable without sockets.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from voice.asr import Transcript
from voice.budget import VOICE_BARGEIN_TOTAL, VOICE_TURNS_TOTAL, TurnBudget
from voice.config import VoiceSettings
from voice.languages import route_voice
from voice.speech_style import is_correction

JsonSender = Callable[[dict[str, Any]], Awaitable[None]]
AudioSender = Callable[[bytes], Awaitable[None]]


@dataclass
class TurnResult:
    transcript: Transcript
    agent_text: str
    metrics: dict[str, Any] = field(default_factory=dict)
    barge_in: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    refused: bool = False


class VoiceSession:
    def __init__(
        self,
        *,
        settings: VoiceSettings,
        vad: Any,
        asr: Any,
        tts: Any,
        agent: Any,
        send_json: JsonSender,
        send_audio: AudioSender,
        tenant_id: str = "default",
        session_id: str = "voice-session",
        conversation_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.vad = vad
        self.asr = asr
        self.tts = tts
        self.agent = agent
        self.send_json = send_json
        self.send_audio = send_audio
        self.tenant_id = tenant_id
        self.session_id = session_id
        self.conversation_id = conversation_id or session_id

        self._buf = bytearray()
        self._preroll: deque[bytes] = deque(maxlen=8)
        self._in_speech = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._agent_speaking = False
        self._cancel = asyncio.Event()
        self._turn_lock = asyncio.Lock()
        self._partial_lock = asyncio.Lock()
        self._last_partial_ms = 0
        self._last_partial_text = ""
        self._turn_task: asyncio.Task[Any] | None = None
        self.barge_in_count = 0
        self.frames_after_barge_in = 0
        self.last_result: TurnResult | None = None
        self.language = settings.tts_default_language
        self.voice = settings.tts_voice

    # ------------------------------------------------------------------
    # Frame ingestion / turn detection
    # ------------------------------------------------------------------
    @property
    def utterance_ms(self) -> int:
        return int(len(self._buf) / 2 / self.settings.sample_rate * 1000)

    def feed_audio(self, frame: bytes) -> dict[str, Any]:
        """Consume one PCM16 frame and return turn-control events.

        ``barge_in`` is emitted exactly once when speech starts while the agent
        is still sending audio. The caller must cancel the active TTS stream.
        """
        if not frame:
            return {}
        speech = self.vad.is_speech(frame)
        events: dict[str, Any] = {}
        if speech:
            self._silence_ms = 0
            if not self._in_speech:
                self._in_speech = True
                self._speech_ms = 0
                self._buf.clear()
                for preroll_frame in self._preroll:
                    self._buf.extend(preroll_frame)
                self._preroll.clear()
                events["speech_start"] = True
            self._speech_ms += self.settings.frame_ms
            self._buf.extend(frame)
            if self._agent_speaking and not self._cancel.is_set():
                self._cancel.set()
                self.barge_in_count += 1
                VOICE_BARGEIN_TOTAL.inc()
                events["barge_in"] = True
        elif not self._in_speech:
            self._preroll.append(frame)
        else:
            self._silence_ms += self.settings.frame_ms
            self._buf.extend(frame)
            if self._silence_ms >= self.settings.vad_min_silence_ms:
                # VAD waits for vad_min_silence_ms to confirm the endpoint, but
                # Whisper does not need all of that silence. Keeping a short tail
                # avoids transcribing dead air and lowers ASR latency.
                tail_ms = max(0, min(self.settings.asr_tail_ms, self._silence_ms))
                tail_bytes = int(self.settings.sample_rate * tail_ms / 1000) * 2
                pcm = bytes(self._buf[:-tail_bytes] if tail_bytes else self._buf)
                self._reset_utterance()
                events["end_of_speech"] = pcm
        return events

    def take_utterance(self) -> bytes:
        """Force-flush the current speech buffer (push-to-talk / tests)."""
        if not self._buf:
            return b""
        pcm = bytes(self._buf)
        self._reset_utterance()
        return pcm

    def _reset_utterance(self) -> None:
        self._buf.clear()
        self._preroll.clear()
        self._in_speech = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._last_partial_text = ""
        self._last_partial_ms = self.utterance_ms

    def partial_due(self) -> bool:
        if not self.settings.partials_enabled or not self._in_speech:
            return False
        return (self.utterance_ms - self._last_partial_ms) >= self.settings.partial_interval_ms

    async def emit_partial(self) -> None:
        async with self._partial_lock:
            if not self._in_speech or not self._buf:
                return
            self._last_partial_ms = self.utterance_ms
            pcm = bytes(self._buf)
            language = None if self.language in ("auto", "") else self.language
            try:
                transcript = await asyncio.to_thread(
                    self.asr.transcribe_pcm, pcm, sample_rate=self.settings.sample_rate, language=language
                )
            except Exception:
                return
            text = transcript.text.strip()
            if text and text != self._last_partial_text:
                self._last_partial_text = text
                await self.send_json({"type": "partial", "text": text})

    # ------------------------------------------------------------------
    # Turn processing
    # ------------------------------------------------------------------
    def start_turn(self, pcm: bytes, *, language: str | None = None) -> asyncio.Task[Any]:
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        self._turn_task = asyncio.create_task(self.process_utterance(pcm, language=language))
        return self._turn_task

    async def process_utterance(
        self, pcm: bytes, *, language: str | None = None
    ) -> TurnResult | None:
        async with self._turn_lock:
            budget = TurnBudget(budget_ms=self.settings.budget_ms)
            budget.end_of_speech_at = time.perf_counter()
            lang = language or (self.language if self.language not in ("auto", "") else None)
            try:
                transcript = await asyncio.to_thread(
                    self.asr.transcribe_pcm, pcm, sample_rate=self.settings.sample_rate, language=lang
                )
            except Exception as exc:  # pragma: no cover - model failure path
                await self.send_json({"type": "error", "stage": "asr", "error": str(exc)})
                VOICE_TURNS_TOTAL.labels(outcome="asr_error").inc()
                return None
            budget.mark_asr()
            await self.send_json(
                {
                    "type": "transcript",
                    "text": transcript.text,
                    "language": transcript.language,
                    "language_probability": transcript.language_probability,
                    "asr_ms": budget.asr_ms(),
                }
            )

            decision = route_voice(
                transcript.text,
                transcript.language,
                transcript.language_probability,
                default_voice=self.settings.tts_voice,
            )
            self.language = decision.language
            self.voice = decision.voice
            if decision.code_switched or is_correction(transcript.text):
                await self.send_json(
                    {
                        "type": "language",
                        "language": decision.language,
                        "voice": decision.voice,
                        "code_switched": decision.code_switched,
                        "reason": decision.reason,
                    }
                )

            try:
                reply = await asyncio.to_thread(
                    self.agent.respond,
                    transcript.text,
                    tenant_id=self.tenant_id,
                    session_id=self.conversation_id,
                )
            except Exception as exc:  # pragma: no cover - agent failure path
                await self.send_json({"type": "error", "stage": "agent", "error": str(exc)})
                VOICE_TURNS_TOTAL.labels(outcome="agent_error").inc()
                return None
            budget.mark_llm()
            await self.send_json(
                {
                    "type": "agent_text",
                    "text": reply.text,
                    "backend": getattr(reply, "backend", "unknown"),
                    "tool_calls": getattr(reply, "tool_calls", []),
                    "refused": getattr(reply, "refused", False),
                    "language": decision.language,
                    "voice": decision.voice,
                    "llm_ms": budget.llm_ms(),
                }
            )

            self._agent_speaking = True
            self.frames_after_barge_in = 0
            first = True
            cancelled = False
            self._cancel.clear()
            try:
                async for frame in self.tts.stream(
                    reply.text,
                    voice=decision.voice,
                    sample_rate=self.settings.sample_rate,
                    cancel=self._cancel,
                ):
                    if self._cancel.is_set():
                        cancelled = True
                        break
                    if first:
                        budget.mark_tts_first()
                    await self.send_audio(frame)
                    if self.settings.pacing:
                        await asyncio.sleep(self.tts.frame_ms / 1000.0)
                    if first:
                        budget.mark_first_audio_sent()
                        first = False
                    if self._cancel.is_set():
                        # Count the single frame that may already be in flight.
                        self.frames_after_barge_in += 1
                        cancelled = True
            finally:
                self._agent_speaking = False

            budget.barge_in = self.barge_in_count > 0
            budget.cancelled = cancelled
            metrics = budget.observe()
            metrics.update(
                {
                    "transcript": transcript.text,
                    "agent_text": reply.text,
                    "language": decision.language,
                    "voice": decision.voice,
                    "code_switched": decision.code_switched,
                    "frames_after_barge_in": self.frames_after_barge_in,
                    "tool_calls": getattr(reply, "tool_calls", []),
                    "refused": getattr(reply, "refused", False),
                    "budget_ok": budget.budget_ok(),
                }
            )
            await self.send_json({"type": "audio_end", "text": reply.text})
            await self.send_json({"type": "metrics", **metrics})
            VOICE_TURNS_TOTAL.labels(
                outcome="barge_in" if cancelled else "ok"
            ).inc()
            self.last_result = TurnResult(
                transcript=transcript,
                agent_text=reply.text,
                metrics=metrics,
                barge_in=cancelled,
                tool_calls=getattr(reply, "tool_calls", []),
                refused=getattr(reply, "refused", False),
            )
            return self.last_result

    def cancel(self) -> None:
        self._cancel.set()
        self._agent_speaking = False

    async def close(self) -> None:
        self.cancel()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except (asyncio.CancelledError, Exception):
                pass
