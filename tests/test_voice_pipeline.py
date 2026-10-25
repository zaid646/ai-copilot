"""CPU-only voice pipeline tests (ASR/TTS/VAD mocked)."""
from __future__ import annotations

import asyncio
from typing import Any

from voice.agent_bridge import AgentResponse
from voice.asr import Transcript
from voice.budget import VOICE_ASR_MS, VOICE_E2E_MS, metric_count
from voice.config import VoiceSettings
from voice.languages import is_code_switched, route_voice
from voice.pipeline import VoiceSession
from voice.speech_style import inject, is_correction
from voice.tts import PiperTTS


class ScriptedVAD:
    def is_speech(self, pcm16: bytes) -> bool:
        return bool(pcm16) and pcm16[0] == 1

    def reset(self) -> None:
        return None


class FakeASR:
    def __init__(self, text: str = "I need a refund for order A123 for 75 dollars") -> None:
        self.text = text
        self.calls = 0

    def transcribe_pcm(self, pcm16: bytes, *, sample_rate: int = 16000, language=None, **_: Any):
        self.calls += 1
        return Transcript(
            text=self.text, language=language or "en", language_probability=0.99, duration_s=1.0
        )


class FakeTTS:
    def __init__(self, frames: int = 6, barge_after: int | None = None, session_getter=None) -> None:
        self.frames = frames
        self.barge_after = barge_after
        self.session_getter = session_getter
        self.sent = 0

    async def stream(self, text: str, *, voice=None, sample_rate=16000, cancel=None):
        for i in range(self.frames):
            if self.barge_after is not None and i == self.barge_after:
                session = self.session_getter()
                session.feed_audio(bytes([1]) + b"\x00" * 1023)
            yield b"\x00\x01" * 160


class FakeAgent:
    def __init__(self) -> None:
        self.calls = 0

    def respond(self, text: str, *, tenant_id: str, session_id: str) -> AgentResponse:
        self.calls += 1
        return AgentResponse(text="I heard order A123. Should I refund seventy five dollars?")


def speech_frame() -> bytes:
    # 32 ms of PCM16 at 16 kHz = 512 samples = 1024 bytes
    return bytes([1]) + b"\x00" * 1023


def silence_frame() -> bytes:
    return b"\x00" * 1024


def make_session(*, tts=None, asr=None):
    sent_json: list[dict[str, Any]] = []
    sent_audio: list[bytes] = []

    async def send_json(payload: dict[str, Any]) -> None:
        sent_json.append(payload)

    async def send_audio(payload: bytes) -> None:
        sent_audio.append(payload)

    session: VoiceSession

    def get_session() -> VoiceSession:
        return session

    if tts is None:
        tts = FakeTTS()
    settings = VoiceSettings(frame_ms=32, vad_min_silence_ms=64, partial_interval_ms=64)
    session = VoiceSession(
        settings=settings,
        vad=ScriptedVAD(),
        asr=asr or FakeASR(),
        tts=tts,
        agent=FakeAgent(),
        send_json=send_json,
        send_audio=send_audio,
        tenant_id="tenant-a",
    )
    return session, sent_json, sent_audio


def test_voice_turn_emits_hop_metrics():
    async def run():
        before_asr = metric_count(VOICE_ASR_MS)
        before_e2e = metric_count(VOICE_E2E_MS)
        session, sent_json, sent_audio = make_session()
        session.feed_audio(speech_frame())
        events = {}
        for _ in range(3):
            events = session.feed_audio(silence_frame())
            if "end_of_speech" in events:
                break
        assert "end_of_speech" in events
        result = await session.process_utterance(events["end_of_speech"])
        assert result is not None
        types = [m["type"] for m in sent_json]
        assert "transcript" in types
        assert "agent_text" in types
        assert "metrics" in types
        assert sent_audio
        metrics = next(m for m in sent_json if m["type"] == "metrics")
        assert metrics["asr_ms"] is not None
        assert metrics["llm_ms"] is not None
        assert metrics["tts_ms"] is not None
        assert metrics["e2e_ms"] is not None
        assert metric_count(VOICE_ASR_MS) == before_asr + 1
        assert metric_count(VOICE_E2E_MS) == before_e2e + 1

    asyncio.run(run())


def test_barge_in_stops_tts_within_one_frame():
    async def run():
        holder = {}

        class Holder:
            def __call__(self, session):
                holder["session"] = session

        def session_getter():
            return holder["session"]

        tts = FakeTTS(frames=10, barge_after=3, session_getter=session_getter)
        session, _, sent_audio = make_session(tts=tts)
        # barge_after needs the live session; patch after construction
        holder["session"] = session

        pcm = speech_frame() * 6 + silence_frame() * 3
        result = await session.process_utterance(pcm)
        assert result is not None
        # A speech frame fired while the agent was speaking; at most one frame
        # may be in flight after cancellation.
        assert result.barge_in is True
        assert session.frames_after_barge_in <= 1
        assert len(sent_audio) < 10

    asyncio.run(run())


def test_vad_end_of_speech_after_silence():
    session, _, _ = make_session()
    assert "speech_start" in session.feed_audio(speech_frame())
    assert "end_of_speech" not in session.feed_audio(silence_frame())
    events = session.feed_audio(silence_frame())
    assert "end_of_speech" in events


def test_partial_transcript_is_emitted():
    async def run():
        session, sent_json, _ = make_session()
        for _ in range(3):
            session.feed_audio(speech_frame())
        assert session.partial_due()
        await session.emit_partial()
        assert any(m["type"] == "partial" for m in sent_json)

    asyncio.run(run())


def test_language_routing_and_code_switch_fallback():
    assert route_voice("hola, necesito un reembolso", "es", 0.99).voice == "es_ES-sharvard-medium"
    decision = route_voice("I need a refund por favor", "en", 0.99)
    assert decision.code_switched is True
    assert decision.voice == "en_US-lessac-medium"
    assert is_code_switched("I need a refund por favor", "en")


def test_voice_style_injection_and_correction_detection():
    prompt = inject("You are a support agent.")
    assert "CONFIRM BEFORE ACTION" in prompt
    assert inject(prompt) == prompt
    assert is_correction("No, I said A123, not A132")
    assert not is_correction("I need a refund")


def test_budget_records_failure_over_budget():
    import time

    from voice.budget import TurnBudget

    budget = TurnBudget(budget_ms=1)
    time.sleep(0.005)
    budget.mark_asr()
    budget.mark_llm()
    budget.mark_tts_first()
    budget.mark_first_audio_sent()
    metrics = budget.observe()
    assert metrics["budget_ok"] is False
    assert metrics["e2e_ms"] >= 0


def test_piper_stream_yields_before_synthesizing_the_whole_sentence():
    class FakePiper:
        class Config:
            sample_rate = 16000

        config = Config()

        def synthesize(self, _text):
            yield type("Chunk", (), {"audio_int16_bytes": b"\x01\x00" * 1600})()
            raise RuntimeError("second chunk should not be requested by this test")

    async def run():
        tts = PiperTTS(voice="fake", sample_rate=16000, frame_ms=20)
        tts._voices["fake"] = FakePiper()
        frames = []
        try:
            async for frame in tts.stream("hello", voice="fake", sample_rate=16000):
                frames.append(frame)
        except RuntimeError:
            pass
        assert len(frames) == 5
        assert all(len(frame) == 640 for frame in frames)

    asyncio.run(run())
