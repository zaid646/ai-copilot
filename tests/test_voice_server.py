"""WebSocket protocol test with CPU-only fakes."""
from __future__ import annotations

from fastapi.testclient import TestClient

from agents.voice_support import SupportAgent
from voice.agent_bridge import SupportBackend
from voice.asr import Transcript
from voice.server import create_app


class FakeVAD:
    def is_speech(self, pcm16: bytes) -> bool:
        return bool(pcm16) and pcm16[0] == 1

    def reset(self) -> None:
        return None


class FakeASR:
    def transcribe_pcm(self, pcm16: bytes, *, sample_rate: int = 16000, language=None, **kwargs):
        return Transcript(text="What is the status of ticket T-ABC123?", language="en")


class FakeTTS:
    async def stream(self, text: str, *, voice=None, sample_rate=16000, cancel=None):
        for _ in range(3):
            if cancel is not None and cancel.is_set():
                return
            yield b"\x00\x00" * 160


def factory(settings, tenant_id):
    ticketing = SupportAgent()
    return {
        "vad": FakeVAD(),
        "asr": FakeASR(),
        "tts": FakeTTS(),
        "agent": SupportBackend(ticketing.ticketing),
    }


def test_websocket_voice_turn_end_to_end():
    app = create_app(factory)
    client = TestClient(app)
    with client.websocket_connect("/ws/voice") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        ws.send_json({"type": "start", "language": "en"})
        assert ws.receive_json()["type"] == "started"

        # 96 ms of speech -> 96 ms of silence (>= min silence)
        ws.send_bytes(bytes([1]) + b"\x00" * 1023)
        ws.send_bytes(bytes([1]) + b"\x00" * 1023)
        ws.send_bytes(bytes([1]) + b"\x00" * 1023)
        for _ in range(11):
            ws.send_bytes(b"\x00" * 1024)

        seen = []
        audio = 0
        for _ in range(20):
            message = ws.receive()
            if message.get("bytes") is not None:
                audio += 1
                continue
            payload = message.get("text")
            if payload is None:
                break
            import json

            item = json.loads(payload)
            seen.append(item["type"])
            if item["type"] == "metrics":
                metrics = item
                break
        assert "transcript" in seen
        assert "agent_text" in seen
        assert "metrics" in seen
        assert audio >= 1
        assert metrics["e2e_ms"] is not None
        assert metrics["budget_ms"] == 1000
