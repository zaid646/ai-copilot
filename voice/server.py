"""FastAPI WebSocket voice endpoint.

Wire protocol
-------------
Client -> server:
  ``{"type":"start","language":"en","session_id":"..."}``
  binary frames: little-endian PCM16 mono at ``VOICE_SAMPLE_RATE`` (16 kHz)
  ``{"type":"end_turn"}``   force end-of-speech (used by tests / push-to-talk)
  ``{"type":"barge_in"}``   explicit barge-in signal

Server -> client:
  ``{"type":"ready", ...}``
  ``{"type":"partial","text":...}``
  ``{"type":"transcript",...}`` ``{"type":"agent_text",...}``
  binary frames: PCM16 audio at 16 kHz
  ``{"type":"audio_end"}`` ``{"type":"metrics",...}`` ``{"type":"error",...}``

Authentication is optional in local mode. Pass ``?token=<JWT>`` to bind the
session to a tenant; an invalid token is rejected with close code 4401.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from auth.tenant import DEFAULT_TENANT, tenant_from_token
from voice.agent_bridge import build_backend
from voice.asr import FasterWhisperASR
from voice.config import VoiceSettings
from voice.pipeline import VoiceSession
from voice.tts import PiperTTS
from voice.vad import make_vad

ComponentFactory = Callable[[VoiceSettings, str], dict[str, Any]]


def _default_factory(settings: VoiceSettings, tenant_id: str) -> dict[str, Any]:
    return {
        "vad": make_vad(settings.vad_kind, settings.model_dir, settings.vad_threshold),
        "asr": FasterWhisperASR(
            settings.asr_model,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
            beam_size=settings.asr_beam_size,
            cpu_threads=settings.asr_cpu_threads,
        ),
        "tts": PiperTTS(settings.tts_voice, model_dir=settings.model_dir),
        "agent": build_backend(settings.agent_backend),
    }


_BACKENDS: dict[tuple[str, str], Any] = {}


def _cached_backend(settings: VoiceSettings, tenant_id: str) -> Any:
    key = (settings.agent_backend, tenant_id)
    if key not in _BACKENDS:
        _BACKENDS[key] = build_backend(settings.agent_backend)
    return _BACKENDS[key]


def create_app(component_factory: ComponentFactory | None = None) -> FastAPI:
    app = FastAPI(title="AI Copilot Voice", version="0.1.0")
    app.state.voice_factory = component_factory

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "ai-copilot-voice"}

    @app.websocket("/ws/voice")
    async def voice_ws(ws: WebSocket) -> None:
        token = ws.query_params.get("token")
        if token:
            try:
                scope = tenant_from_token(token)
                tenant_id = scope.tenant_id
            except Exception:
                await ws.close(code=4401)
                return
        else:
            tenant_id = DEFAULT_TENANT

        settings = VoiceSettings.from_env()
        factory = ws.app.state.voice_factory or (
            lambda s, t: {
                "vad": make_vad(s.vad_kind, s.model_dir, s.vad_threshold),
                "asr": FasterWhisperASR(
                    s.asr_model,
                    device=s.asr_device,
                    compute_type=s.asr_compute_type,
                    beam_size=s.asr_beam_size,
                    cpu_threads=s.asr_cpu_threads,
                ),
                "tts": PiperTTS(s.tts_voice, model_dir=s.model_dir),
                "agent": _cached_backend(s, t),
            }
        )
        components = factory(settings, tenant_id)
        await ws.accept()

        session_id = f"ws-{id(ws)}"
        session = VoiceSession(
            settings=settings,
            vad=components["vad"],
            asr=components["asr"],
            tts=components["tts"],
            agent=components["agent"],
            send_json=ws.send_json,
            send_audio=ws.send_bytes,
            tenant_id=tenant_id,
            session_id=session_id,
            conversation_id=session_id,
        )
        partial_task: asyncio.Task[Any] | None = None
        await ws.send_json(
            {
                "type": "ready",
                "sample_rate": settings.sample_rate,
                "frame_ms": settings.frame_ms,
                "budget_ms": settings.budget_ms,
                "tenant_id": tenant_id,
            }
        )
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    events = session.feed_audio(message["bytes"])
                    if events.get("barge_in"):
                        session.cancel()
                        await ws.send_json({"type": "barge_in"})
                    if events.get("end_of_speech"):
                        session.start_turn(events["end_of_speech"])
                    elif session.partial_due():
                        if partial_task is None or partial_task.done():
                            partial_task = asyncio.create_task(session.emit_partial())
                    continue
                raw = message.get("text")
                if raw is None:
                    continue
                try:
                    command = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "error": "invalid_json"})
                    continue
                kind = command.get("type")
                if kind == "start":
                    session.language = command.get("language") or settings.tts_default_language
                    session.session_id = command.get("session_id") or session_id
                    session.conversation_id = session.session_id
                    await ws.send_json(
                        {
                            "type": "started",
                            "session_id": session.session_id,
                            "language": session.language,
                        }
                    )
                elif kind == "end_turn":
                    pcm = session.take_utterance()
                    if pcm:
                        session.start_turn(pcm, language=session.language)
                    else:
                        await ws.send_json({"type": "error", "error": "empty_utterance"})
                elif kind == "barge_in":
                    session.cancel()
                    await ws.send_json({"type": "barge_in"})
                elif kind == "ping":
                    await ws.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            session.cancel()
            if partial_task and not partial_task.done():
                partial_task.cancel()
            await session.close()

    @app.get("/metrics")
    async def metrics() -> Any:
        from observability.metrics import metrics_response

        return metrics_response()

    return app


app = create_app()
