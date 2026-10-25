"""End-to-end simulated caller eval over a real localhost WebSocket.

Pipeline (all real, CPU-only): Silero VAD -> faster-whisper int8 ASR ->
LLM-planned support agent + SQLite ticketing -> Piper TTS -> WebSocket PCM.

The caller's lines are produced by a local LLM (scripted goals from YAML are the
fallback when the LLM omits a required entity). Recordings, ASR round-trips,
tool trajectories, latency, barge-in and hallucination judgments are written to
``evals/voice_results/simulated_caller.json``.

Usage:
  python -m evals.simulated_caller --caller llm --judge llm
  python -m evals.simulated_caller --caller scripted --judge rule
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
import websockets
import yaml

from core.config import settings as app_settings
from evals.trajectory import score_trajectory
from integrations.ticketing import TicketingSystem
from voice.agent_bridge import LLMSupportBackend
from voice.asr import FasterWhisperASR, resample_linear
from voice.config import VoiceSettings
from voice.server import create_app
from voice.tts import PiperTTS
from voice.vad import make_vad

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "evals" / "simulated_caller_scenarios.yaml"
OUT = ROOT / "evals" / "voice_results" / "simulated_caller.json"
PORT = 8765
SAMPLE_RATE = 16000
FRAME_BYTES = 1024  # 32 ms PCM16 @ 16 kHz
HOP_KEYS = ("asr_ms", "llm_ms", "tts_ms", "network_ms", "e2e_ms")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _pcm16(array: np.ndarray) -> bytes:
    return np.clip(array, -32768, 32767).astype(np.int16).tobytes()


def to_16k(pcm: bytes, rate: int) -> bytes:
    if rate == SAMPLE_RATE:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return _pcm16(resample_linear(samples, rate, SAMPLE_RATE) * 32768.0)


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9$' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if r == h else 1))
            )
        prev = cur
    return round(prev[-1] / len(ref), 4)


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(values, q)), 2)


def hop_rows(metrics: list[dict[str, Any]]) -> list[dict[str, float | None]]:
    return [{key: m.get(key) for key in HOP_KEYS} for m in metrics]


def hop_stats(hops: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key in HOP_KEYS:
        values = [float(h[key]) for h in hops if h.get(key) is not None]
        out[key] = {
            "p50_ms": pct(values, 50),
            "p95_ms": pct(values, 95),
            "n": len(values),
        }
    return out


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# caller
# ---------------------------------------------------------------------------
class Caller:
    def __init__(self, mode: str, model: str) -> None:
        self.mode = mode
        self.model = model

    def line(
        self, goal: str, script: dict[str, Any], agent_said: str, previous: list[str]
    ) -> tuple[str, str]:
        if self.mode == "scripted":
            return script["text"], "scripted"
        from core.llm import chat

        prompt = (
            f"Goal: {goal}\n"
            f"Draft caller line: {script['text']}\n"
            f"Required exact names/numbers: {script.get('must_keep', [])}\n"
            f"Agent just said: {agent_said or '(nothing yet)'}\n"
            f"Earlier caller lines: {previous}\n\n"
            "Write the caller's next line as ONE short natural spoken sentence. "
            "Keep every required name and number exactly. No stage directions, no quotes."
        )
        try:
            text = chat(
                "You are a polite customer on a phone call. Reply with one spoken sentence only.",
                prompt,
                temperature=0.0,
                think=False,
                model=self.model,
            ).strip().strip('"')
        except Exception:
            return script["text"], "fallback_error"
        low = text.lower()
        keep = [str(k).lower() for k in script.get("must_keep", [])]
        if not text or len(text) > 220 or any(k not in low for k in keep):
            return script["text"], "fallback_validation"
        draft = script["text"].lower().strip()
        if draft.startswith(("no,", "correction")):
            if not re.search(r"\b(no|wrong|correction|said|mean|instead)\b", low):
                return script["text"], "fallback_validation"
        if draft.startswith(("yes", "sí", "si", "go ahead", "correct")):
            if not re.search(r"\b(yes|yeah|confirm|go ahead|sí|si|correct)\b", low):
                return script["text"], "fallback_validation"
            if "actually" in low or "wait" in low:
                return script["text"], "fallback_validation"
            if re.search(r"\b(like|want|need|request)\b", low):
                return script["text"], "fallback_validation"
        return text, "llm"


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------
def start_server(
    ticketing: TicketingSystem,
    asr: FasterWhisperASR,
    tts: PiperTTS,
    agent_model: str | None = None,
) -> tuple[Any, threading.Thread]:
    backend = LLMSupportBackend(ticketing, model=agent_model)

    def factory(s: VoiceSettings, tenant_id: str) -> dict[str, Any]:
        return {
            "vad": make_vad(s.vad_kind, s.model_dir, s.vad_threshold),
            "asr": asr,
            "tts": tts,
            "agent": backend,
        }

    app = create_app(factory)
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("voice server did not start")
        time.sleep(0.05)
    return server, thread


# ---------------------------------------------------------------------------
# websocket client
# ---------------------------------------------------------------------------
class WsState:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.audio: list[bytes] = []
        self.metrics: list[dict[str, Any]] = []
        self.agent_texts: list[dict[str, Any]] = []
        self.transcripts: list[dict[str, Any]] = []
        self.barge_events = 0
        self.first_audio = asyncio.Event()
        self.changed = asyncio.Event()

    def consume(self, raw: Any) -> None:
        if isinstance(raw, bytes):
            self.audio.append(raw)
            self.first_audio.set()
            return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        self.messages.append(msg)
        kind = msg.get("type")
        if kind == "metrics":
            self.metrics.append(msg)
        elif kind == "agent_text":
            self.agent_texts.append(msg)
        elif kind == "transcript":
            self.transcripts.append(msg)
        elif kind == "barge_in":
            self.barge_events += 1
        self.changed.set()


async def reader(ws: Any, state: WsState) -> None:
    try:
        async for raw in ws:
            state.consume(raw)
    except Exception:
        return


async def wait_for(predicate: Any, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while not predicate():
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for voice turn")
        await asyncio.sleep(0.02)


async def send_pcm(ws: Any, pcm: bytes) -> None:
    for offset in range(0, len(pcm), FRAME_BYTES):
        await ws.send(pcm[offset : offset + FRAME_BYTES])


async def send_silence(ws: Any, frames: int = 20) -> None:
    for _ in range(frames):
        await ws.send(b"\x00" * FRAME_BYTES)


async def run_scenario(scn: dict[str, Any], caller: Caller, ticketing: TicketingSystem,
                       tts: PiperTTS, agent_asr: FasterWhisperASR, judge: str) -> dict[str, Any]:
    if scn.get("seed_ticket"):
        seed = scn["seed_ticket"]
        if ticketing.get_ticket(scn["tenant"], seed["ticket_id"]) is None:
            ticketing.create_ticket(
                scn["tenant"], "cust-001", seed["subject"],
                ticket_id=seed["ticket_id"],
            )
        ticket = ticketing.get_ticket(scn["tenant"], seed["ticket_id"])
        if ticket and seed.get("status"):
            ticketing.update_status(scn["tenant"], seed["ticket_id"], seed["status"])

    from auth.tenant import issue_tenant_token

    token = issue_tenant_token("eval-caller", "user", tenant_id=scn["tenant"])
    state = WsState()
    async with websockets.connect(
        f"ws://127.0.0.1:{PORT}/ws/voice?token={token}"
    ) as ws:
        read_task = asyncio.create_task(reader(ws, state))
        await ws.send(json.dumps({"type": "start", "session_id": f"eval-{scn['id']}", "language": "en"}))
        await wait_for(lambda: any(m.get("type") == "ready" for m in state.messages), 10)

        turns = scn["turns"]
        barge_at = scn.get("barge_in_after_turn")
        audio_cursor = 0
        turn_audio: list[list[bytes]] = []

        # Generate all caller lines before the call starts. This keeps barge-in
        # realistic: the interrupting audio is ready the instant agent audio begins.
        caller_turns: list[dict[str, str]] = []
        for turn in turns:
            text, source = await asyncio.to_thread(
                caller.line,
                scn["goal"],
                turn,
                "The agent is helping with the previous turn." if caller_turns else "",
                [t["text"] for t in caller_turns],
            )
            caller_turns.append({"text": text, "source": source})
        for idx, turn in enumerate(turns):
            text = caller_turns[idx]["text"]
            pcm, rate = tts.synthesize_pcm(text)
            pcm16 = to_16k(pcm, rate)
            if barge_at is not None and idx == barge_at + 1:
                await wait_for(lambda: state.first_audio.is_set(), 120)
            await send_pcm(ws, pcm16)
            await send_silence(ws)
            if barge_at is not None and idx == barge_at:
                # Do not wait for the interrupted turn's metrics; keep listening.
                continue
            if barge_at is not None and idx == barge_at + 1:
                target = 1
            elif barge_at is not None:
                target = 2
            else:
                target = idx + 1
            def _completed(target: int = target) -> bool:
                return sum(1 for m in state.metrics if not m.get("cancelled")) >= target

            await wait_for(_completed, 180)
            new_audio = state.audio[audio_cursor:]
            audio_cursor = len(state.audio)
            turn_audio.append(new_audio)
        await wait_for(
            lambda: sum(1 for m in state.metrics if not m.get("cancelled"))
            >= (len(turns) - 1 if barge_at is not None else len(turns)),
            180,
        )
        read_task.cancel()

    all_tool_calls: list[dict[str, Any]] = []
    for msg in state.agent_texts:
        all_tool_calls.extend(msg.get("tool_calls", []) or [])

    # Transcribe the final response audio end-to-end.
    final_audio = b"".join(turn_audio[-1]) if turn_audio else b""
    response_asr = agent_asr.transcribe_pcm(final_audio, sample_rate=SAMPLE_RATE) if final_audio else None
    response_text = response_asr.text if response_asr else ""

    tool_score = score_trajectory(scn.get("expected_tools", []), all_tool_calls)
    phrases = [p.lower() for p in scn.get("success_phrases", [])]
    phrase_ok = all(p in response_text.lower() for p in phrases) if phrases else True
    expected_tools = scn.get("expected_tools", [])
    if not expected_tools:
        tools_ok = not all_tool_calls
    else:
        tools_ok = (
            tool_score["order_correctness"] == 1.0
            and tool_score["argument_validity"] == 1.0
        )
    refusal_ok = True
    if scn.get("expect_refusal"):
        refusal_ok = any(m.get("refused") for m in state.agent_texts)

    confirmation_ok = True
    if scn.get("requires_confirmation"):
        confirming = [
            m for m in state.agent_texts
            if not m.get("tool_calls")
            and ("confirm" in m["text"].lower() or "go ahead" in m["text"].lower())
        ]
        write_tools = {"create_ticket", "issue_refund", "update_ticket"}
        executing = [
            m for m in state.agent_texts
            if any(c.get("tool") in write_tools for c in m.get("tool_calls", []))
        ]
        confirmation_ok = bool(confirming and executing) and state.agent_texts.index(
            confirming[0]
        ) < state.agent_texts.index(executing[0])

    code_switch_ok = True
    if scn.get("expect_code_switch"):
        code_switch_ok = any(m.get("code_switched") for m in state.messages if m.get("type") == "language")

    task_success = bool(tools_ok and phrase_ok and refusal_ok and confirmation_ok and code_switch_ok)

    caller_wer = []
    for i, t in enumerate(caller_turns):
        if i < len(state.transcripts):
            caller_wer.append(wer(t["text"], state.transcripts[i]["text"]))

    hallucination = judge_hallucination(
        judge, scn, state.agent_texts, all_tool_calls, response_text
    )

    e2e = [m["e2e_ms"] for m in state.metrics if m.get("e2e_ms") is not None]

    return {
        "id": scn["id"],
        "title": scn["title"],
        "goal": scn["goal"],
        "caller_turns": caller_turns,
        "caller_transcripts": [t.get("text") for t in state.transcripts],
        "caller_wer": round(statistics.mean(caller_wer), 4) if caller_wer else None,
        "agent_turns": [
            {"text": m["text"], "backend": m.get("backend"), "tool_calls": m.get("tool_calls", [])}
            for m in state.agent_texts
        ],
        "response_asr": response_text,
        "response_asr_wer": (
            round(wer(state.agent_texts[-1]["text"], response_text), 4)
            if state.agent_texts and response_text
            else None
        ),
        "tool_calls": all_tool_calls,
        "expected_tools": scn.get("expected_tools", []),
        "tool_score": tool_score,
        "task_success": task_success,
        "checks": {
            "tools_ok": tools_ok,
            "phrase_ok": phrase_ok,
            "refusal_ok": refusal_ok,
            "confirmation_ok": confirmation_ok,
            "code_switch_ok": code_switch_ok,
        },
        "barge_in": {
            "expected": barge_at is not None,
            "events": state.barge_events,
            "frames_after_barge_in": next(
                (m.get("frames_after_barge_in") for m in state.metrics if m.get("cancelled")), None
            ),
        },
        "hallucination": hallucination,
        "latency": {
            "turns": e2e,
            "hops": hop_rows(state.metrics),
            "p50_ms": pct(e2e, 50),
            "p95_ms": pct(e2e, 95),
            "per_hop": hop_stats(hop_rows(state.metrics)),
            "under_budget": sum(1 for m in state.metrics if m.get("budget_ok")),
            "turn_count": len(state.metrics),
        },
    }


_JUDGE_PROMPT = """Judge whether the agent's FINAL reply invented facts. PASS only
if every factual claim is entailed by the EVIDENCE below. FAIL if it invents a
ticket ID, amount, status or action not in the evidence. Ticket IDs present in a
create_ticket result or in later tool args are grounded. Earlier turns may be
confirmations before a caller correction; judge only the final reply against the
latest caller statements. JSON only.

EVIDENCE:
{evidence}

FINAL agent reply:
{replies}

Reply: {{"label":"pass"|"fail","reason":"short"}}"""


def judge_hallucination(
    mode: str,
    scn: dict[str, Any],
    agent_texts: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    response_asr: str,
) -> dict[str, Any]:
    replies = [m["text"] for m in agent_texts][-1:]
    if mode == "rule":
        return {"label": "pass", "reason": "rule mode (not judged)", "mode": "rule"}
    from core.config import settings
    from core.llm import chat

    evidence = [f"seed: {json.dumps(scn.get('seed_ticket', {}))}"]
    evidence += [f"caller: {t.get('text', '')}" for t in scn.get("turns", [])]
    for call in tool_calls:
        evidence.append(
            f"tool {call.get('tool')}: args={json.dumps(call.get('args', {}))} "
            f"result={json.dumps(call.get('result', {}))}"
        )
    prompt = _JUDGE_PROMPT.format(
        evidence="\n".join(evidence),
        replies=json.dumps(replies),
    )
    old_model = settings.lmstudio_chat_model
    settings.lmstudio_chat_model = (
        "gemma-4-12b-agentic-fable5-composer2.5-v2-3.5x-tau2"
    )
    try:
        raw = chat(
            "You are a strict hallucination judge. JSON only.",
            prompt,
            temperature=0.0,
            think=False,
        )
        out = _extract_json(raw)
    except Exception as exc:
        return {"label": "fail", "reason": f"judge_error: {exc}", "mode": "llm"}
    finally:
        settings.lmstudio_chat_model = old_model
    label = str(out.get("label", "")).lower()
    return {
        "label": "pass" if label == "pass" else "fail",
        "reason": str(out.get("reason", ""))[:240],
        "mode": "llm",
    }


# ---------------------------------------------------------------------------
def setup_agent_model() -> str:
    from agents.llm_support import DEFAULT_MODEL

    return DEFAULT_MODEL


def warm_agent(model: str) -> None:
    from core.llm import chat_json

    try:
        chat_json(
            "Return JSON.",
            'Return {"ok": true}',
            temperature=0.0,
            think=False,
            model=model,
            use_cache=False,
        )
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caller", choices=["llm", "scripted"], default="llm")
    parser.add_argument("--judge", choices=["llm", "rule"], default="llm")
    parser.add_argument("--only", default=None, help="run one scenario id")
    parser.add_argument("--out", default=str(OUT), help="result JSON path")
    args = parser.parse_args()

    scenarios = yaml.safe_load(SCENARIOS.read_text())["scenarios"]
    if args.only:
        scenarios = [s for s in scenarios if s["id"] == args.only]

    # Stream TTS in real time so the barge-in scenario exercises a genuine
    # interruption instead of a burst that finishes before the caller speaks.
    os.environ.setdefault("VOICE_PACING", "1")
    settings = VoiceSettings.from_env()
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    ticketing = TicketingSystem()
    prompt = (
        "Customer support call. Hola, sí, por favor, gracias. Order ID B123. "
        "Ticket ID T-123456. Refund, reembolso, complaint, booking, billing."
    )
    asr = FasterWhisperASR(
        settings.asr_model, device="cpu", compute_type="int8",
        beam_size=settings.asr_beam_size, initial_prompt=prompt, seed=settings.seed,
        cpu_threads=settings.asr_cpu_threads,
    )
    agent_asr = FasterWhisperASR(
        settings.asr_model, device="cpu", compute_type="int8",
        beam_size=settings.asr_beam_size, initial_prompt=prompt, seed=settings.seed,
        cpu_threads=settings.asr_cpu_threads,
    )
    tts = PiperTTS(settings.tts_voice, model_dir=settings.model_dir)
    # Warm models so latency is steady-state.
    asr.transcribe_pcm(b"\x00\x00" * SAMPLE_RATE, sample_rate=SAMPLE_RATE)
    tts.synthesize_pcm("Warm up.")

    agent_model = setup_agent_model()
    warm_agent(agent_model)
    server, thread = start_server(ticketing, asr, tts, agent_model)
    caller = Caller(args.caller, agent_model)
    results = []
    try:
        for scn in scenarios:
            print(f"== {scn['id']}: {scn['title']}", flush=True)
            started = time.time()
            result = asyncio.run(
                run_scenario(scn, caller, ticketing, tts, agent_asr, args.judge)
            )
            result["wall_seconds"] = round(time.time() - started, 1)
            results.append(result)
            print(
                f"   success={result['task_success']} tools={result['tool_score']['tool_choice_accuracy']:.2f}"
                f" args={result['tool_score']['argument_validity']:.2f}"
                f" e2e_p50={result['latency']['p50_ms']}ms"
                f" halluc={result['hallucination']['label']}",
                flush=True,
            )
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    e2e = [v for r in results for v in r["latency"]["turns"]]
    all_hops = [h for r in results for h in r["latency"]["hops"]]
    per_hop = hop_stats(all_hops)
    summary = {
        "n_scenarios": len(results),
        "task_success": sum(1 for r in results if r["task_success"]),
        "task_success_rate": round(sum(1 for r in results if r["task_success"]) / len(results), 4),
        "tool_choice_accuracy": round(
            statistics.mean(r["tool_score"]["tool_choice_accuracy"] for r in results), 4
        ),
        "argument_validity": round(
            statistics.mean(r["tool_score"]["argument_validity"] for r in results), 4
        ),
        "order_correctness": round(
            statistics.mean(r["tool_score"]["order_correctness"] for r in results), 4
        ),
        "hallucination_pass_rate": round(
            sum(1 for r in results if r["hallucination"]["label"] == "pass") / len(results), 4
        ),
        "e2e_p50_ms": pct(e2e, 50),
        "e2e_p95_ms": pct(e2e, 95),
        "asr_p50_ms": per_hop["asr_ms"]["p50_ms"],
        "asr_p95_ms": per_hop["asr_ms"]["p95_ms"],
        "llm_p50_ms": per_hop["llm_ms"]["p50_ms"],
        "llm_p95_ms": per_hop["llm_ms"]["p95_ms"],
        "tts_p50_ms": per_hop["tts_ms"]["p50_ms"],
        "tts_p95_ms": per_hop["tts_ms"]["p95_ms"],
        "network_p50_ms": per_hop["network_ms"]["p50_ms"],
        "network_p95_ms": per_hop["network_ms"]["p95_ms"],
        "per_hop": per_hop,
        "e2e_under_budget": sum(1 for r in results for m in r["latency"]["turns"] if m <= 1000),
        "e2e_turns": len(e2e),
        "caller_wer_mean": round(
            statistics.mean(r["caller_wer"] for r in results if r["caller_wer"] is not None), 4
        ),
        "barge_in_ok": all(
            r["barge_in"]["events"] >= 1
            and (r["barge_in"]["frames_after_barge_in"] or 0) <= 1
            for r in results
            if r["barge_in"]["expected"]
        ),
    }
    payload = {
        "run": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": "local macOS Apple Silicon CPU (no GPU)",
            "asr": f"faster-whisper {settings.asr_model} {settings.asr_compute_type}",
            "tts": f"piper {settings.tts_voice}",
            "vad": "silero v5 (onnx, CPU)",
            "agent_backend": "llm",
            "agent_model": f"{agent_model} via LM Studio",
            "caller_model": "ornith-1.5-9b via LM Studio" if args.caller == "llm" else "scripted",
            "judge": (
                "llm (gemma-4-12b-agentic via LM Studio)"
                if args.judge == "llm"
                else "rule"
            ),
            "transport": f"websockets localhost:{PORT}",
            "pacing": settings.pacing,
            "seed": {
                "voice": settings.seed,
                "llm": app_settings.llm_seed,
                "asr_decoding": {
                    "beam_size": settings.asr_beam_size,
                    "temperature": 0.0,
                    "seed_supported": False,
                    "note": (
                        "faster-whisper transcribe() has no seed argument; "
                        "process RNG is seeded and temperature=0.0 disables sampling"
                    ),
                },
                "llm_note": (
                    "seed is sent to LM Studio / OpenAI-compatible / Gemini; "
                    "LM Studio accepts it but does not echo it, so effectiveness "
                    "is not independently verifiable"
                ),
            },
        },
        "summary": summary,
        "scenarios": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
