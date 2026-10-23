# Voice evaluation: metrics, seeds and latency

`python -m evals.simulated_caller --caller llm --judge llm` runs the real
localhost WebSocket pipeline (Silero VAD -> faster-whisper -> **LLM-planned
support agent** -> SQLite ticketing -> Piper TTS) and writes
`evals/voice_results/simulated_caller.json`. Use `VOICE_ASR_MODEL=tiny.en`
for the tiny ablation and `--out` for a different output path.

## Voice agent

The default voice backend is `build_backend("llm")` (`LLMSupportBackend`).
It is a real LLM agent: the caller's intent, slot values, and confirmation
state are planned by the LLM via `core.llm.chat_json` with a strict JSON
schema. Deterministic code owns the safety boundary: ticket IDs/amounts are
validated, writes require an explicit confirmation, all four ticketing tools
(`create_ticket`, `classify_ticket`, `route_ticket`, `notify_customer`) are
executed in order, and every turn records structured tool calls for the
trajectory scorer.

The regex `SupportBackend` remains only as an offline unit-test mock and
`build_backend("support")` fallback; it is not on the default voice path.
Unit tests mock `agents.llm_support.chat_json` and never call LM Studio.

## Per-hop metrics

Every turn records the five hops defined in `voice/budget.py`:

| hop | start | end |
| --- | --- | --- |
| `asr_ms` | VAD end-of-speech | final transcript |
| `llm_ms` | final transcript | agent text |
| `tts_ms` | agent text | first synthesized frame |
| `network_ms` | first frame produced | first frame sent to the socket |
| `e2e_ms` | VAD end-of-speech | first frame sent |

The JSON contains per-turn `latency.hops`, per-scenario `latency.per_hop`, and
run-level `summary.per_hop` plus flat p50/p95 fields. p50/p95 are linear
percentiles over all turns.

## Seeds

- Voice/ASR seed: `VOICE_SEED` (default `42`); LLM seed: `LLM_SEED` (default `42`).
- Agent LLM calls use `use_cache=False` so reported `llm_ms` is measured, not
  disk-cache hits. Caller/judge calls remain cached.
- LLM requests send `seed` to LM Studio/OpenAI-compatible/Gemini and include it
  in the disk-cache key. LM Studio accepts the field but does not echo it, so
  effectiveness cannot be independently verified.
- `faster-whisper` exposes no seed argument. The eval seeds Python/NumPy RNG
  and forces `temperature=0.0` (no random fallback). Beam size is explicit
  (`VOICE_ASR_BEAM_SIZE`, default `1`).
- `VOICE_PACING=1` is enabled for the eval so the barge-in scenario exercises a
  genuine real-time interruption; it affects wall time, not first-frame `e2e_ms`.

## LLM-in-loop results

Host: local macOS Apple Silicon CPU, no GPU, 10 cores (4P/6E), N=18 turns
(plus 1 barge turn per size). ASR: faster-whisper int8 CPU. TTS: Piper
`en_US-lessac-medium`. VAD: Silero v5 ONNX CPU.

Agent LLM: `ornith-1.5-9b` via LM Studio. Caller LLM: `ornith-1.5-9b` via LM
Studio. Judge: `gemma-4-12b-agentic-fable5-composer2.5-v2-3.5x-tau2` via LM
Studio.

### base.en (headline)

| metric | value |
| --- | --- |
| task success | 8/9 |
| tool choice accuracy | 1.0000 |
| argument validity | 1.0000 |
| order correctness | 1.0000 |
| hallucination pass rate | 1.0000 |
| e2e p50 ms | 18,589 |
| e2e p95 ms | 24,442 |
| ASR p50/p95 ms | 842 / 1,003 |
| LLM p50/p95 ms | 17,762 / 23,241 |
| TTS p50/p95 ms | 257 / 320 |
| network p50/p95 ms | 21 / 22 |
| e2e under 1000 ms budget | 0 / 18 |
| caller WER | 7.83% |

### tiny.en ablation

| metric | value |
| --- | --- |
| task success | 6/9 |
| tool choice accuracy | 0.8889 |
| argument validity | 0.8889 |
| order correctness | 0.8889 |
| hallucination pass rate | 1.0000 |
| e2e p50 ms | 19,834 |
| e2e p95 ms | 55,413 |
| ASR p50/p95 ms | 393 / 482 |
| LLM p50/p95 ms | 19,139 / 54,693 |
| TTS p50/p95 ms | 248 / 471 |
| network p50/p95 ms | 21 / 24 |
| e2e under 1000 ms budget | 0 / 18 |
| caller WER | 17.34% |

## Tiny vs base anomaly

Before this change the "LLM" hop was the regex `SupportBackend` (about
0.7 ms p50 / 1.4 ms p95), so e2e was ASR-dominated and `tiny.en` looked like
the winner: e2e p50 686 ms vs 1,292 ms, budget 18/18 vs 1/18, task success
7/9 vs 6/9, but caller WER 21.3% vs 19.95%.

With the real LLM agent, the LLM hop is ~18 seconds p50 and dominates e2e
(about 95%+ of latency). `tiny.en` saves only ~449 ms p50 of ASR versus
`base.en`, so it cannot win budget: both sizes are 0/18 under 1 s. Base also
wins task success (8/9 vs 6/9) and caller WER (7.83% vs 17.34%). The tiny
failures are ASR phrase recognition on final responses ("complaint registered",
"refund request logged", ticket ID), not tool correctness. Therefore `base.en`
remains the headline; `tiny.en` is reported as the accuracy/latency ablation,
not as the default.

## LLM latency caveat and serving comparison

The local LM Studio instance runs on CPU/Metal without GPU offload, so LLM hop
p95 is noisy (up to ~55 s for tiny and ~24 s for base). These runs reflect a
single local eval on the configured caller/judge models; production latency
will depend on the LLM endpoint and model.

### CPU vs GPU serving (availability at eval time)

| serving | endpoint | model | status | notes |
| --- | --- | --- | --- | --- |
| LM Studio CPU | `http://localhost:1234/v1` | `ornith-1.5-9b` | **unavailable now** | `Connection refused` on port 1234 at eval time; numbers below are the last good CPU run. |
| vLLM GPU | `http://localhost:8100/v1` (SSH tunnel `8100→remote 8000`) | `qwen3-30b-a3b` GPTQ-Int4 | **unavailable** | Tunnel not running; runbook uses placeholders (`<SSH_PORT>`, `<INSTANCE_HOST>`, `<VLLM_API_KEY>`) with no credentials available. |
| Colab GPU | `colab ssh -s t4-adopted` / `-s tfx` | — | **CPU-only** | Both sessions are CPU-only (`torch.cuda.is_available()=False`, no NVIDIA driver, no vLLM, no Qwen3-30B). |

Because no LLM endpoint was reachable, a fresh GPU (or CPU) LLM-in-loop run could **not** be produced. The table below reports the last good CPU/LM Studio run honestly; the `<1 s` budget did **not** hold because the LLM planning hop (~17–19 s p50) dominates e2e (~95%+ of latency).

| size | task success | e2e p50/p95 ms | LLM p50/p95 ms | ASR p50/p95 ms | under budget |
| --- | --- | --- | --- | --- | --- |
| `base.en` (headline) | 8/9 | 18,589 / 24,442 | 17,762 / 23,241 | 842 / 1,003 | 0 / 18 |
| `tiny.en` (ablation) | 6/9 | 19,834 / 55,413 | 19,139 / 54,693 | 393 / 482 | 0 / 18 |

`tiny.en` does not win budget or task success; its ASR advantage (~449 ms p50) is swamped by the LLM hop. `base.en` remains the headline. Remaining bottleneck: **LLM planning latency**, not ASR/TTS/network.

Record run logs: `evals/voice_results/simulated_caller.json` and
`simulated_caller_tiny.json`.
