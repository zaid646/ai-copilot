# Voice evaluation: metrics, seeds and latency

`python -m evals.simulated_caller --caller llm --judge llm` runs the real
localhost WebSocket pipeline (Silero VAD -> faster-whisper -> support agent ->
Piper TTS) and writes `evals/voice_results/simulated_caller.json`.

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
run-level `summary.per_hop` / flat `asr_p50_ms`, `tts_p95_ms`, etc. p50/p95 are
linear-interpolated percentiles over all turns.

## Seeds

- Voice/ASR seed: `VOICE_SEED` (default `42`); LLM seed: `LLM_SEED` (default `42`).
- LLM requests send `seed` to LM Studio, OpenAI-compatible vLLM, and Gemini and
  include it in the disk-cache key. LM Studio accepts the field but does not echo
  it, so its effectiveness cannot be independently verified.
- `faster-whisper` 1.x `transcribe()` exposes **no seed argument**. The eval
  seeds Python/NumPy RNG and forces `temperature=0.0` (no random fallback). Beam
  size is explicit (`VOICE_ASR_BEAM_SIZE`, default `1`).

## Latency result (Apple Silicon CPU, no GPU)

Primary config `base.en` (quality-first), 18 turns:

| hop | p50 | p95 |
| --- | --- | --- |
| ASR | 1050 ms | 1367 ms |
| TTS | 230 ms | 289 ms |
| network | 0.15 ms | 0.45 ms |
| e2e | 1292 ms | 1611 ms |

The 1000 ms e2e budget is missed; ASR is ~81% of e2e. A `tiny.en` ablation
(`evals/voice_results/simulated_caller_tiny.json`) reaches e2e p50 686 ms /
p95 930 ms, but task success drops to 7/9 and caller WER rises to 21.3%, so it
is not the default. First-chunk TTS streaming, 8 ASR threads, trailing-silence
trimming and 1-beam decoding are enabled; partial-transcript agent start is not
used because the support agent is deterministic and speculative tool execution
could act on a wrong transcript.

Record run logs: `evals/voice_results/simulated_caller.json` and
`simulated_caller_tiny.json`.
