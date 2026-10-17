"""Prometheus metrics shared by the API and the model service."""
from __future__ import annotations

from fastapi import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# --- HTTP / API ---------------------------------------------------------
REQUEST_COUNT = Counter(
    "copilot_requests_total", "API requests", ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "copilot_request_latency_seconds", "API request latency", ["endpoint"]
)

# --- Agent --------------------------------------------------------------
AGENT_RUNS = Counter("agent_runs_total", "Agent graph runs", ["status"])
AGENT_STEP_LATENCY = Histogram(
    "agent_step_latency_seconds", "Per-node latency", ["node"]
)
AGENT_STEPS = Counter("agent_steps_total", "Steps executed per node", ["node"])
AGENT_LOOP_ITERATIONS = Histogram(
    "agent_loop_iterations", "Supervisor loop iterations per run", buckets=[1, 2, 3, 4, 5, 6, 8]
)

# --- LLM ----------------------------------------------------------------
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens used", ["kind"])
LLM_ERRORS = Counter("llm_errors_total", "LLM call errors")

# --- Tools --------------------------------------------------------------
TOOL_CALLS = Counter("tool_calls_total", "Tool invocations", ["tool", "status"])
TOOL_LATENCY = Histogram("tool_latency_seconds", "Tool latency", ["tool"])

# --- Guardrails ---------------------------------------------------------
GUARDRAIL_EVENTS = Counter(
    "guardrail_events_total", "Guardrail decisions", ["direction", "action"]
)
GUARDRAIL_BLOCKS = Counter(
    "guardrail_blocks_total", "Guardrail blocks", ["direction", "rule"]
)

# --- Model service ------------------------------------------------------
MODEL_LATENCY = Histogram(
    "model_predict_latency_seconds", "Model inference latency", ["endpoint"]
)
MODEL_REQUESTS = Counter(
    "model_requests_total", "Model service requests", ["endpoint", "status"]
)
MODEL_LAST_MAE = Gauge("model_last_val_mae", "Validation MAE of the served model")


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
