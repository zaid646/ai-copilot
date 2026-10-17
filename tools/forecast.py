"""Tool: forecast_demand — call the ML model service for a weekly forecast."""
from __future__ import annotations

import time
from typing import Any

import httpx

from core.config import settings
from core.logging import get_logger
from observability.metrics import TOOL_CALLS, TOOL_LATENCY
from tools.contract import ErrorCode, err, ok

log = get_logger("tools.forecast")

KNOWN_CATEGORIES = ["Beverages", "Dairy", "Household", "Produce", "Snacks"]
TIMEOUT_S = 20.0


def forecast_demand(region: str, category: str, horizon_weeks: int = 4) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        region_id = (region or "").strip().upper()
        cat = (category or "").strip()
        if not region_id or not cat:
            return err(ErrorCode.INPUT_INVALID, "region and category are required")
        if not (1 <= int(horizon_weeks) <= 12):
            return err(ErrorCode.INPUT_INVALID, "horizon_weeks must be between 1 and 12")

        url = f"{settings.model_service_url}/predict"
        body = {"region": region_id, "category": cat, "horizon": int(horizon_weeks)}
        try:
            resp = httpx.post(url, json=body, timeout=TIMEOUT_S)
        except httpx.TimeoutException:
            TOOL_CALLS.labels(tool="forecast_demand", status="timeout").inc()
            return err(ErrorCode.UPSTREAM_TIMEOUT, f"model service timed out after {TIMEOUT_S}s")
        except httpx.HTTPError as exc:
            TOOL_CALLS.labels(tool="forecast_demand", status="unavailable").inc()
            return err(ErrorCode.MODEL_UNAVAILABLE, f"model service unreachable: {exc}")

        if resp.status_code != 200:
            TOOL_CALLS.labels(tool="forecast_demand", status="unavailable").inc()
            return err(ErrorCode.MODEL_UNAVAILABLE, f"model service returned {resp.status_code}")

        payload = resp.json()
        if "error" in payload and "forecast" not in payload:
            return err(ErrorCode.INPUT_INVALID, payload["error"])

        TOOL_CALLS.labels(tool="forecast_demand", status="success").inc()
        return ok(payload)
    finally:
        TOOL_LATENCY.labels(tool="forecast_demand").observe(time.perf_counter() - start)
