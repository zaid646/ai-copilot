"""FastAPI model service for demand forecasting.

Endpoints: POST /predict · GET /health · GET /metrics
Loads ml/artifacts/model.pt at startup.

Run:  uv run uvicorn ml.serve:app --port 8001
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.db import execute, query
from core.logging import configure_logging, get_logger, set_trace_id
from observability.metrics import (
    MODEL_LAST_MAE,
    MODEL_LATENCY,
    MODEL_REQUESTS,
    metrics_response,
)

configure_logging()
log = get_logger("ml.serve")

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
MODEL_VERSION = "lstm-weekly-v1"

app = FastAPI(title="Demand Forecast Service", version=MODEL_VERSION)

_state: dict[str, Any] = {"model": None, "scalers": {}, "config": {}}


class LSTMForecaster(torch.nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def load_model() -> None:
    ckpt = torch.load(ARTIFACTS / "model.pt", map_location="cpu", weights_only=False)
    model = LSTMForecaster(hidden=ckpt["config"].get("hidden", 32))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _state["model"] = model
    _state["scalers"] = ckpt["scalers"]
    _state["config"] = ckpt["config"]
    _state["metrics"] = ckpt.get("metrics", {})
    mae = _state["metrics"].get("lstm", {}).get("mae")
    if mae is not None:
        MODEL_LAST_MAE.set(float(mae))
    log.info("model loaded", version=MODEL_VERSION, window=ckpt["config"].get("window"))


load_model()


class PredictRequest(BaseModel):
    region: str = Field(..., description="region id, e.g. WEST")
    category: str = Field(..., description="product category, e.g. Beverages")
    horizon: int = Field(4, ge=1, le=12, description="weeks to forecast")


def _recent_series(region: str, category: str, window: int) -> list[float]:
    rows = query(
        """
        SELECT date_trunc('week', dd.full_date)::date AS wk, sum(f.revenue) AS revenue
        FROM fact_sales f
        JOIN dim_date dd   ON f.date_id = dd.date_id
        JOIN dim_product p ON f.product_id = p.product_id
        WHERE f.region_id = %s AND p.category = %s
        GROUP BY 1 ORDER BY 1 DESC LIMIT %s
        """,
        (region, category, window),
    )
    return [float(r["revenue"]) for r in reversed(rows)]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": MODEL_VERSION, "model_loaded": _state["model"] is not None}


@app.get("/metrics")
def metrics():
    return metrics_response()


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    set_trace_id(f"model-{int(time.time()*1000)}")
    t0 = time.perf_counter()
    try:
        key = f"{req.region}|{req.category}"
        scaler = _state["scalers"].get(key)
        if scaler is None:
            MODEL_REQUESTS.labels(endpoint="predict", status="error").inc()
            return {"error": f"unknown series {key}", "known_series": sorted(_state["scalers"].keys())}

        window = _state["config"]["window"]
        hist = _recent_series(req.region, req.category, window)
        if len(hist) < window:
            MODEL_REQUESTS.labels(endpoint="predict", status="insufficient_data").inc()
            return {"error": "insufficient history", "have": len(hist), "need": window}

        mu, sd = scaler["mean"], scaler["std"]
        seq = [(v - mu) / sd for v in hist]
        model: LSTMForecaster = _state["model"]

        forecasts_scaled: list[float] = []
        with torch.no_grad():
            for _ in range(req.horizon):
                x = torch.tensor(seq[-window:], dtype=torch.float32).reshape(1, window, 1)
                nxt = float(model(x).item())
                forecasts_scaled.append(nxt)
                seq.append(nxt)

        forecasts = [round(v * sd + mu, 2) for v in forecasts_scaled]
        latency_ms = int((time.perf_counter() - t0) * 1000)
        MODEL_LATENCY.labels(endpoint="predict").observe(time.perf_counter() - t0)
        MODEL_REQUESTS.labels(endpoint="predict", status="success").inc()

        execute(
            "INSERT INTO model_metrics (endpoint, model_version, latency_ms, score, inputs_json) "
            "VALUES (%s,%s,%s,%s,%s)",
            ("predict", MODEL_VERSION, latency_ms, forecasts[0],
             json.dumps({"region": req.region, "category": req.category, "horizon": req.horizon})),
        )

        return {
            "region": req.region,
            "category": req.category,
            "model_version": MODEL_VERSION,
            "horizon_weeks": req.horizon,
            "last_actual_week": hist[-1],
            "forecast": forecasts,
            "latency_ms": latency_ms,
        }
    except Exception as exc:  # pragma: no cover
        MODEL_REQUESTS.labels(endpoint="predict", status="error").inc()
        log.error("predict failed", error=str(exc))
        raise
