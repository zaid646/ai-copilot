"""FastAPI model service for demand forecasting.

Endpoints: POST /predict · GET /health · GET /metrics
Loads ml/artifacts/model.pt at startup.

Run:  uv run uvicorn ml.serve:app --port 8001
"""

from __future__ import annotations

import os
import sys
import time

# Torch and LightGBM/XGBoost each ship their own OpenMP runtime on macOS.
# Loading both with the default thread count segfaults when the joblib bundle
# is unpickled; a single thread keeps the service stable (and predictions are
# still well inside the latency budget).
os.environ.setdefault("OMP_NUM_THREADS", "1")
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import joblib
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
FORECAST_BUNDLE = Path(__file__).resolve().parents[1] / "forecast" / "models" / "forecaster.joblib"
MODEL_VERSION = "lstm-weekly-v1"
TREE_VERSION = "xgb-lightgbm-weekly-v1"
DEFAULT_MODEL = os.getenv("FORECAST_MODEL", "lstm").lower()

app = FastAPI(title="Demand Forecast Service", version=MODEL_VERSION)

_state: dict[str, Any] = {
    "model": None,
    "scalers": {},
    "config": {},
    "bundle": None,
    "bundle_loaded": False,
}


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


def load_forecast_bundle() -> dict[str, Any] | None:
    """Lazily load the XGBoost/LightGBM serving bundle."""
    if _state["bundle_loaded"]:
        return _state["bundle"]
    _state["bundle_loaded"] = True
    if not FORECAST_BUNDLE.exists():
        log.warning("forecast bundle missing", path=str(FORECAST_BUNDLE))
        return None
    _state["bundle"] = joblib.load(FORECAST_BUNDLE)
    log.info("forecast bundle loaded", path=str(FORECAST_BUNDLE))
    return _state["bundle"]


def load_model() -> None:
    path = ARTIFACTS / "model.pt"
    if not path.exists():
        log.warning("lstm artifact missing", path=str(path))
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
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


if DEFAULT_MODEL == "lstm":
    load_model()
else:
    log.info("lstm model lazy-loaded (tree default)", default_model=DEFAULT_MODEL)


class PredictRequest(BaseModel):
    region: str = Field(..., description="region id, e.g. WEST")
    category: str = Field(..., description="product category, e.g. Beverages")
    horizon: int = Field(4, ge=1, le=12, description="weeks to forecast")
    model: str | None = Field(
        None, description="lstm (default), xgb, or lightgbm; overrides FORECAST_MODEL"
    )


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


def _recent_series_dated(region: str, category: str, window: int) -> list[tuple[date, float]]:
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
    return [(r["wk"], float(r["revenue"])) for r in reversed(rows)]


def _known_tree_series(bundle: dict[str, Any]) -> list[str]:
    return [
        f"{region}|{category}" for region in bundle["regions"] for category in bundle["categories"]
    ]


def _record_prediction(
    version: str, latency_ms: int, score: float, payload: dict[str, Any]
) -> None:
    execute(
        "INSERT INTO model_metrics (endpoint, model_version, latency_ms, score, inputs_json) "
        "VALUES (%s,%s,%s,%s,%s)",
        ("predict", version, latency_ms, score, json.dumps(payload)),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": MODEL_VERSION,
        "model_loaded": _state["model"] is not None,
        "default_model": DEFAULT_MODEL,
        "available_models": ["lstm"] + (["xgb", "lightgbm"] if FORECAST_BUNDLE.exists() else []),
        "tree_bundle": FORECAST_BUNDLE.exists(),
    }


@app.get("/metrics")
def metrics():
    return metrics_response()


def _tree_predict(req: PredictRequest, family: str) -> dict[str, Any]:
    from forecast.xgb_forecaster import forecast_horizon

    bundle = load_forecast_bundle()
    if bundle is None:
        MODEL_REQUESTS.labels(endpoint="predict", status="missing_bundle").inc()
        return {"error": "tree model bundle not found", "path": str(FORECAST_BUNDLE)}

    known = _known_tree_series(bundle)
    key = f"{req.region}|{req.category}"
    if key not in known:
        MODEL_REQUESTS.labels(endpoint="predict", status="error").inc()
        return {"error": f"unknown series {key}", "known_series": known}

    rows = _recent_series_dated(req.region, req.category, 24)
    need = max(bundle["window"], 12)
    if len(rows) < need:
        MODEL_REQUESTS.labels(endpoint="predict", status="insufficient_data").inc()
        return {"error": "insufficient history", "have": len(rows), "need": need}

    t0 = time.perf_counter()
    history = [value for _, value in rows]
    last_week = rows[-1][0]
    forecasts = forecast_horizon(
        bundle,
        history,
        last_week,
        req.region,
        req.category,
        req.horizon,
        model=family,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    MODEL_LATENCY.labels(endpoint="predict").observe(time.perf_counter() - t0)
    MODEL_REQUESTS.labels(endpoint="predict", status="success").inc()
    _record_prediction(
        TREE_VERSION,
        latency_ms,
        forecasts[0],
        {"region": req.region, "category": req.category, "horizon": req.horizon, "model": family},
    )
    return {
        "region": req.region,
        "category": req.category,
        "model": family,
        "model_version": TREE_VERSION,
        "horizon_weeks": req.horizon,
        "last_actual_week": last_week.isoformat(),
        "last_actual_value": round(history[-1], 2),
        "forecast": forecasts,
        "latency_ms": latency_ms,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    set_trace_id(f"model-{int(time.time() * 1000)}")
    family = (req.model or DEFAULT_MODEL).lower()
    if family in {"xgb", "xgboost"}:
        return _tree_predict(req, "xgb")
    if family in {"lightgbm", "lgbm"}:
        return _tree_predict(req, "lightgbm")
    if family != "lstm":
        MODEL_REQUESTS.labels(endpoint="predict", status="error").inc()
        return {"error": f"unknown model {family}", "available_models": ["lstm", "xgb", "lightgbm"]}

    if _state["model"] is None:
        load_model()
    if _state["model"] is None:
        MODEL_REQUESTS.labels(endpoint="predict", status="missing_lstm").inc()
        return {"error": "LSTM artifact not found; use model=xgb or model=lightgbm"}

    t0 = time.perf_counter()
    try:
        key = f"{req.region}|{req.category}"
        scaler = _state["scalers"].get(key)
        if scaler is None:
            MODEL_REQUESTS.labels(endpoint="predict", status="error").inc()
            return {
                "error": f"unknown series {key}",
                "known_series": sorted(_state["scalers"].keys()),
            }

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

        _record_prediction(
            MODEL_VERSION,
            latency_ms,
            forecasts[0],
            {
                "region": req.region,
                "category": req.category,
                "horizon": req.horizon,
                "model": "lstm",
            },
        )

        return {
            "region": req.region,
            "category": req.category,
            "model": "lstm",
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


@app.post("/predict_xgb")
def predict_xgb(req: PredictRequest) -> dict[str, Any]:
    return _tree_predict(req.model_copy(update={"model": "xgb"}), "xgb")


@app.post("/predict_lightgbm")
def predict_lightgbm(req: PredictRequest) -> dict[str, Any]:
    return _tree_predict(req.model_copy(update={"model": "lightgbm"}), "lightgbm")


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Demand forecast FastAPI service")
    parser.add_argument("--model", choices=["lstm", "xgb", "lightgbm"], default=DEFAULT_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args, _ = parser.parse_known_args()
    os.environ["FORECAST_MODEL"] = args.model
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    _cli()
