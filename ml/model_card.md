# Model Card — Demand Forecast (lstm-weekly-v1)

## Overview
- **Task:** Weekly revenue (demand) forecast for a `(region, category)` series.
- **Model:** 1-layer LSTM (hidden=32) → MLP head → scalar next-week revenue.
- **Input:** last 8 weeks of weekly revenue for the series (standardized).
- **Output:** next-week revenue (extendable to an N-week horizon by rolling predictions).
- **Framework:** PyTorch (trained on Apple MPS).
- **Version:** `lstm-weekly-v1` (artifact: `ml/artifacts/model.pt`).

## Data
- Source: `fact_sales` joined with `dim_date` / `dim_product`, aggregated weekly.
- Series: 4 regions × 5 categories = 20 series, 78 weeks each.
- Split: chronological — last 20 weeks per series held out (one-step-ahead
  backtest, 400 validation points).

## Metrics (validation, held-out weeks)
| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| LSTM (ours) | 288.39 | 387.88 | 37.2% |
| Last-value baseline | 343.24 | 434.85 | 36.4% |

**Improvement:** ~16% MAE reduction vs the last-value baseline (400 one-step-ahead
predictions). Note: the series are close to random walks with an injected
structural break (West, Q3); a short 6-week holdout is dominated by that break's
variance, so a 20-week holdout is used for a more stable estimate.

## Intended use
- Copilot tool `forecast_demand(region, category, horizon_weeks)` to support
  "what happens next" reasoning alongside the deterministic KPI tool.
- Directional planning aid — **not** a financial system of record.

## Limitations
- Only 20 coarse series; no product-level or customer-level forecasts.
- Weekly granularity; 20-week holdout per series → still limited validation.
- Series are near random walks, so gains over last-value are modest (~16% MAE).
- Does not ingest exogenous regressors (promotions, price changes) yet.
- Performance degrades on series with structural breaks (e.g., the injected
  West-region Q3 drop is not predictable from history alone).

## Ethical / operational notes
- Forecasts are advisory; the copilot must cite the model version and horizon.
- No PII in training data (aggregated revenue only).

## Reproduce
```bash
uv run python -m ml.train        # trains + writes artifacts/model.pt + metrics.json
uv run python -m ml.evaluate     # writes artifacts/eval_report.json
uv run uvicorn ml.serve:app --port 8001   # serves /predict
```
