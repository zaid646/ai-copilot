# Tree-based demand forecaster (`xgb-lightgbm-weekly-v1`)

Gradient-boosted trees as a tabular alternative to the LSTM in `ml/`. The
forecast task, data split, and baseline are identical, so the numbers are
directly comparable.

## Task and split

- 20 weekly revenue series (`4 regions x 5 categories`) from `fact_sales`.
- 8-week lookback, one-step-ahead target.
- Validation = the **last 20 targets of every series** (400 points), exactly
  the split used by `ml.train` / `ml.evaluate`.
- Features use only data up to the forecast origin. Calendar features describe
  the target week, which is known at forecast time.
- No random split and no shuffling.

## Features (46)

| Group | Features |
|---|---|
| Lags | `lag_1` ... `lag_8` |
| Rolling history | mean / std / min / max over 4, 8, 12 weeks |
| Change | `diff_1`, `pct_1`, `momentum_4` |
| Calendar (target week) | week-of-year sin/cos (1st + 2nd harmonic), month sin/cos, week, month, quarter, day-of-year, quarter one-hots |
| Series identity | region and category one-hots |

The most important features are the 12-week rolling mean (19.7% of combined
importance), then target-week day-of-year and the 8/4-week rolling means. The
raw `lag_1` is deliberately not dominant: the model smooths across recent
weeks rather than copying the last value.

## Results (400 held-out one-step backtests)

| Model | MAE | RMSE | MAPE | MAE vs baseline |
|---|---:|---:|---:|---:|
| XGBoost | 268.132 | 338.613 | 27.294% | **-22.41%** |
| LightGBM | **265.169** | **334.783** | **27.211%** | **-23.27%** |
| LSTM (reproduced locally) | 265.362 | 346.022 | 28.090% | -23.22% |
| Last-value baseline (reproduced) | 345.586 | 437.222 | 33.508% | — |

LightGBM is **0.07%** better than the reproduced LSTM on MAE (0.19 revenue
units) and XGBoost is **1.04%** worse. Treat the LSTM and trees as practically
tied on this small, near-random-walk dataset; the tree models are materially
better than the last-value baseline.

> **Reproducibility note.** The pre-existing `ml/model_card.md` reports LSTM
> MAE 288.39 and baseline 343.24. Re-running `ml.train` + `ml.evaluate` against
> the seeded Docker warehouse produces **265.362 / 345.586** instead. The
> original figures are retained as a committed reference in
> `results/xgb_metrics.json` (`reference_committed`), but all comparisons here
> use the **locally reproduced** LSTM and baseline on the *same* 400 windows.
> We do not claim to reproduce the model card exactly.

## Reproduce

```bash
# Postgres must be up and seeded: make up && make seed
uv run python -m ml.train          # LSTM + ml/artifacts/model.pt (for the LSTM row)
uv run python -m ml.evaluate       # ml/artifacts/eval_report.json
uv run python -m forecast.xgb_forecaster
```

Outputs:

- `forecast/results/xgb_metrics.json` — full metrics, features, versions,
  committed-reference values, top importances.
- `forecast/results/feature_importance.png` — top-20 combined importance.
- `forecast/models/forecaster.joblib` — serving bundle (gitignored).

The run is deterministic (`seed=42`, single-threaded XGBoost/LightGBM,
`deterministic=True` for LightGBM); two consecutive runs produced identical
metrics and feature rankings.

## Serving

`ml/serve.py` loads both families. Select with the request field or env var:

```bash
FORECAST_MODEL=xgb uv run uvicorn ml.serve:app --port 8001
# or
FORECAST_MODEL=lightgbm uv run uvicorn ml.serve:app --port 8001
```

```bash
curl -X POST localhost:8001/predict -H 'content-type: application/json' \
  -d '{"region":"WEST","category":"Beverages","horizon":4,"model":"lightgbm"}'
```

`POST /predict_xgb` and `POST /predict_lightgbm` are convenience aliases;
`model` may be `lstm` (default), `xgb`, or `lightgbm`. Multi-step forecasts are
recursive: each prediction is appended to the history and features are rebuilt.

On macOS, PyTorch and LightGBM/XGBoost each ship an OpenMP runtime; importing
both with default threading segfaults when the joblib bundle is unpickled.
`ml/serve.py` pins `OMP_NUM_THREADS=1` before importing torch. Latency is ~3-6
ms per request on CPU.

## Caveats

- Validation is still only 20 weeks per series; the injected West Q3 structural
  break dominates the errors.
- `day_of_year` can act as a time index and may absorb that break; it is known
  at forecast time, but this is a small-sample artifact, not a causal driver.
- No exogenous regressors (promotions, price, weather) yet.
- The 46 features include region/category one-hots, so the bundle is tied to the
  current 20 series. A new series needs retraining or a fallback model.
- Advisory only; not a financial system of record.
