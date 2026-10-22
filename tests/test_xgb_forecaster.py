"""Smoke tests for the XGBoost/LightGBM forecaster (no database needed)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from forecast.features import build_supervised, make_series
from forecast.xgb_forecaster import (
    _metrics,
    build_bundle,
    forecast_horizon,
    predict_next,
    train_lightgbm,
    train_xgboost,
)

START = date(2024, 1, 1)


def _built():
    keys = [("NORTH", "Beverages"), ("WEST", "Snacks"), ("EAST", "Beverages")]
    values = {key: [100.0 + 2.0 * i + 12.0 * math.sin(i / 4.0) for i in range(70)] for key in keys}
    return build_supervised(make_series(values, start=START))


def test_both_models_train_and_predict():
    built = _built()
    xgb = train_xgboost(built["X_train"], built["y_train"], seed=7)
    lgb = train_lightgbm(built["X_train"], built["y_train"], seed=7)
    bundle = build_bundle(xgb, lgb, built, metrics={}, seed=7)

    history = [100.0 + 2.0 * i for i in range(24)]
    last_week = START + timedelta(weeks=23)
    for model in ("xgb", "lightgbm"):
        prediction = predict_next(
            bundle, history, "NORTH", "Beverages", last_week + timedelta(weeks=1), model=model
        )
        assert math.isfinite(prediction)
        horizon = forecast_horizon(bundle, history, last_week, "NORTH", "Beverages", 4, model=model)
        assert len(horizon) == 4
        assert all(math.isfinite(value) for value in horizon)


def test_predictions_are_deterministic():
    built = _built()
    xgb_a = train_xgboost(built["X_train"], built["y_train"], seed=11)
    xgb_b = train_xgboost(built["X_train"], built["y_train"], seed=11)
    history = list(np.linspace(90.0, 140.0, 24))
    target = START + timedelta(weeks=24)

    assert predict_next(
        build_bundle(xgb_a, xgb_a, built, {}, 11),
        history,
        "WEST",
        "Snacks",
        target,
    ) == predict_next(
        build_bundle(xgb_b, xgb_b, built, {}, 11),
        history,
        "WEST",
        "Snacks",
        target,
    )


def test_unknown_series_and_model_are_rejected():
    built = _built()
    xgb = train_xgboost(built["X_train"], built["y_train"], seed=3)
    bundle = build_bundle(xgb, xgb, built, {}, 3)

    with pytest.raises(ValueError, match="unknown region"):
        predict_next(bundle, [1.0] * 12, "SOUTH", "Beverages", START)
    with pytest.raises(ValueError, match="unknown category"):
        predict_next(bundle, [1.0] * 12, "NORTH", "Toys", START)
    with pytest.raises(ValueError, match="unknown model"):
        predict_next(bundle, [1.0] * 12, "NORTH", "Beverages", START, model="prophet")


def test_metrics_match_sklearn():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 180.0, 330.0])
    result = _metrics(y_true, y_pred, "test")

    assert result["mae"] == pytest.approx(20.0)
    assert result["rmse"] == pytest.approx(math.sqrt((100 + 400 + 900) / 3), abs=1e-3)
    assert result["mape"] == pytest.approx((10 / 100 + 20 / 200 + 30 / 300) / 3 * 100, abs=1e-3)
    assert result["n"] == 3
