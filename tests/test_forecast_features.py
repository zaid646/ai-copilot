"""Tests for the leakage-free weekly feature builder (no database needed)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from forecast.features import (
    LAGS,
    VAL_PERIODS,
    WINDOW,
    build_feature_row,
    build_supervised,
    make_series,
)


def _synthetic_series(n: int = 60) -> dict[tuple[str, str], list[float]]:
    keys = [("NORTH", "Beverages"), ("WEST", "Snacks"), ("EAST", "Beverages")]
    return {key: [100.0 + 2.0 * i + 10.0 * math.sin(i / 3.0) for i in range(n)] for key in keys}


def test_split_matches_lstm_window_scheme():
    series = make_series(_synthetic_series(), start=date(2024, 1, 1))
    built = build_supervised(series)

    assert built["window"] == WINDOW
    assert len(built["y_val"]) == len(series) * VAL_PERIODS == 60
    assert len(built["y_train"]) == len(series) * (60 - WINDOW - VAL_PERIODS) == 96
    assert built["X_train"].shape == (96, len(built["feature_names"]))
    assert built["X_val"].shape == (60, len(built["feature_names"]))


def test_validation_targets_are_last_twenty_weeks():
    values = _synthetic_series()["NORTH", "Beverages"]
    series = make_series({("NORTH", "Beverages"): values}, start=date(2024, 1, 1))
    built = build_supervised(series)

    assert [row["target_index"] for row in built["val_meta"]] == list(range(40, 60))
    assert (
        built["val_meta"][0]["target_week"] == (date(2024, 1, 1) + timedelta(weeks=40)).isoformat()
    )


def test_features_only_use_history_before_target():
    values = _synthetic_series()["NORTH", "Beverages"]
    series = make_series({("NORTH", "Beverages"): values}, start=date(2024, 1, 1))
    built = build_supervised(series)
    names = built["feature_names"]
    lag_1_index = names.index("lag_1")
    target_index = built["val_meta"][0]["target_index"]

    assert built["X_val"][0, lag_1_index] == values[target_index - 1]
    assert built["X_val"][0, lag_1_index] != values[target_index]
    assert built["X_val"][0, names.index(f"lag_{LAGS}")] == values[target_index - LAGS]
    assert np.isfinite(built["X_train"]).all()
    assert np.isfinite(built["X_val"]).all()


def test_calendar_and_categorical_encodings():
    regions = ["NORTH", "WEST"]
    categories = ["Beverages", "Snacks"]
    row = build_feature_row([100.0] * 12, "WEST", "Snacks", date(2024, 6, 30), regions, categories)

    assert row["region_WEST"] == 1.0 and row["region_NORTH"] == 0.0
    assert row["category_Snacks"] == 1.0 and row["category_Beverages"] == 0.0
    assert row["quarter"] == 2.0 and row["quarter_2"] == 1.0
    assert math.isclose(row["month_sin"], math.sin(2 * math.pi * 6 / 12), rel_tol=1e-9)
    assert row["roll_mean_12"] == pytest.approx(100.0)
    assert row["diff_1"] == 0.0
    assert row["pct_1"] == 0.0


def test_short_history_uses_available_periods():
    row = build_feature_row(
        [10.0, 20.0], "NORTH", "Beverages", date(2024, 1, 8), ["NORTH"], ["Beverages"]
    )

    assert row["lag_1"] == 20.0
    assert row["lag_2"] == 10.0
    assert math.isnan(row["lag_3"])
    assert row["roll_mean_4"] == 15.0
    assert row["roll_std_4"] == 5.0
