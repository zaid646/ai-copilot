"""Feature engineering for weekly demand forecasting.

The 20 ``(region, category)`` series and the chronological split match
``ml/train.py`` exactly:

* lookback window = 8 weeks
* validation = the last 20 one-step-ahead targets of every series (400 points)

Features are computed only from data available at the forecast origin.  Rolling
statistics are over the *history* ending at ``t``; the target is ``t + 1``.
Calendar features describe the target week, which is known at forecast time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

WINDOW = 8
VAL_PERIODS = 20
LAGS = 8
ROLL_WINDOWS = (4, 8, 12)


@dataclass
class Series:
    region: str
    category: str
    values: np.ndarray
    dates: list[date]

    @property
    def key(self) -> tuple[str, str]:
        return self.region, self.category


def load_series() -> dict[tuple[str, str], Series]:
    """Load weekly revenue plus week dates from the warehouse.

    Uses the same aggregate as ``ml.train.load_series`` so the split is
    identical; only the week label is added.
    """
    from core.db import query

    rows = query(
        """
        SELECT f.region_id, p.category,
               date_trunc('week', dd.full_date)::date AS wk,
               sum(f.revenue) AS revenue
        FROM fact_sales f
        JOIN dim_date dd   ON f.date_id = dd.date_id
        JOIN dim_product p ON f.product_id = p.product_id
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
    )
    grouped: dict[tuple[str, str], Series] = {}
    for row in rows:
        key = (str(row["region_id"]), str(row["category"]))
        series = grouped.get(key)
        if series is None:
            series = Series(region=key[0], category=key[1], values=np.empty(0), dates=[])
            grouped[key] = series
        series.values = np.append(series.values, float(row["revenue"]))
        series.dates.append(row["wk"])
    return grouped


def _series_from_values(
    region: str,
    category: str,
    values,
    start: date = date(2024, 1, 1),
) -> Series:
    array = np.asarray(values, dtype=float)
    dates = [start + timedelta(weeks=index) for index in range(len(array))]
    return Series(region=region, category=category, values=array, dates=dates)


def feature_names(regions: list[str], categories: list[str]) -> list[str]:
    names = [f"lag_{lag}" for lag in range(1, LAGS + 1)]
    for window in ROLL_WINDOWS:
        names.extend(
            [
                f"roll_mean_{window}",
                f"roll_std_{window}",
                f"roll_min_{window}",
                f"roll_max_{window}",
            ]
        )
    names.extend(["diff_1", "pct_1", "momentum_4"])
    names.extend(
        [
            "week_sin_1",
            "week_cos_1",
            "week_sin_2",
            "week_cos_2",
            "month_sin",
            "month_cos",
            "week_of_year",
            "month",
            "quarter",
            "day_of_year",
        ]
    )
    names.extend([f"quarter_{quarter}" for quarter in (1, 2, 3, 4)])
    names.extend([f"region_{region}" for region in regions])
    names.extend([f"category_{category}" for category in categories])
    return names


def _calendar_features(target_date: date) -> dict[str, float]:
    iso_week = target_date.isocalendar().week
    month = target_date.month
    quarter = (month - 1) // 3 + 1
    day_of_year = target_date.timetuple().tm_yday
    week_angle = 2.0 * math.pi * iso_week / 52.0
    month_angle = 2.0 * math.pi * month / 12.0
    features = {
        "week_sin_1": math.sin(week_angle),
        "week_cos_1": math.cos(week_angle),
        "week_sin_2": math.sin(2.0 * week_angle),
        "week_cos_2": math.cos(2.0 * week_angle),
        "month_sin": math.sin(month_angle),
        "month_cos": math.cos(month_angle),
        "week_of_year": float(iso_week),
        "month": float(month),
        "quarter": float(quarter),
        "day_of_year": float(day_of_year),
    }
    for value in (1, 2, 3, 4):
        features[f"quarter_{value}"] = 1.0 if quarter == value else 0.0
    return features


def build_feature_row(
    history: np.ndarray | list[float],
    region: str,
    category: str,
    target_date: date,
    regions: list[str],
    categories: list[str],
) -> dict[str, float]:
    """Build one feature dictionary from history ending at the origin week."""
    values = np.asarray(history, dtype=float)
    if values.size == 0:
        raise ValueError("history is empty")
    row: dict[str, float] = {}
    for lag in range(1, LAGS + 1):
        row[f"lag_{lag}"] = float(values[-lag]) if values.size >= lag else float("nan")
    for window in ROLL_WINDOWS:
        sample = values[-window:] if values.size >= window else values
        row[f"roll_mean_{window}"] = float(np.mean(sample))
        row[f"roll_std_{window}"] = float(np.std(sample)) if sample.size > 1 else 0.0
        row[f"roll_min_{window}"] = float(np.min(sample))
        row[f"roll_max_{window}"] = float(np.max(sample))
    lag_1 = row["lag_1"]
    lag_2 = row["lag_2"] if not math.isnan(row["lag_2"]) else lag_1
    row["diff_1"] = lag_1 - lag_2
    row["pct_1"] = (lag_1 - lag_2) / lag_2 if abs(lag_2) > 1e-9 else 0.0
    row["momentum_4"] = lag_1 - row["roll_mean_4"]
    row.update(_calendar_features(target_date))
    for name in regions:
        row[f"region_{name}"] = 1.0 if region == name else 0.0
    for name in categories:
        row[f"category_{name}"] = 1.0 if category == name else 0.0
    return row


def build_supervised(
    series: dict[tuple[str, str], Series],
    *,
    window: int = WINDOW,
    val_periods: int = VAL_PERIODS,
) -> dict[str, Any]:
    """Build train/validation matrices with the LSTM's chronological split."""
    if not series:
        raise ValueError("no series supplied")
    regions = sorted({key[0] for key in series})
    categories = sorted({key[1] for key in series})
    names = feature_names(regions, categories)

    x_train: list[list[float]] = []
    y_train: list[float] = []
    x_val: list[list[float]] = []
    y_val: list[float] = []
    val_meta: list[dict[str, Any]] = []

    for key, item in series.items():
        n = len(item.values)
        if n <= window + val_periods:
            raise ValueError(f"series {key} too short: {n} points")
        first_val_target = n - val_periods
        for target_index in range(window, n):
            origin = target_index - 1
            target_date = item.dates[target_index]
            features = build_feature_row(
                item.values[:target_index],
                item.region,
                item.category,
                target_date,
                regions,
                categories,
            )
            vector = [features[name] for name in names]
            target = float(item.values[target_index])
            if target_index >= first_val_target:
                x_val.append(vector)
                y_val.append(target)
                val_meta.append(
                    {
                        "series": f"{item.region}|{item.category}",
                        "origin_index": origin,
                        "target_index": target_index,
                        "target_week": target_date.isoformat(),
                        "lag_1": float(item.values[origin]),
                    }
                )
            else:
                x_train.append(vector)
                y_train.append(target)

    return {
        "X_train": np.asarray(x_train, dtype=float),
        "y_train": np.asarray(y_train, dtype=float),
        "X_val": np.asarray(x_val, dtype=float),
        "y_val": np.asarray(y_val, dtype=float),
        "feature_names": names,
        "val_meta": val_meta,
        "regions": regions,
        "categories": categories,
        "window": window,
        "val_periods": val_periods,
    }


def make_series(
    values_by_key: dict[tuple[str, str], list[float]], start: date
) -> dict[tuple[str, str], Series]:
    """Test/CLI helper: build synthetic series with weekly dates."""
    return {
        key: _series_from_values(key[0], key[1], values, start=start)
        for key, values in values_by_key.items()
    }
