"""XGBoost / LightGBM weekly demand forecaster.

Evaluation is the same 400 one-step-ahead backtests used by the LSTM:
20 region x category series, the last 20 weeks of each series held out.

Run:
    .venv/bin/python -m forecast.xgb_forecaster
Artifacts:
    forecast/results/xgb_metrics.json
    forecast/results/feature_importance.png
    forecast/models/forecaster.joblib  (gitignored; for the FastAPI service)
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import date, timedelta
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from forecast.features import (
    build_feature_row,
    build_supervised,
    load_series,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = Path(__file__).resolve().parent / "models"
SEED = 42


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict[str, Any]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denominator = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)
    return {
        "name": name,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 3),
        "n": int(len(y_true)),
    }


def train_xgboost(x_train: np.ndarray, y_train: np.ndarray, seed: int = SEED):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(x_train, y_train)
    return model


def train_lightgbm(x_train: np.ndarray, y_train: np.ndarray, seed: int = SEED):
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    return model


def _load_lstm_metrics() -> dict[str, Any] | None:
    path = ROOT / "ml" / "artifacts" / "eval_report.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    overall = payload.get("overall", {})
    lstm = overall.get("lstm")
    baseline = overall.get("baseline")
    if not lstm or not baseline:
        return None
    return {
        "lstm": lstm,
        "baseline": baseline,
        "source": str(path.relative_to(ROOT)),
        "note": "Reproduced locally by ml.train/ml.evaluate on the seeded warehouse.",
    }


def _importance(model, feature_names: list[str]) -> np.ndarray:
    values = np.asarray(model.feature_importances_, dtype=float)
    total = values.sum()
    if total > 0:
        values = values / total
    return values


def _plot_importance(
    importance: np.ndarray,
    feature_names: list[str],
    out: Path,
    top_n: int = 20,
) -> list[dict[str, float]]:
    order = np.argsort(importance)[::-1][:top_n]
    top = [
        {"feature": feature_names[int(i)], "importance": round(float(importance[int(i)]), 6)}
        for i in order
    ]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(order))))
    labels = [row["feature"] for row in top][::-1]
    values = [row["importance"] for row in top][::-1]
    ax.barh(labels, values, color="#1f77b4")
    ax.set_xlabel("Mean normalized importance (XGBoost + LightGBM)")
    ax.set_title("Demand forecaster — top feature importances")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return top


def _versions() -> dict[str, str]:
    packages = ["numpy", "scipy", "scikit-learn", "xgboost", "lightgbm", "joblib"]
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def build_bundle(
    xgb_model,
    lgb_model,
    built: dict[str, Any],
    metrics: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "models": {"xgb": xgb_model, "lightgbm": lgb_model},
        "feature_names": built["feature_names"],
        "regions": built["regions"],
        "categories": built["categories"],
        "window": built["window"],
        "val_periods": built["val_periods"],
        "seed": seed,
        "metrics": metrics,
    }


def predict_next(
    bundle: dict[str, Any],
    history: list[float] | np.ndarray,
    region: str,
    category: str,
    target_date: date,
    model: str = "xgb",
) -> float:
    """One-step prediction from history ending at the origin week."""
    key = {"xgb": "xgb", "xgboost": "xgb", "lightgbm": "lightgbm", "lgbm": "lightgbm"}.get(
        model.lower()
    )
    if key is None:
        raise ValueError(f"unknown model: {model}")
    if region not in bundle["regions"]:
        raise ValueError(f"unknown region: {region}")
    if category not in bundle["categories"]:
        raise ValueError(f"unknown category: {category}")
    row = build_feature_row(
        history,
        region,
        category,
        target_date,
        bundle["regions"],
        bundle["categories"],
    )
    vector = np.asarray([[row[name] for name in bundle["feature_names"]]], dtype=float)
    prediction = bundle["models"][key].predict(vector)[0]
    return float(prediction)


def forecast_horizon(
    bundle: dict[str, Any],
    history: list[float] | np.ndarray,
    last_week: date,
    region: str,
    category: str,
    horizon: int,
    model: str = "xgb",
) -> list[float]:
    """Rolling multi-step forecast, recomputing features after each step."""
    values = [float(v) for v in history]
    forecasts: list[float] = []
    for step in range(1, horizon + 1):
        target_date = last_week + timedelta(weeks=step)
        prediction = predict_next(bundle, values, region, category, target_date, model=model)
        forecasts.append(round(prediction, 2))
        values.append(prediction)
    return forecasts


def main() -> int:
    parser = argparse.ArgumentParser(description="XGBoost/LightGBM weekly demand forecaster")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "xgb_metrics.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-model", action="store_true", help="skip the serving bundle")
    args = parser.parse_args()

    series = load_series()
    built = build_supervised(series)
    x_train, y_train = built["X_train"], built["y_train"]
    x_val, y_val = built["X_val"], built["y_val"]
    if len(y_val) != 400:
        raise SystemExit(f"expected 400 validation backtests, got {len(y_val)}")

    xgb_model = train_xgboost(x_train, y_train, seed=args.seed)
    lgb_model = train_lightgbm(x_train, y_train, seed=args.seed)
    xgb_pred = np.asarray(xgb_model.predict(x_val), dtype=float)
    lgb_pred = np.asarray(lgb_model.predict(x_val), dtype=float)
    baseline = np.asarray([row["lag_1"] for row in built["val_meta"]], dtype=float)

    metrics: dict[str, Any] = {
        "task": "weekly demand one-step-ahead forecasting",
        "n_series": len(series),
        "n_backtests": int(len(y_val)),
        "split": {
            "window": built["window"],
            "validation_weeks_per_series": built["val_periods"],
            "validation_targets_per_series": len(y_val) // max(1, len(series)),
            "scheme": "chronological last-20-weeks per series, identical to ml.train",
        },
        "seed": args.seed,
        "models": {
            "xgboost": _metrics(y_val, xgb_pred, "xgboost"),
            "lightgbm": _metrics(y_val, lgb_pred, "lightgbm"),
            "baseline": _metrics(y_val, baseline, "last_value"),
        },
        "reference_committed": {
            "source": "ml/model_card.md (pre-existing, not re-run here)",
            "status": "superseded",
            "lstm": {"mae": 288.39, "rmse": 387.88, "mape": 37.2},
            "baseline": {"mae": 343.24, "rmse": 434.85, "mape": 36.4},
            "note": (
                "Superseded: the earlier 288.39/343.24 reference came from an "
                "unseeded run. The seeded reproduction is LSTM 265.362 / baseline "
                "345.586; all comparisons use that reproduced pair on the exact "
                "same 400 windows."
            ),
        },
        "feature_count": len(built["feature_names"]),
        "features": built["feature_names"],
        "versions": _versions(),
    }

    lstm = _load_lstm_metrics()
    if lstm is not None:
        metrics["models"]["lstm"] = lstm["lstm"]
        metrics["models"]["reproduced_lstm"] = lstm["lstm"]
        metrics["models"]["reproduced_baseline"] = lstm["baseline"]
        metrics["lstm_source"] = lstm
    else:
        metrics["models"]["lstm"] = None
        metrics["lstm_source"] = {
            "note": "ml/artifacts/eval_report.json not found; run ml.train + ml.evaluate first."
        }

    baseline_mae = metrics["models"]["baseline"]["mae"]
    for name in ("xgboost", "lightgbm"):
        model_mae = metrics["models"][name]["mae"]
        metrics["models"][name]["improvement_vs_baseline_pct"] = round(
            (baseline_mae - model_mae) / baseline_mae * 100.0, 2
        )
        if metrics["models"].get("lstm"):
            lstm_mae = metrics["models"]["lstm"]["mae"]
            metrics["models"][name]["improvement_vs_lstm_pct"] = round(
                (lstm_mae - model_mae) / lstm_mae * 100.0, 2
            )

    importance = (
        _importance(xgb_model, built["feature_names"])
        + _importance(lgb_model, built["feature_names"])
    ) / 2.0
    top = _plot_importance(
        importance,
        built["feature_names"],
        RESULTS_DIR / "feature_importance.png",
    )
    metrics["top_features"] = top
    metrics["feature_importance_all"] = {
        name: round(float(value), 6)
        for name, value in sorted(
            zip(built["feature_names"], importance, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if not args.no_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        bundle = build_bundle(xgb_model, lgb_model, built, metrics, args.seed)
        joblib.dump(bundle, MODELS_DIR / "forecaster.joblib")

    summary = {
        name: {key: metrics["models"][name][key] for key in ("mae", "rmse", "mape")}
        for name in ("xgboost", "lightgbm", "baseline", "lstm")
        if metrics["models"].get(name)
    }
    print(
        json.dumps(
            {"n_backtests": len(y_val), "models": summary, "top_features": top[:5]}, indent=2
        )
    )
    print(f"results -> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
