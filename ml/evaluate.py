"""Evaluate the trained forecast model on the held-out weeks.

Uses the *same* windowing scheme as ml.train (window start index s predicts
value at s+WINDOW; validation = last VAL_PERIODS windows of each series), so
the numbers reproduce training-time validation.

Run:  uv run python -m ml.evaluate
Writes: ml/artifacts/eval_report.json (+ prints a per-series table)
"""
from __future__ import annotations

import json

import numpy as np
import torch

from core.config import ROOT
from core.logging import configure_logging, get_logger
from ml.train import VAL_PERIODS, WINDOW, LSTMForecaster, load_series, metrics_from

log = get_logger("ml.evaluate")
ARTIFACTS = ROOT / "ml" / "artifacts"


def main() -> None:
    configure_logging()
    ckpt = torch.load(ARTIFACTS / "model.pt", map_location="cpu", weights_only=False)
    window = ckpt["config"].get("window", WINDOW)
    scalers = ckpt["scalers"]

    model = LSTMForecaster(hidden=ckpt["config"].get("hidden", 32))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    series = load_series()
    per_series = []
    all_t, all_p, all_bt, all_bp = [], [], [], []

    for (region, category), values in series.items():
        key = f"{region}|{category}"
        mu, sd = scalers[key]["mean"], scalers[key]["std"]
        arr = np.asarray(values, dtype=np.float32)
        scaled = (arr - mu) / sd
        total_windows = len(arr) - window
        start = max(0, total_windows - VAL_PERIODS)

        truths, preds, btruths, bpreds = [], [], [], []
        with torch.no_grad():
            for s in range(start, total_windows):
                x = torch.tensor(scaled[s : s + window], dtype=torch.float32).reshape(1, window, 1)
                preds.append(float(model(x).item()) * sd + mu)
                truths.append(float(arr[s + window]))
                bpreds.append(float(arr[s + window - 1]))   # last-value baseline
                btruths.append(float(arr[s + window]))

        m = metrics_from(np.asarray(truths), np.asarray(preds), key)
        per_series.append({"series": key, "mae": m.mae, "rmse": m.rmse, "mape": m.mape})
        all_t += truths
        all_p += preds
        all_bt += btruths
        all_bp += bpreds

    overall = metrics_from(np.asarray(all_t), np.asarray(all_p), "lstm")
    baseline = metrics_from(np.asarray(all_bt), np.asarray(all_bp), "last_value")
    improvement = (baseline.mae - overall.mae) / baseline.mae * 100

    report = {
        "model_version": "lstm-weekly-v1",
        "n_series": len(series),
        "validation_weeks_per_series": VAL_PERIODS,
        "overall": {
            "lstm": overall.__dict__,
            "baseline": baseline.__dict__,
            "improvement_pct": round(improvement, 2),
        },
        "per_series": sorted(per_series, key=lambda r: r["mae"]),
    }
    (ARTIFACTS / "eval_report.json").write_text(json.dumps(report, indent=2))

    print("\nPer-series MAE (best -> worst):")
    for row in report["per_series"]:
        print(f"  {row['series']:>22}  MAE={row['mae']:>9.2f}  RMSE={row['rmse']:>9.2f}  MAPE={row['mape']:>6.1f}%")
    print(f"\nOverall LSTM  MAE={overall.mae}  RMSE={overall.rmse}  MAPE={overall.mape}%")
    print(f"Baseline      MAE={baseline.mae}  RMSE={baseline.rmse}  MAPE={baseline.mape}%")
    print(f"Improvement   {improvement:.1f}% MAE reduction vs baseline\n")
    log.info("eval report written", path=str(ARTIFACTS / "eval_report.json"),
             lstm_mae=overall.mae, baseline_mae=baseline.mae,
             improvement_pct=round(improvement, 2))


if __name__ == "__main__":
    main()
