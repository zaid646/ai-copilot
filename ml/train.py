"""Train a demand-forecast model (PyTorch LSTM) vs a last-value baseline.

Task: given the last WINDOW weeks of weekly revenue for a (region, category)
series, predict the next week's revenue. 20 series (4 regions x 5 categories).

Weekly aggregation is used because daily revenue is too noisy at this scale.

Run:  uv run python -m ml.train
Artifacts: ml/artifacts/model.pt, ml/artifacts/metrics.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

from core.config import ROOT
from core.db import query
from core.logging import configure_logging, get_logger

log = get_logger("ml.train")

ARTIFACTS = ROOT / "ml" / "artifacts"
WINDOW = 8           # lookback weeks
HORIZON = 1          # predict next week
VAL_PERIODS = 20     # last N weeks of each series -> validation (one-step backtest)
EPOCHS = 300
HIDDEN = 32
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


@dataclass
class Metrics:
    name: str
    mae: float
    rmse: float
    mape: float


class LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def load_series() -> dict[tuple[str, str], list[float]]:
    """Weekly revenue series per (region_id, category)."""
    rows = query(
        """
        SELECT f.region_id, p.category, date_trunc('week', dd.full_date)::date AS wk,
               sum(f.revenue) AS revenue
        FROM fact_sales f
        JOIN dim_date dd   ON f.date_id = dd.date_id
        JOIN dim_product p ON f.product_id = p.product_id
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
    )
    series: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        series[(r["region_id"], r["category"])].append(float(r["revenue"]))
    return dict(series)


def build_windows(series: dict[tuple[str, str], list[float]]):
    xs, ys, keys, scalers = [], [], [], {}
    for key, values in series.items():
        arr = np.asarray(values, dtype=np.float32)
        n_train = len(arr) - VAL_PERIODS
        mu = float(arr[:n_train].mean())
        sd = float(arr[:n_train].std() or 1.0)
        scalers[key] = {"mean": mu, "std": sd}
        scaled = (arr - mu) / sd
        for i in range(len(scaled) - WINDOW):
            xs.append(scaled[i : i + WINDOW])
            ys.append(scaled[i + WINDOW])
            keys.append(key)
    X = torch.tensor(np.stack(xs), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.asarray(ys), dtype=torch.float32)
    return X, y, keys, scalers


def last_value_baseline(series: dict[tuple[str, str], list[float]]) -> tuple[np.ndarray, np.ndarray]:
    truths, preds = [], []
    for values in series.values():
        arr = np.asarray(values, dtype=np.float32)
        n_train = len(arr) - VAL_PERIODS
        for i in range(n_train - 1, len(arr) - 1):
            preds.append(arr[i])
            truths.append(arr[i + 1])
    return np.asarray(truths), np.asarray(preds)


def metrics_from(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> Metrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return Metrics(name=name, mae=round(mae, 3), rmse=round(rmse, 3), mape=round(mape, 3))


def main() -> None:
    configure_logging()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("training", device=device)

    series = load_series()
    log.info("series loaded", n_series=len(series))

    X, y, keys, scalers = build_windows(series)

    split_idx, cursor = {}, 0
    for k, vals in series.items():
        n = len(vals) - WINDOW
        n_train = max(1, n - VAL_PERIODS)
        split_idx[k] = (cursor, cursor + n_train, cursor + n)
        cursor += n

    tr = torch.zeros(len(keys), dtype=torch.bool)
    va = torch.zeros(len(keys), dtype=torch.bool)
    for k, (s, a, e) in split_idx.items():
        tr[s:a] = True
        va[a:e] = True

    Xtr, ytr = X[tr].to(device), y[tr].to(device)
    Xva, yva = X[va].to(device), y[va].to(device)

    model = LSTMForecaster().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva), yva).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0 or epoch == 1:
            log.info("epoch", epoch=epoch, train_loss=round(float(loss.item()), 4),
                     val_loss=round(val_loss, 4))

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    lstm_true, lstm_pred = [], []
    for k, (s, a, e) in split_idx.items():
        mu, sd = scalers[k]["mean"], scalers[k]["std"]
        with torch.no_grad():
            p = model(X[a:e].to(device)).cpu().numpy()
        t = y[a:e].numpy()
        lstm_true.append(t * sd + mu)
        lstm_pred.append(p * sd + mu)
    lstm_true, lstm_pred = np.concatenate(lstm_true), np.concatenate(lstm_pred)

    base_true, base_pred = last_value_baseline(series)

    lstm_m = metrics_from(lstm_true, lstm_pred, "lstm")
    base_m = metrics_from(base_true, base_pred, "last_value")
    improvement = (base_m.mae - lstm_m.mae) / base_m.mae * 100

    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {
            "window": WINDOW,
            "horizon": HORIZON,
            "hidden": HIDDEN,
            "granularity": "week",
        },
        "scalers": {f"{k[0]}|{k[1]}": v for k, v in scalers.items()},
        "metrics": {"lstm": asdict(lstm_m), "baseline": asdict(base_m),
                    "improvement_pct": round(improvement, 2)},
    }
    torch.save(checkpoint, ARTIFACTS / "model.pt")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(checkpoint["metrics"], indent=2))

    log.info("model saved", path=str(ARTIFACTS / "model.pt"))
    log.info("lstm metrics", **asdict(lstm_m))
    log.info("baseline metrics", **asdict(base_m))
    log.info("winner", model="lstm" if lstm_m.mae <= base_m.mae else "last_value",
             improvement_pct=round(improvement, 2))


if __name__ == "__main__":
    main()
