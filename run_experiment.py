#!/usr/bin/env python3
"""Run a local TimesFM 3 synthetic forecasting experiment."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator

plt.switch_backend("Agg")


MODEL_ID = "google/timesfm-3.0-pytorch"
DEFAULT_CONTEXT = 256
DEFAULT_HORIZON = 64
SEASON_LENGTH = 7


@dataclass(frozen=True)
class SyntheticDataset:
  targets: np.ndarray
  promotion: np.ndarray
  foot_traffic: np.ndarray
  context_length: int
  horizon: int

  @property
  def history(self) -> np.ndarray:
    return self.targets[:, : self.context_length]

  @property
  def future(self) -> np.ndarray:
    return self.targets[:, self.context_length :]

  @property
  def past_promotion(self) -> np.ndarray:
    return self.promotion[: self.context_length]

  @property
  def future_promotion(self) -> np.ndarray:
    return self.promotion[self.context_length :]


def make_synthetic_dataset(
  context_length: int = DEFAULT_CONTEXT,
  horizon: int = DEFAULT_HORIZON,
  seed: int = 7,
) -> SyntheticDataset:
  """Create two related daily demand series with known promotion events."""
  if context_length < 2 * SEASON_LENGTH:
    raise ValueError("context_length must cover at least two seasonal cycles")
  if horizon < 1:
    raise ValueError("horizon must be positive")

  rng = np.random.default_rng(seed)
  size = context_length + horizon
  t = np.arange(size, dtype=np.float32)

  # Campaign days are irregular, so the univariate forecaster cannot infer the
  # future schedule from seasonality. The schedule is genuinely useful only
  # when supplied as a known-future covariate.
  promotion = (rng.random(size) < 0.16).astype(np.float32)
  weekly = np.sin(2 * np.pi * t / SEASON_LENGTH).astype(np.float32)
  foot_traffic = (
    180 + 0.10 * t + 22 * weekly + 30 * promotion + rng.normal(0, 4, size)
  ).astype(np.float32)

  demand_a = (
    92
    + 0.055 * t
    + 13 * weekly
    + 0.16 * foot_traffic
    + 28 * promotion
    + rng.normal(0, 2.0, size)
  )
  demand_b = (
    61
    + 0.035 * t
    + 8 * np.cos(2 * np.pi * (t - 1) / SEASON_LENGTH)
    + 0.10 * foot_traffic
    + 18 * promotion
    + rng.normal(0, 1.6, size)
  )

  return SyntheticDataset(
    targets=np.stack([demand_a, demand_b]).astype(np.float32),
    promotion=promotion,
    foot_traffic=foot_traffic,
    context_length=context_length,
    horizon=horizon,
  )


def seasonal_naive(history: np.ndarray, horizon: int) -> np.ndarray:
  """Repeat the last observed weekly cycle."""
  if history.ndim != 2:
    raise ValueError("history must have shape (variates, time)")
  if history.shape[1] < SEASON_LENGTH:
    raise ValueError("history is shorter than one seasonal cycle")
  indices = np.arange(horizon) % SEASON_LENGTH
  return history[:, -SEASON_LENGTH:][:, indices]


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
  """Return compact aggregate and per-target point-forecast metrics."""
  if actual.shape != predicted.shape:
    raise ValueError(f"shape mismatch: actual={actual.shape}, predicted={predicted.shape}")
  error = np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
  absolute_error = np.abs(error)
  denom = np.maximum(np.sum(np.abs(actual), axis=1), 1e-12)
  return {
    "mae": float(np.mean(absolute_error)),
    "rmse": float(np.sqrt(np.mean(np.square(error)))),
    "wape": float(np.sum(absolute_error) / np.maximum(np.sum(np.abs(actual)), 1e-12)),
    "per_target_mae": np.mean(absolute_error, axis=1).tolist(),
    "per_target_wape": (np.sum(absolute_error, axis=1) / denom).tolist(),
  }


def choose_device(requested: str) -> str:
  if requested != "auto":
    return requested
  if torch.backends.mps.is_available():
    return "mps"
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"


def git_revision(repo: Path) -> str | None:
  try:
    return subprocess.check_output(
      ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def run_experiment(
  output_dir: Path,
  device: str = "auto",
  context_length: int = DEFAULT_CONTEXT,
  horizon: int = DEFAULT_HORIZON,
  seed: int = 7,
) -> dict[str, Any]:
  dataset = make_synthetic_dataset(context_length, horizon, seed)
  resolved_device = choose_device(device)

  load_started = time.perf_counter()
  forecaster = TimesFM3Evaluator(
    ModelConfig(
      checkpoint_path=MODEL_ID,
      per_core_batch_size=2,
      device=resolved_device,
    )
  )
  load_seconds = time.perf_counter() - load_started

  inference_times: dict[str, float] = {}

  started = time.perf_counter()
  univariate_outputs = list(
    forecaster.predict_batch(
      [dataset.history[0], dataset.history[1]],
      horizon=horizon,
      return_quantiles=True,
      use_symmetric_averaging=False,
    )
  )
  inference_times["univariate_seconds"] = time.perf_counter() - started
  univariate = np.stack([item.forecast for item in univariate_outputs])

  started = time.perf_counter()
  multivariate_output = next(
    forecaster.predict_batch(
      contexts=[dataset.history],
      horizon=horizon,
      past_only_covariates=[dataset.foot_traffic[:context_length][None, :]],
      past_future_covariates=[dataset.promotion[None, :]],
      return_quantiles=True,
      use_symmetric_averaging=False,
    )
  )
  inference_times["multivariate_covariate_seconds"] = time.perf_counter() - started
  multivariate = multivariate_output.forecast

  counterfactual_promotion = dataset.promotion.copy()
  counterfactual_promotion[context_length:] = 0
  started = time.perf_counter()
  counterfactual_output = next(
    forecaster.predict_batch(
      contexts=[dataset.history],
      horizon=horizon,
      past_only_covariates=[dataset.foot_traffic[:context_length][None, :]],
      past_future_covariates=[counterfactual_promotion[None, :]],
      return_quantiles=False,
      use_symmetric_averaging=False,
    )
  )
  inference_times["counterfactual_seconds"] = time.perf_counter() - started
  counterfactual = counterfactual_output.forecast

  baseline = seasonal_naive(dataset.history, horizon)
  predictions = {
    "seasonal_naive": baseline,
    "timesfm3_univariate": univariate,
    "timesfm3_multivariate_covariates": multivariate,
  }
  metrics = {
    name: forecast_metrics(dataset.future, forecast)
    for name, forecast in predictions.items()
  }

  future_promo = dataset.future_promotion.astype(bool)
  promo_delta = multivariate - counterfactual
  promotion_response = {
    "scheduled_promotion_days": int(np.sum(future_promo)),
    "synthetic_true_lift_per_target": [32.8, 21.0],
    "mean_predicted_lift_on_promotion_days": float(
      np.mean(promo_delta[:, future_promo])
    ),
    "per_target_mean_predicted_lift": np.mean(
      promo_delta[:, future_promo], axis=1
    ).tolist(),
  }

  output_dir.mkdir(parents=True, exist_ok=True)
  plot_path = output_dir / "forecast.png"
  plot_forecast(dataset, predictions, multivariate_output.quantiles, plot_path)

  result = {
    "model": {
      "id": MODEL_ID,
      "upstream_commit": git_revision(Path(__file__).parent / "upstream-timesfm"),
      "license": "timesfm-non-commercial-license-v1.0",
    },
    "environment": {
      "device": resolved_device,
      "platform": platform.platform(),
      "python": platform.python_version(),
      "torch": torch.__version__,
      "mps_available": bool(torch.backends.mps.is_available()),
    },
    "data": {
      "context_length": context_length,
      "horizon": horizon,
      "targets": int(dataset.targets.shape[0]),
      "seed": seed,
    },
    "timing": {"load_seconds": load_seconds, **inference_times},
    "metrics": metrics,
    "promotion_response": promotion_response,
    "output_shapes": {
      "univariate_forecast": list(univariate.shape),
      "multivariate_forecast": list(multivariate.shape),
      "multivariate_quantiles": list(multivariate_output.quantiles.shape),
    },
    "artifacts": {"plot": str(plot_path)},
  }
  result_path = output_dir / "results.json"
  result["artifacts"]["results"] = str(result_path)
  result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  return result


def plot_forecast(
  dataset: SyntheticDataset,
  predictions: dict[str, np.ndarray],
  quantiles: np.ndarray,
  output_path: Path,
) -> None:
  history_window = min(56, dataset.context_length)
  start = dataset.context_length - history_window
  history_x = np.arange(start, dataset.context_length)
  future_x = np.arange(dataset.context_length, dataset.context_length + dataset.horizon)

  figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
  colors = {
    "seasonal_naive": "#777777",
    "timesfm3_univariate": "#D55E00",
    "timesfm3_multivariate_covariates": "#0072B2",
  }
  labels = ["Demand A", "Demand B"]
  for target_index, axis in enumerate(axes):
    axis.plot(
      history_x,
      dataset.targets[target_index, start : dataset.context_length],
      color="#222222",
      label="history",
    )
    axis.plot(
      future_x,
      dataset.future[target_index],
      color="#009E73",
      linewidth=2,
      label="actual",
    )
    for name, values in predictions.items():
      axis.plot(
        future_x,
        values[target_index],
        color=colors[name],
        linewidth=1.7,
        label=name,
      )
    axis.fill_between(
      future_x,
      quantiles[target_index, :, 0],
      quantiles[target_index, :, -1],
      color="#56B4E9",
      alpha=0.16,
      label="TimesFM 3 p10-p90",
    )
    for position in future_x[dataset.future_promotion.astype(bool)]:
      axis.axvspan(position - 0.45, position + 0.45, color="#F0E442", alpha=0.22)
    axis.set_title(labels[target_index])
    axis.set_ylabel("synthetic demand")
    axis.grid(alpha=0.2)

  axes[0].legend(ncol=3, fontsize=8, loc="upper left")
  axes[-1].set_xlabel("day (yellow = scheduled promotion)")
  figure.suptitle("TimesFM 3: univariate vs multivariate + known-future covariate")
  figure.tight_layout()
  figure.savefig(output_path, dpi=160)
  plt.close(figure)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
  parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
  parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
  args = parser.parse_args()

  result = run_experiment(
    output_dir=args.output_dir,
    device=args.device,
    context_length=args.context,
    horizon=args.horizon,
    seed=args.seed,
  )
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
