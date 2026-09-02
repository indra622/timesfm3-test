#!/usr/bin/env python3
"""Benchmark TimesFM 3 on small, auditable slices of two real datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator

from run_experiment import MODEL_ID, choose_device, git_revision

plt.switch_backend("Agg")


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "real"


@dataclass(frozen=True)
class BenchmarkDataset:
  name: str
  values: np.ndarray
  target_names: list[str]
  context_length: int
  horizon: int
  season_length: int
  past_only_covariates: np.ndarray | None
  past_future_covariates: np.ndarray | None
  metadata: dict[str, Any]

  @property
  def history(self) -> np.ndarray:
    return self.values[:, : self.context_length]

  @property
  def future(self) -> np.ndarray:
    return self.values[:, self.context_length :]

  def validate(self) -> None:
    expected_length = self.context_length + self.horizon
    if self.values.shape != (len(self.target_names), expected_length):
      raise ValueError("target shape does not match names or forecast window")
    if not np.isfinite(self.values).all():
      raise ValueError("targets contain non-finite values")
    if self.past_only_covariates is not None:
      if self.past_only_covariates.shape[1] != self.context_length:
        raise ValueError("past-only covariates must end at the forecast origin")
      if not np.isfinite(self.past_only_covariates).all():
        raise ValueError("past-only covariates contain non-finite values")
    if self.past_future_covariates is not None:
      if self.past_future_covariates.shape[1] != expected_length:
        raise ValueError("past-future covariates must cover context and horizon")
      if not np.isfinite(self.past_future_covariates).all():
        raise ValueError("past-future covariates contain non-finite values")


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def seasonal_naive(history: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
  if history.ndim != 2:
    raise ValueError("history must have shape (targets, time)")
  if history.shape[1] < season_length:
    raise ValueError("history is shorter than one seasonal cycle")
  indices = np.arange(horizon) % season_length
  return history[:, -season_length:][:, indices]


def point_metrics(
  actual: np.ndarray, predicted: np.ndarray, history: np.ndarray
) -> dict[str, Any]:
  if actual.shape != predicted.shape:
    raise ValueError("actual and predicted shapes differ")
  error = predicted.astype(np.float64) - actual.astype(np.float64)
  absolute_error = np.abs(error)
  squared_scale = np.mean(np.diff(history.astype(np.float64), axis=1) ** 2, axis=1)
  squared_scale = np.maximum(squared_scale, 1e-12)
  rmsse = np.sqrt(np.mean(error**2, axis=1) / squared_scale)
  return {
    "mae": float(np.mean(absolute_error)),
    "rmse": float(np.sqrt(np.mean(error**2))),
    "wape": float(
      np.sum(absolute_error) / np.maximum(np.sum(np.abs(actual)), 1e-12)
    ),
    "mean_rmsse": float(np.mean(rmsse)),
    "per_target_mae": np.mean(absolute_error, axis=1).tolist(),
    "per_target_rmsse": rmsse.tolist(),
  }


def interval_metrics(actual: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
  if quantiles.shape[:2] != actual.shape or quantiles.shape[-1] != 9:
    raise ValueError("expected quantiles with shape (targets, horizon, 9)")
  lower = quantiles[:, :, 0]
  upper = quantiles[:, :, -1]
  return {
    "p10_p90_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
    "p10_p90_mean_width": float(np.mean(upper - lower)),
  }


def load_m5(
  raw_root: Path = RAW_ROOT / "m5",
  target_count: int = 16,
  context_length: int = 512,
  horizon: int = 28,
) -> BenchmarkDataset:
  sales_path = raw_root / "sales_train_evaluation.csv"
  calendar_path = raw_root / "calendar.csv"
  prices_path = raw_root / "sell_prices.csv"
  for path in (sales_path, calendar_path, prices_path):
    if not path.is_file():
      raise FileNotFoundError(f"missing M5 file: {path}")

  last_day = 1941
  first_day = last_day - context_length - horizon + 1
  split_day = last_day - horizon
  window_days = [f"d_{day}" for day in range(first_day, last_day + 1)]
  context_days = [f"d_{day}" for day in range(first_day, split_day + 1)]
  metadata_columns = ["id", "item_id", "dept_id", "cat_id", "store_id"]
  sales = pd.read_csv(sales_path, usecols=metadata_columns + window_days)
  candidates = sales[(sales["store_id"] == "CA_1") & (sales["dept_id"] == "FOODS_3")]
  totals = candidates[context_days].sum(axis=1).rename("context_total")
  selected = (
    candidates.join(totals)
    .sort_values(["context_total", "id"], ascending=[False, True])
    .head(target_count)
  )
  if len(selected) != target_count:
    raise ValueError("M5 slice contains fewer targets than requested")

  values = selected[window_days].to_numpy(dtype=np.float32)
  item_ids = selected["item_id"].tolist()
  target_names = selected["id"].tolist()

  calendar = pd.read_csv(calendar_path).set_index("d").loc[window_days]
  weekday_angle = 2 * np.pi * (calendar["wday"].to_numpy() - 1) / 7
  event_any = (
    calendar["event_name_1"].notna() | calendar["event_name_2"].notna()
  ).to_numpy(dtype=np.float32)
  calendar_covariates = np.stack(
    [
      np.sin(weekday_angle),
      np.cos(weekday_angle),
      event_any,
      calendar["snap_CA"].to_numpy(dtype=np.float32),
    ]
  ).astype(np.float32)

  price_parts = []
  wanted_items = set(item_ids)
  for chunk in pd.read_csv(prices_path, chunksize=500_000):
    price_parts.append(
      chunk[(chunk["store_id"] == "CA_1") & chunk["item_id"].isin(wanted_items)]
    )
  price_table = pd.concat(price_parts, ignore_index=True)
  weeks = calendar["wm_yr_wk"].to_numpy()
  price_covariates = []
  for item_id in item_ids:
    item_prices = price_table[price_table["item_id"] == item_id]
    price_by_week = item_prices.groupby("wm_yr_wk")["sell_price"].last()
    aligned = price_by_week.reindex(weeks).ffill().bfill()
    if aligned.isna().any():
      raise ValueError(f"price series is incomplete for {item_id}")
    context_median = float(np.median(aligned.iloc[:context_length]))
    normalized = aligned.to_numpy(dtype=np.float32) / max(context_median, 1e-6) - 1
    price_covariates.append(normalized)

  dataset = BenchmarkDataset(
    name="m5_ca1_foods3_top16",
    values=values,
    target_names=target_names,
    context_length=context_length,
    horizon=horizon,
    season_length=7,
    past_only_covariates=None,
    past_future_covariates=np.vstack(
      [calendar_covariates, np.stack(price_covariates)]
    ).astype(np.float32),
    metadata={
      "source": "M5 Forecasting - Accuracy; downloaded from Zenodo record 10203108",
      "license_boundary": "Subject to the M5 competition rules",
      "store": "CA_1",
      "department": "FOODS_3",
      "selection": "top 16 by context-only unit sales, tie-broken by id",
      "window_start": str(calendar.iloc[0]["date"]),
      "forecast_start": str(calendar.iloc[context_length]["date"]),
      "forecast_end": str(calendar.iloc[-1]["date"]),
      "covariates": [
        "weekday_sin",
        "weekday_cos",
        "event_any",
        "snap_CA",
        *[f"relative_price:{item_id}" for item_id in item_ids],
      ],
      "file_sha256": {
        path.name: file_sha256(path)
        for path in (calendar_path, sales_path, prices_path)
      },
    },
  )
  dataset.validate()
  return dataset


def load_beijing(
  raw_root: Path = RAW_ROOT / "beijing",
  target_count: int = 4,
  context_length: int = 336,
  horizon: int = 24,
) -> BenchmarkDataset:
  station_root = raw_root / "stations" / "PRSA_Data_20130301-20170228"
  paths = sorted(station_root.glob("*.csv"))
  if len(paths) != 12:
    raise FileNotFoundError("expected 12 extracted Beijing station CSV files")

  frames: dict[str, pd.DataFrame] = {}
  missing_counts: dict[str, int] = {}
  for path in paths:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame[["year", "month", "day", "hour"]])
    station = str(frame["station"].iloc[0])
    frames[station] = frame.set_index("timestamp")
    missing_counts[station] = int(frame["PM2.5"].isna().sum())

  selected_stations = [
    station
    for station, _ in sorted(missing_counts.items(), key=lambda item: (item[1], item[0]))[
      :target_count
    ]
  ]
  targets = pd.concat(
    [frames[station]["PM2.5"].rename(station) for station in selected_stations],
    axis=1,
  )

  stop = None
  for candidate_stop in range(len(targets), context_length + horizon, -1):
    future = targets.iloc[candidate_stop - horizon : candidate_stop]
    if future.notna().all().all():
      stop = candidate_stop
      break
  if stop is None:
    raise ValueError("no complete Beijing holdout window was found")

  start = stop - context_length - horizon
  history_frame = targets.iloc[start : stop - horizon].interpolate(
    method="linear", limit_direction="both"
  )
  future_frame = targets.iloc[stop - horizon : stop]
  if history_frame.isna().any().any() or future_frame.isna().any().any():
    raise ValueError("Beijing target window still contains missing values")
  values = np.concatenate(
    [history_frame.to_numpy().T, future_frame.to_numpy().T], axis=1
  ).astype(np.float32)

  weather_names = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
  past_weather = []
  history_index = history_frame.index
  for feature in weather_names:
    station_weather = pd.concat(
      [frames[station][feature].rename(station) for station in selected_stations],
      axis=1,
    )
    mean_weather = station_weather.mean(axis=1).loc[history_index]
    mean_weather = mean_weather.interpolate(method="linear", limit_direction="both")
    past_weather.append(mean_weather.to_numpy(dtype=np.float32))

  full_index = targets.index[start:stop]
  hour_angle = 2 * np.pi * full_index.hour.to_numpy() / 24
  weekday_angle = 2 * np.pi * full_index.dayofweek.to_numpy() / 7
  calendar_covariates = np.stack(
    [
      np.sin(hour_angle),
      np.cos(hour_angle),
      np.sin(weekday_angle),
      np.cos(weekday_angle),
    ]
  ).astype(np.float32)

  archive_path = raw_root / "beijing-multi-site-air-quality-data.zip"
  dataset = BenchmarkDataset(
    name="beijing_pm25_low_missing_4",
    values=values,
    target_names=selected_stations,
    context_length=context_length,
    horizon=horizon,
    season_length=24,
    past_only_covariates=np.stack(past_weather).astype(np.float32),
    past_future_covariates=calendar_covariates,
    metadata={
      "source": "UCI Beijing Multi-Site Air Quality, DOI 10.24432/C5RK5G",
      "license": "CC BY 4.0",
      "selection": "four stations with the fewest total PM2.5 missing values",
      "selected_stations": selected_stations,
      "selected_station_missing_counts": {
        station: missing_counts[station] for station in selected_stations
      },
      "context_start": str(history_frame.index[0]),
      "forecast_start": str(future_frame.index[0]),
      "forecast_end": str(future_frame.index[-1]),
      "hours_skipped_from_dataset_end": int(len(targets) - stop),
      "context_imputation": "linear interpolation within context only",
      "holdout_imputation": "none",
      "past_only_covariates": weather_names,
      "past_future_covariates": [
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
      ],
      "archive_sha256": file_sha256(archive_path),
    },
  )
  dataset.validate()
  return dataset


def run_dataset(
  forecaster: TimesFM3Evaluator,
  dataset: BenchmarkDataset,
  output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
  timings: dict[str, float] = {}

  started = time.perf_counter()
  univariate_outputs = list(
    forecaster.predict_batch(
      [row for row in dataset.history],
      horizon=dataset.horizon,
      return_quantiles=True,
      use_symmetric_averaging=False,
    )
  )
  timings["univariate_seconds"] = time.perf_counter() - started
  univariate = np.stack([output.forecast for output in univariate_outputs])
  univariate_quantiles = np.stack([output.quantiles for output in univariate_outputs])

  started = time.perf_counter()
  multivariate_output = next(
    forecaster.predict_batch(
      contexts=[dataset.history],
      horizon=dataset.horizon,
      return_quantiles=True,
      use_symmetric_averaging=False,
    )
  )
  timings["multivariate_seconds"] = time.perf_counter() - started

  covariate_arguments: dict[str, Any] = {}
  if dataset.past_only_covariates is not None:
    covariate_arguments["past_only_covariates"] = [dataset.past_only_covariates]
  if dataset.past_future_covariates is not None:
    covariate_arguments["past_future_covariates"] = [
      dataset.past_future_covariates
    ]
  started = time.perf_counter()
  covariate_output = next(
    forecaster.predict_batch(
      contexts=[dataset.history],
      horizon=dataset.horizon,
      return_quantiles=True,
      use_symmetric_averaging=False,
      **covariate_arguments,
    )
  )
  timings["multivariate_covariates_seconds"] = time.perf_counter() - started

  forecasts = {
    "seasonal_naive": seasonal_naive(
      dataset.history, dataset.horizon, dataset.season_length
    ),
    "timesfm3_univariate": univariate,
    "timesfm3_multivariate": multivariate_output.forecast,
    "timesfm3_multivariate_covariates": covariate_output.forecast,
  }
  quantiles = {
    "timesfm3_univariate": univariate_quantiles,
    "timesfm3_multivariate": multivariate_output.quantiles,
    "timesfm3_multivariate_covariates": covariate_output.quantiles,
  }
  metrics: dict[str, Any] = {}
  for name, forecast in forecasts.items():
    metrics[name] = point_metrics(dataset.future, forecast, dataset.history)
    if name in quantiles:
      metrics[name].update(interval_metrics(dataset.future, quantiles[name]))

  output_root.mkdir(parents=True, exist_ok=True)
  plot_path = output_root / f"{dataset.name}.png"
  plot_dataset(dataset, forecasts, covariate_output.quantiles, plot_path)
  result = {
    "dataset": dataset.name,
    "targets": len(dataset.target_names),
    "target_names": dataset.target_names,
    "context_length": dataset.context_length,
    "horizon": dataset.horizon,
    "season_length": dataset.season_length,
    "metadata": dataset.metadata,
    "timing": timings,
    "metrics": metrics,
    "output_shapes": {
      "forecast": list(covariate_output.forecast.shape),
      "quantiles": list(covariate_output.quantiles.shape),
    },
    "artifacts": {"plot": str(plot_path)},
  }
  result_path = output_root / f"{dataset.name}.json"
  result["artifacts"]["results"] = str(result_path)
  result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  return result


def plot_dataset(
  dataset: BenchmarkDataset,
  forecasts: dict[str, np.ndarray],
  quantiles: np.ndarray,
  output_path: Path,
) -> None:
  colors = {
    "seasonal_naive": "#777777",
    "timesfm3_univariate": "#D55E00",
    "timesfm3_multivariate": "#CC79A7",
    "timesfm3_multivariate_covariates": "#0072B2",
  }
  show_count = min(4, len(dataset.target_names))
  figure, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
  history_window = min(dataset.context_length, max(4 * dataset.season_length, 56))
  history_x = np.arange(-history_window, 0)
  future_x = np.arange(dataset.horizon)
  for target_index, axis in enumerate(axes.flat):
    if target_index >= show_count:
      axis.axis("off")
      continue
    axis.plot(
      history_x,
      dataset.history[target_index, -history_window:],
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
    for name, forecast in forecasts.items():
      axis.plot(
        future_x,
        forecast[target_index],
        color=colors[name],
        linewidth=1.4,
        label=name,
      )
    axis.fill_between(
      future_x,
      quantiles[target_index, :, 0],
      quantiles[target_index, :, -1],
      color="#56B4E9",
      alpha=0.14,
      label="covariate p10-p90",
    )
    axis.axvline(0, color="#444444", linestyle="--", alpha=0.5)
    axis.set_title(dataset.target_names[target_index])
    axis.grid(alpha=0.2)
  axes[0, 0].legend(ncol=2, fontsize=8, loc="upper left")
  figure.suptitle(dataset.name)
  figure.tight_layout()
  figure.savefig(output_path, dpi=160)
  plt.close(figure)


def compact_summary(result: dict[str, Any]) -> dict[str, Any]:
  return {
    "dataset": result["dataset"],
    "targets": result["targets"],
    "metrics": {
      name: {
        "mae": values["mae"],
        "wape": values["wape"],
        "mean_rmsse": values["mean_rmsse"],
        **(
          {"p10_p90_coverage": values["p10_p90_coverage"]}
          if "p10_p90_coverage" in values
          else {}
        ),
      }
      for name, values in result["metrics"].items()
    },
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", choices=["both", "m5", "beijing"], default="both")
  parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
  parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
  args = parser.parse_args()

  loaders = {
    "m5": load_m5,
    "beijing": load_beijing,
  }
  selected = list(loaders) if args.dataset == "both" else [args.dataset]
  datasets = [loaders[name]() for name in selected]

  resolved_device = choose_device(args.device)
  started = time.perf_counter()
  forecaster = TimesFM3Evaluator(
    ModelConfig(
      checkpoint_path=MODEL_ID,
      per_core_batch_size=16,
      device=resolved_device,
    )
  )
  load_seconds = time.perf_counter() - started
  results = [run_dataset(forecaster, dataset, args.output_dir) for dataset in datasets]

  summary = {
    "model": {
      "id": MODEL_ID,
      "upstream_commit": git_revision(PROJECT_ROOT / "upstream-timesfm"),
      "license": "timesfm-non-commercial-license-v1.0",
    },
    "environment": {
      "device": resolved_device,
      "platform": platform.platform(),
      "python": platform.python_version(),
      "torch": torch.__version__,
      "mps_available": bool(torch.backends.mps.is_available()),
    },
    "model_load_seconds": load_seconds,
    "benchmarks": [compact_summary(result) for result in results],
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  summary_path = args.output_dir / "summary.json"
  summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
