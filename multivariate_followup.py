#!/usr/bin/env python3
"""Rolling-origin tests that isolate useful and misleading cross-series inputs."""

from __future__ import annotations

import argparse
import json
import math
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

from real_data_benchmarks import (
    file_sha256,
    interval_metrics,
    point_metrics,
    seasonal_naive,
)
from run_experiment import MODEL_ID, choose_device, git_revision

plt.switch_backend("Agg")


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "multivariate"
DCRNN_COMMIT = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"


@dataclass(frozen=True)
class RollingPanel:
    """A fixed target panel with matched related and control covariates."""

    name: str
    target_values: np.ndarray
    target_names: list[str]
    related_covariates: np.ndarray
    control_covariates: np.ndarray
    past_future_covariates: np.ndarray | None
    timestamps: list[str]
    origins: list[int]
    context_length: int
    horizon: int
    season_length: int
    metadata: dict[str, Any]
    zero_is_missing: bool = False

    def validate(self) -> None:
        target_count, total_length = self.target_values.shape
        if target_count != len(self.target_names):
            raise ValueError("target names do not match target values")
        if len(self.timestamps) != total_length:
            raise ValueError("timestamps do not match target values")
        if self.related_covariates.shape[0] != target_count:
            raise ValueError("related covariates do not match target count")
        if self.control_covariates.shape != self.related_covariates.shape:
            raise ValueError("control and related covariates must have the same shape")
        if self.related_covariates.shape[-1] != total_length:
            raise ValueError("past-only covariates do not cover the panel")
        if self.past_future_covariates is not None:
            if self.past_future_covariates.shape[0] != target_count:
                raise ValueError("past-future covariates do not match target count")
            if self.past_future_covariates.shape[-1] != total_length:
                raise ValueError("past-future covariates do not cover the panel")
        if not self.origins:
            raise ValueError("at least one forecast origin is required")
        for origin in self.origins:
            if origin < self.context_length or origin + self.horizon > total_length:
                raise ValueError(f"invalid forecast origin: {origin}")
        for array in (
            self.target_values,
            self.related_covariates,
            self.control_covariates,
        ):
            if not np.isfinite(array).all():
                raise ValueError("panel contains non-finite values")


def rolling_origins(
    context_length: int,
    horizon: int,
    num_windows: int,
    step: int,
) -> tuple[list[int], int]:
    if min(context_length, horizon, num_windows, step) <= 0:
        raise ValueError("rolling-origin parameters must be positive")
    origins = [context_length + index * step for index in range(num_windows)]
    total_length = origins[-1] + horizon
    return origins, total_length


def load_m5_cross_store(
    raw_root: Path = RAW_ROOT / "m5",
    item_count: int = 8,
    context_length: int = 512,
    horizon: int = 28,
    num_windows: int = 5,
    step: int = 28,
) -> RollingPanel:
    """Load the same M5 items across three California stores."""
    sales_path = raw_root / "sales_train_evaluation.csv"
    calendar_path = raw_root / "calendar.csv"
    prices_path = raw_root / "sell_prices.csv"
    for path in (sales_path, calendar_path, prices_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing M5 file: {path}")

    origins, total_length = rolling_origins(context_length, horizon, num_windows, step)
    last_day = 1941
    first_day = last_day - total_length + 1
    window_days = [f"d_{day}" for day in range(first_day, last_day + 1)]
    selection_days = window_days[:context_length]
    metadata_columns = ["id", "item_id", "dept_id", "cat_id", "store_id"]
    sales = pd.read_csv(sales_path, usecols=metadata_columns + window_days)
    stores = ["CA_1", "CA_2", "CA_3"]
    candidates = sales[
        sales["store_id"].isin(stores) & sales["dept_id"].eq("FOODS_3")
    ].copy()
    item_totals = (
        candidates.groupby("item_id")[selection_days]
        .sum()
        .sum(axis=1)
        .sort_values(ascending=False)
    )
    selected_items = sorted(
        item_totals.head(item_count).index,
        key=lambda item: (-float(item_totals.loc[item]), item),
    )
    selected = candidates[candidates["item_id"].isin(selected_items)].copy()
    selected["item_order"] = selected["item_id"].map(
        {item: index for index, item in enumerate(selected_items)}
    )
    selected["store_order"] = selected["store_id"].map(
        {store: index for index, store in enumerate(stores)}
    )
    selected = selected.sort_values(["item_order", "store_order"])
    expected_targets = item_count * len(stores)
    if len(selected) != expected_targets:
        raise ValueError("M5 cross-store slice is incomplete")

    target_values = selected[window_days].to_numpy(dtype=np.float32)
    target_names = [
        f"{item_id}:{store_id}"
        for item_id, store_id in zip(selected["item_id"], selected["store_id"])
    ]
    related_parts: list[np.ndarray] = []
    control_parts: list[np.ndarray] = []
    for item_index in range(item_count):
        item_slice = target_values[item_index * 3 : (item_index + 1) * 3]
        control_item_index = (item_index + item_count // 2) % item_count
        control_item_slice = target_values[
            control_item_index * 3 : (control_item_index + 1) * 3
        ]
        for store_index in range(3):
            related_parts.append(np.delete(item_slice, store_index, axis=0))
            control_parts.append(np.delete(control_item_slice, store_index, axis=0))
    related_covariates = np.stack(related_parts).astype(np.float32)
    control_covariates = np.stack(control_parts).astype(np.float32)

    calendar = pd.read_csv(calendar_path).set_index("d").loc[window_days]
    weekday_angle = 2 * np.pi * (calendar["wday"].to_numpy() - 1) / 7
    event_any = (
        calendar["event_name_1"].notna() | calendar["event_name_2"].notna()
    ).to_numpy(dtype=np.float32)
    weeks = calendar["wm_yr_wk"].to_numpy()

    wanted_items = set(selected_items)
    price_parts = []
    for chunk in pd.read_csv(prices_path, chunksize=500_000):
        price_parts.append(
            chunk[chunk["store_id"].isin(stores) & chunk["item_id"].isin(wanted_items)]
        )
    price_table = pd.concat(price_parts, ignore_index=True)
    target_covariates: list[np.ndarray] = []
    for row in selected.itertuples(index=False):
        store_id = str(row.store_id)
        item_id = str(row.item_id)
        item_prices = price_table[
            price_table["store_id"].eq(store_id) & price_table["item_id"].eq(item_id)
        ]
        aligned = (
            item_prices.groupby("wm_yr_wk")["sell_price"]
            .last()
            .reindex(weeks)
            .ffill()
            .bfill()
        )
        if aligned.isna().any():
            raise ValueError(f"price series is incomplete for {item_id}:{store_id}")
        context_median = float(np.median(aligned.iloc[:context_length]))
        relative_price = (
            aligned.to_numpy(dtype=np.float32) / max(context_median, 1e-6) - 1
        )
        target_covariates.append(
            np.stack(
                [
                    np.sin(weekday_angle),
                    np.cos(weekday_angle),
                    event_any,
                    calendar[f"snap_{store_id[:2]}"].to_numpy(dtype=np.float32),
                    relative_price,
                ]
            ).astype(np.float32)
        )

    panel = RollingPanel(
        name="m5_same8sku_ca3stores_rolling5",
        target_values=target_values,
        target_names=target_names,
        related_covariates=related_covariates,
        control_covariates=control_covariates,
        past_future_covariates=np.stack(target_covariates),
        timestamps=calendar["date"].astype(str).tolist(),
        origins=origins,
        context_length=context_length,
        horizon=horizon,
        season_length=7,
        metadata={
            "source": "M5 Forecasting - Accuracy; Zenodo record 10203108 transport",
            "license_boundary": "Subject to the M5 competition rules",
            "selection": "top 8 FOODS_3 SKUs by three-store aggregate sales in earliest context only",
            "selected_items": selected_items,
            "stores": stores,
            "related_covariates": "same SKU in the other two CA stores",
            "control_covariates": "a different selected SKU in the other two CA stores, paired by a fixed half-list rotation",
            "past_future_covariates": [
                "weekday_sin",
                "weekday_cos",
                "event_any",
                "store_snap",
                "target_relative_price",
            ],
            "window_start": str(calendar.iloc[0]["date"]),
            "final_forecast_end": str(calendar.iloc[-1]["date"]),
            "file_sha256": {
                path.name: file_sha256(path)
                for path in (calendar_path, sales_path, prices_path)
            },
        },
    )
    panel.validate()
    return panel


def haversine_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Return pairwise great-circle distance in kilometers."""
    lat = np.radians(latitudes.astype(np.float64))
    lon = np.radians(longitudes.astype(np.float64))
    delta_lat = lat[:, None] - lat[None, :]
    delta_lon = lon[:, None] - lon[None, :]
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(delta_lon / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def lowest_missing_indices(
    values: np.ndarray,
    selection_end: int,
    keep_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank sensors by prefix-only zero markers and retain a fixed fraction."""
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    zero_rate = np.mean(values[:, :selection_end] <= 0, axis=1)
    keep_count = max(1, math.floor(len(zero_rate) * keep_fraction))
    order = np.lexsort((np.arange(len(zero_rate)), zero_rate))
    return order[:keep_count], zero_rate


def choose_metr_sensor_groups(
    values: np.ndarray,
    sensor_ids: list[str],
    locations: pd.DataFrame,
    distances: pd.DataFrame,
    target_count: int = 16,
    companion_count: int = 4,
    selection_end: int | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray, dict[str, Any]]:
    """Choose a low-missing connected cluster and matched near/far companions."""
    if selection_end is None:
        selection_end = math.floor(values.shape[1] * 0.8)
    if not 0 < selection_end <= values.shape[1]:
        raise ValueError("selection_end must fall within the observed series")
    eligible, zero_rate = lowest_missing_indices(values, selection_end)
    if len(eligible) < target_count + companion_count:
        raise ValueError("not enough low-missing METR-LA sensors")

    id_to_index = {sensor_id: index for index, sensor_id in enumerate(sensor_ids)}
    road = np.full((len(sensor_ids), len(sensor_ids)), np.inf, dtype=np.float64)
    np.fill_diagonal(road, 0.0)
    for row in distances.itertuples(index=False):
        from_id = str(row[0])
        to_id = str(row[1])
        if from_id in id_to_index and to_id in id_to_index:
            road[id_to_index[from_id], id_to_index[to_id]] = float(row[2])
    symmetric_road = np.minimum(road, road.T)
    eligible_mask = np.zeros(len(sensor_ids), dtype=bool)
    eligible_mask[eligible] = True
    neighborhood_counts = np.sum(
        np.isfinite(symmetric_road[:, eligible])
        & (symmetric_road[:, eligible] <= 8_000),
        axis=1,
    )
    anchor_candidates = eligible[np.argsort(-neighborhood_counts[eligible])]
    anchor = int(anchor_candidates[0])
    ranked = np.argsort(symmetric_road[anchor])
    targets = [
        int(index)
        for index in ranked
        if eligible_mask[index] and np.isfinite(symmetric_road[anchor, index])
    ][:target_count]
    if len(targets) != target_count:
        raise ValueError("could not form a connected METR-LA target cluster")

    location_frame = locations.set_index("sensor_id").loc[sensor_ids]
    geo = haversine_matrix(
        location_frame["latitude"].to_numpy(),
        location_frame["longitude"].to_numpy(),
    )
    related: list[list[int]] = []
    controls: list[list[int]] = []
    for target in targets:
        near = [
            int(index)
            for index in np.argsort(symmetric_road[target])
            if index != target
            and eligible_mask[index]
            and np.isfinite(symmetric_road[target, index])
        ][:companion_count]
        far = [
            int(index)
            for index in np.argsort(-geo[target])
            if index != target and eligible_mask[index]
        ][:companion_count]
        if len(near) != companion_count or len(far) != companion_count:
            raise ValueError("could not form METR-LA companion groups")
        related.append(near)
        controls.append(far)

    metadata = {
        "selection_prefix_points": selection_end,
        "eligible_sensor_fraction": 0.8,
        "eligible_maximum_prefix_zero_rate": float(np.max(zero_rate[eligible])),
        "anchor_sensor": sensor_ids[anchor],
        "target_sensors": [sensor_ids[index] for index in targets],
        "related_sensor_ids": [
            [sensor_ids[index] for index in group] for group in related
        ],
        "control_sensor_ids": [
            [sensor_ids[index] for index in group] for group in controls
        ],
        "mean_related_road_distance_m": float(
            np.mean(
                [
                    symmetric_road[target, companion]
                    for target, group in zip(targets, related)
                    for companion in group
                ]
            )
        ),
        "mean_control_geo_distance_km": float(
            np.mean(
                [
                    geo[target, companion]
                    for target, group in zip(targets, controls)
                    for companion in group
                ]
            )
        ),
    }
    return targets, np.asarray(related), np.asarray(controls), metadata


def complete_metr_origins(
    target_values: np.ndarray,
    context_length: int,
    horizon: int,
    num_windows: int,
    step: int,
) -> list[int]:
    """Choose recent spaced origins whose target holdouts contain no zero markers."""
    origins: list[int] = []
    candidate = target_values.shape[1] - horizon
    while candidate >= context_length and len(origins) < num_windows:
        future = target_values[:, candidate : candidate + horizon]
        if np.all(future > 0):
            origins.append(candidate)
            candidate -= step
        else:
            candidate -= 1
    if len(origins) != num_windows:
        raise ValueError("could not find enough complete METR-LA forecast windows")
    return sorted(origins)


def load_metr_la(
    raw_root: Path = RAW_ROOT / "metr_la",
    target_count: int = 16,
    companion_count: int = 4,
    context_length: int = 2016,
    horizon: int = 12,
    num_windows: int = 10,
    step: int = 288,
    selection_end: int | None = None,
) -> RollingPanel:
    """Load a graph-selected METR-LA sensor cluster and far-sensor control."""
    data_path = raw_root / "metr-la.h5"
    locations_path = raw_root / "graph" / "graph_sensor_locations.csv"
    distances_path = raw_root / "graph" / "distances_la_2012.csv"
    for path in (data_path, locations_path, distances_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing METR-LA input: {path}")

    frame = pd.read_hdf(data_path)
    sensor_ids = [str(column) for column in frame.columns]
    values = frame.to_numpy(dtype=np.float32).T
    locations = pd.read_csv(locations_path, dtype={"sensor_id": str})
    distances = pd.read_csv(distances_path, dtype={"from": str, "to": str})
    target_indices, related_indices, control_indices, selection = (
        choose_metr_sensor_groups(
            values,
            sensor_ids,
            locations,
            distances,
            target_count=target_count,
            companion_count=companion_count,
            selection_end=selection_end,
        )
    )
    target_values = values[target_indices]
    related_covariates = np.stack([values[group] for group in related_indices]).astype(
        np.float32
    )
    control_covariates = np.stack([values[group] for group in control_indices]).astype(
        np.float32
    )
    origins = complete_metr_origins(
        target_values,
        context_length=context_length,
        horizon=horizon,
        num_windows=num_windows,
        step=step,
    )

    timestamp_index = pd.DatetimeIndex(frame.index)
    time_angle = (
        2
        * np.pi
        * (timestamp_index.hour.to_numpy() * 12 + timestamp_index.minute.to_numpy() / 5)
        / 288
    )
    weekday_angle = 2 * np.pi * timestamp_index.dayofweek.to_numpy() / 7
    calendar = np.stack(
        [
            np.sin(time_angle),
            np.cos(time_angle),
            np.sin(weekday_angle),
            np.cos(weekday_angle),
        ]
    ).astype(np.float32)
    past_future_covariates = np.repeat(calendar[None, :, :], target_count, axis=0)

    panel = RollingPanel(
        name=f"metr_la_graph{target_count}_rolling{num_windows}",
        target_values=target_values,
        target_names=selection["target_sensors"],
        related_covariates=related_covariates,
        control_covariates=control_covariates,
        past_future_covariates=past_future_covariates,
        timestamps=timestamp_index.astype(str).tolist(),
        origins=origins,
        context_length=context_length,
        horizon=horizon,
        season_length=288,
        zero_is_missing=True,
        metadata={
            "source": "METR-LA public file linked by the official DCRNN repository",
            "data_sha256": file_sha256(data_path),
            "dcrnn_commit": DCRNN_COMMIT,
            "locations_sha256": file_sha256(locations_path),
            "distances_sha256": file_sha256(distances_path),
            "interval": "5 minutes",
            "related_covariates": f"{companion_count} nearest eligible sensors by official road-network distance",
            "control_covariates": f"{companion_count} farthest eligible sensors by geographic distance",
            "context_zero_handling": "zero markers interpolated within each context only",
            "holdout_zero_handling": "origins require nonzero target holdouts",
            "past_future_covariates": [
                "time_of_day_sin",
                "time_of_day_cos",
                "weekday_sin",
                "weekday_cos",
            ],
            **selection,
        },
    )
    panel.validate()
    return panel


def interpolate_zero_markers(values: np.ndarray) -> np.ndarray:
    """Interpolate nonpositive traffic missing markers within history only."""
    result = values.astype(np.float32, copy=True)
    x = np.arange(result.shape[-1])
    for index, row in enumerate(result.reshape(-1, result.shape[-1])):
        valid = row > 0
        if not valid.any():
            raise ValueError("history row contains only zero markers")
        row[~valid] = np.interp(x[~valid], x[valid], row[valid])
        result.reshape(-1, result.shape[-1])[index] = row
    return result


def focus_forecasts(outputs: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    forecasts = []
    quantiles = []
    for output in outputs:
        forecast = np.asarray(output.forecast)
        quantile = np.asarray(output.quantiles)
        forecasts.append(forecast.reshape(-1, forecast.shape[-1])[0])
        quantiles.append(quantile.reshape(-1, *quantile.shape[-2:])[0])
    return np.stack(forecasts), np.stack(quantiles)


def predict_focus_batch(
    forecaster: TimesFM3Evaluator,
    history: np.ndarray,
    horizon: int,
    past_only: np.ndarray | None = None,
    past_future: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    outputs = list(
        forecaster.predict_batch(
            contexts=[row for row in history],
            horizon=horizon,
            past_only_covariates=(
                [row for row in past_only] if past_only is not None else None
            ),
            past_future_covariates=(
                [row for row in past_future] if past_future is not None else None
            ),
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )
    return focus_forecasts(outputs)


def metrics_for_forecast(
    actual: np.ndarray,
    forecast: np.ndarray,
    history: np.ndarray,
    quantiles: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics = point_metrics(actual, forecast, history)
    if quantiles is not None:
        metrics.update(interval_metrics(actual, quantiles))
    return metrics


def aggregate_origins(origin_results: list[dict[str, Any]]) -> dict[str, Any]:
    mode_names = list(origin_results[0]["metrics"])
    aggregated: dict[str, Any] = {}
    for mode in mode_names:
        metric_names = [
            name
            for name, value in origin_results[0]["metrics"][mode].items()
            if isinstance(value, float)
        ]
        aggregated[mode] = {
            name: {
                "mean": float(
                    np.mean(
                        [origin["metrics"][mode][name] for origin in origin_results]
                    )
                ),
                "std": float(
                    np.std(
                        [origin["metrics"][mode][name] for origin in origin_results],
                        ddof=1,
                    )
                    if len(origin_results) > 1
                    else 0.0
                ),
            }
            for name in metric_names
        }
        if mode != "timesfm3_univariate":
            deltas = np.asarray(
                [
                    origin["metrics"][mode]["mae"]
                    - origin["metrics"]["timesfm3_univariate"]["mae"]
                    for origin in origin_results
                ]
            )
            aggregated[mode]["paired_mae_vs_univariate"] = {
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "win_rate": float(np.mean(deltas < 0)),
            }
    return aggregated


def run_panel(
    forecaster: TimesFM3Evaluator,
    panel: RollingPanel,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    panel.validate()
    origin_results: list[dict[str, Any]] = []
    timing_totals: dict[str, float] = {}
    for origin_index, origin in enumerate(panel.origins):
        start = origin - panel.context_length
        history = panel.target_values[:, start:origin]
        actual = panel.target_values[:, origin : origin + panel.horizon]
        related = panel.related_covariates[:, :, start:origin]
        control = panel.control_covariates[:, :, start:origin]
        future_covariates = (
            panel.past_future_covariates[:, :, start : origin + panel.horizon]
            if panel.past_future_covariates is not None
            else None
        )
        if panel.zero_is_missing:
            history = interpolate_zero_markers(history)
            related = interpolate_zero_markers(related)
            control = interpolate_zero_markers(control)
        if not np.isfinite(actual).all() or (
            panel.zero_is_missing and np.any(actual <= 0)
        ):
            raise ValueError(f"invalid holdout at origin {origin}")

        forecasts: dict[str, np.ndarray] = {
            "seasonal_naive": seasonal_naive(
                history, panel.horizon, panel.season_length
            )
        }
        quantiles: dict[str, np.ndarray] = {}

        started = time.perf_counter()
        forecasts["timesfm3_univariate"], quantiles["timesfm3_univariate"] = (
            predict_focus_batch(forecaster, history, panel.horizon)
        )
        timing_totals["timesfm3_univariate"] = timing_totals.get(
            "timesfm3_univariate", 0.0
        ) + (time.perf_counter() - started)

        started = time.perf_counter()
        joint = next(
            forecaster.predict_batch(
                contexts=[history],
                horizon=panel.horizon,
                return_quantiles=True,
                use_symmetric_averaging=False,
            )
        )
        forecasts["timesfm3_joint_targets"] = np.asarray(joint.forecast)
        quantiles["timesfm3_joint_targets"] = np.asarray(joint.quantiles)
        timing_totals["timesfm3_joint_targets"] = timing_totals.get(
            "timesfm3_joint_targets", 0.0
        ) + (time.perf_counter() - started)

        for mode, past_only in (
            ("timesfm3_related_covariates", related),
            ("timesfm3_control_covariates", control),
        ):
            started = time.perf_counter()
            forecasts[mode], quantiles[mode] = predict_focus_batch(
                forecaster, history, panel.horizon, past_only=past_only
            )
            timing_totals[mode] = timing_totals.get(mode, 0.0) + (
                time.perf_counter() - started
            )

        if future_covariates is not None:
            mode = "timesfm3_related_plus_future"
            started = time.perf_counter()
            forecasts[mode], quantiles[mode] = predict_focus_batch(
                forecaster,
                history,
                panel.horizon,
                past_only=related,
                past_future=future_covariates,
            )
            timing_totals[mode] = timing_totals.get(mode, 0.0) + (
                time.perf_counter() - started
            )

        metrics = {
            mode: metrics_for_forecast(
                actual,
                forecast,
                history,
                quantiles.get(mode),
            )
            for mode, forecast in forecasts.items()
        }
        origin_results.append(
            {
                "origin_index": origin_index,
                "origin": origin,
                "forecast_start": panel.timestamps[origin],
                "forecast_end": panel.timestamps[origin + panel.horizon - 1],
                "metrics": metrics,
            }
        )
        print(
            json.dumps(
                {
                    "dataset": panel.name,
                    "origin": origin_index + 1,
                    "origins": len(panel.origins),
                    "mae": {mode: values["mae"] for mode, values in metrics.items()},
                }
            )
        )

    result = {
        "dataset": panel.name,
        "targets": len(panel.target_names),
        "target_names": panel.target_names,
        "context_length": panel.context_length,
        "horizon": panel.horizon,
        "season_length": panel.season_length,
        "num_windows": len(panel.origins),
        "metadata": panel.metadata,
        "timing_seconds": timing_totals,
        "origins": origin_results,
        "aggregate": aggregate_origins(origin_results),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / f"{panel.name}.json"
    plot_path = output_root / f"{panel.name}.png"
    result["artifacts"] = {
        "results": str(result_path),
        "plot": str(plot_path),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_rolling_mae(result, plot_path)
    return result


def plot_rolling_mae(result: dict[str, Any], output_path: Path) -> None:
    figure, (mae_axis, delta_axis) = plt.subplots(2, 1, figsize=(13, 9))
    origins = result["origins"]
    x = np.arange(1, len(origins) + 1)
    mode_names = list(origins[0]["metrics"])
    for mode in mode_names:
        mae = [origin["metrics"][mode]["mae"] for origin in origins]
        mae_axis.plot(x, mae, marker="o", linewidth=1.5, label=mode)
        if mode != "timesfm3_univariate":
            univariate = np.asarray(
                [origin["metrics"]["timesfm3_univariate"]["mae"] for origin in origins]
            )
            delta_axis.plot(
                x,
                np.asarray(mae) - univariate,
                marker="o",
                linewidth=1.5,
                label=mode,
            )
    mae_axis.set_ylabel("MAE")
    mae_axis.set_title(f"{result['dataset']} rolling-origin MAE")
    mae_axis.grid(alpha=0.2)
    mae_axis.legend(ncol=2, fontsize=8)
    delta_axis.axhline(0, color="#333333", linestyle="--", linewidth=1)
    delta_axis.set_xlabel("rolling origin")
    delta_axis.set_ylabel("MAE delta vs univariate")
    delta_axis.grid(alpha=0.2)
    delta_axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": result["dataset"],
        "targets": result["targets"],
        "num_windows": result["num_windows"],
        "aggregate": result["aggregate"],
        "artifacts": result["artifacts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["both", "m5", "metr-la"], default="both")
    parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    loaders = {
        "m5": load_m5_cross_store,
        "metr-la": load_metr_la,
    }
    selected = list(loaders) if args.dataset == "both" else [args.dataset]
    panels = [loaders[name]() for name in selected]
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
    results = [run_panel(forecaster, panel, args.output_dir) for panel in panels]
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
        "benchmarks": [compact_result(result) for result in results],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
