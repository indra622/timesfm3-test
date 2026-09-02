#!/usr/bin/env python3
"""Expand METR-LA origins and compare TimesFM 3 with graph-aware ridge baselines."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator

from multivariate_followup import (
    DCRNN_COMMIT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    aggregate_origins,
    interpolate_zero_markers,
    load_metr_la,
    metrics_for_forecast,
    plot_rolling_mae,
    run_panel,
)
from run_experiment import MODEL_ID, choose_device, git_revision

DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 36, 72, 288)
DEFAULT_WINDOWS = 40
DEFAULT_STEP = 288
BOOTSTRAP_RESAMPLES = 10_000


def leakage_safe_selection_end(
    total_length: int,
    context_length: int,
    horizon: int,
    num_windows: int,
    step: int,
) -> int:
    """Use at most the first half, before the earliest nominal context."""
    nominal_context_limit = (
        total_length
        - horizon
        - num_windows * step
        - context_length
    )
    selection_end = min(nominal_context_limit, math.floor(total_length * 0.5))
    if selection_end <= 0:
        raise ValueError("not enough history for leakage-safe sensor selection")
    return selection_end


def dcrnn_adjacency(
    distances: pd.DataFrame,
    sensor_ids: list[str],
    normalized_k: float = 0.1,
) -> np.ndarray:
    """Reproduce DCRNN's directed Gaussian distance adjacency."""
    if not 0 <= normalized_k < 1:
        raise ValueError("normalized_k must be in [0, 1)")
    id_to_index = {sensor_id: index for index, sensor_id in enumerate(sensor_ids)}
    distance_matrix = np.full(
        (len(sensor_ids), len(sensor_ids)), np.inf, dtype=np.float64
    )
    for row in distances.itertuples(index=False):
        from_id, to_id = str(row[0]), str(row[1])
        if from_id in id_to_index and to_id in id_to_index:
            distance_matrix[id_to_index[from_id], id_to_index[to_id]] = float(row[2])
    finite = distance_matrix[np.isfinite(distance_matrix)]
    if finite.size == 0 or float(np.std(finite)) == 0:
        raise ValueError("road distances do not define a usable adjacency")
    adjacency = np.exp(-np.square(distance_matrix / np.std(finite)))
    adjacency[adjacency < normalized_k] = 0.0
    return adjacency.astype(np.float32)


def random_walk(adjacency: np.ndarray) -> np.ndarray:
    """Return a row-normalized random-walk matrix."""
    row_sum = adjacency.sum(axis=1, keepdims=True)
    return np.divide(
        adjacency,
        row_sum,
        out=np.zeros_like(adjacency, dtype=np.float32),
        where=row_sum > 0,
    )


def dual_random_walk_supports(
    adjacency: np.ndarray, max_diffusion_step: int = 2
) -> list[np.ndarray]:
    """Create forward and reverse random-walk powers as in DCRNN's dual filter."""
    if max_diffusion_step <= 0:
        raise ValueError("max_diffusion_step must be positive")
    supports: list[np.ndarray] = []
    for base in (random_walk(adjacency), random_walk(adjacency.T)):
        current = base
        for _ in range(max_diffusion_step):
            supports.append(current.astype(np.float32))
            current = current @ base
    return supports


def lag_features(
    normalized_history: np.ndarray,
    target_index: int,
    sample_times: np.ndarray,
    lags: tuple[int, ...],
    supports: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Build own-lag features plus optional graph-diffused lag features."""
    columns: list[np.ndarray] = []
    graph_supports = supports or []
    for lag in lags:
        snapshot = normalized_history[:, sample_times - lag]
        columns.append(snapshot[target_index])
        for support in graph_supports:
            columns.append(support[target_index] @ snapshot)
    return np.column_stack(columns).astype(np.float64)


def solve_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    prediction_features: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit a standardized direct multi-horizon ridge regression."""
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    standardized = (features - feature_mean) / feature_std
    prediction_standardized = (prediction_features - feature_mean) / feature_std
    design = np.column_stack([np.ones(len(standardized)), standardized])
    prediction_design = np.column_stack(
        [np.ones(len(prediction_standardized)), prediction_standardized]
    )
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ targets,
    )
    return prediction_design @ coefficients


def direct_ridge_forecast(
    history: np.ndarray,
    target_indices: list[int],
    horizon: int,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    supports: list[np.ndarray] | None = None,
    alpha: float = 10.0,
) -> np.ndarray:
    """Forecast each target directly from context-only own and graph lag features."""
    if history.ndim != 2:
        raise ValueError("history must have shape (sensors, time)")
    if max(lags) + horizon >= history.shape[1]:
        raise ValueError("history is too short for the requested lags and horizon")
    means = history.mean(axis=1, keepdims=True)
    scales = history.std(axis=1, keepdims=True)
    scales[scales < 1e-6] = 1.0
    normalized = (history - means) / scales
    sample_times = np.arange(max(lags), history.shape[1] - horizon + 1)
    prediction_time = np.asarray([history.shape[1]])
    forecasts = []
    horizon_offsets = np.arange(horizon)
    for target_index in target_indices:
        features = lag_features(
            normalized, target_index, sample_times, lags, supports=supports
        )
        prediction_features = lag_features(
            normalized, target_index, prediction_time, lags, supports=supports
        )
        targets = normalized[
            target_index, sample_times[:, None] + horizon_offsets[None, :]
        ]
        normalized_forecast = solve_ridge(
            features, targets, prediction_features, alpha=alpha
        )[0]
        forecast = normalized_forecast * scales[target_index, 0] + means[target_index, 0]
        forecasts.append(np.maximum(forecast, 0.0))
    return np.asarray(forecasts, dtype=np.float32)


def moving_block_bootstrap_summary(
    deltas: np.ndarray,
    block_size: int = 7,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 20260902,
) -> dict[str, Any]:
    """Summarize paired deltas with a circular moving-block bootstrap interval."""
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("deltas must be a nonempty vector")
    block_size = min(block_size, len(values))
    blocks_per_sample = math.ceil(len(values) / block_size)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(resamples, blocks_per_sample))
    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(values)
    indices = indices.reshape(resamples, -1)[:, : len(values)]
    boot_means = values[indices].mean(axis=1)
    lower, upper = np.quantile(boot_means, [0.025, 0.975])
    return {
        "mean_delta": float(np.mean(values)),
        "std_delta": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "win_rate": float(np.mean(values < 0)),
        "moving_block_bootstrap_95pct_ci": [float(lower), float(upper)],
        "block_size_origins": block_size,
        "bootstrap_resamples": resamples,
    }


def compare_modes(
    origins: list[dict[str, Any]], mode: str, reference: str
) -> dict[str, Any]:
    deltas = np.asarray(
        [
            origin["metrics"][mode]["mae"]
            - origin["metrics"][reference]["mae"]
            for origin in origins
        ]
    )
    return {
        "mode": mode,
        "reference": reference,
        **moving_block_bootstrap_summary(deltas),
    }


def load_full_metr_graph(
    raw_root: Path,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    frame = pd.read_hdf(raw_root / "metr-la.h5")
    sensor_ids = [str(column) for column in frame.columns]
    values = frame.to_numpy(dtype=np.float32).T
    distances = pd.read_csv(
        raw_root / "graph" / "distances_la_2012.csv",
        dtype={"from": str, "to": str},
    )
    return values, sensor_ids, dcrnn_adjacency(distances, sensor_ids)


def add_graph_baselines(
    result: dict[str, Any],
    panel: Any,
    all_values: np.ndarray,
    sensor_ids: list[str],
    adjacency: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    id_to_index = {sensor_id: index for index, sensor_id in enumerate(sensor_ids)}
    target_indices = [id_to_index[name] for name in panel.target_names]
    graph_supports = dual_random_walk_supports(adjacency)
    permutation = np.random.default_rng(20260902).permutation(len(sensor_ids))
    shuffled_adjacency = adjacency[np.ix_(permutation, permutation)]
    shuffled_supports = dual_random_walk_supports(shuffled_adjacency)
    timings = {
        "ridge_autoregression": 0.0,
        "graph_diffusion_ridge": 0.0,
        "shuffled_graph_diffusion_ridge": 0.0,
    }

    for origin_result, origin in zip(result["origins"], panel.origins):
        start = origin - panel.context_length
        history_all = interpolate_zero_markers(all_values[:, start:origin])
        target_history = history_all[target_indices]
        actual = panel.target_values[:, origin : origin + panel.horizon]
        modes = (
            ("ridge_autoregression", None),
            ("graph_diffusion_ridge", graph_supports),
            ("shuffled_graph_diffusion_ridge", shuffled_supports),
        )
        for mode, supports in modes:
            started = time.perf_counter()
            forecast = direct_ridge_forecast(
                history_all,
                target_indices,
                panel.horizon,
                supports=supports,
                alpha=alpha,
            )
            timings[mode] += time.perf_counter() - started
            origin_result["metrics"][mode] = metrics_for_forecast(
                actual, forecast, target_history
            )

    result["timing_seconds"].update(timings)
    result["aggregate"] = aggregate_origins(result["origins"])
    result["comparisons"] = {
        "timesfm_joint_vs_univariate": compare_modes(
            result["origins"], "timesfm3_joint_targets", "timesfm3_univariate"
        ),
        "timesfm_related_vs_univariate": compare_modes(
            result["origins"],
            "timesfm3_related_covariates",
            "timesfm3_univariate",
        ),
        "timesfm_control_vs_univariate": compare_modes(
            result["origins"],
            "timesfm3_control_covariates",
            "timesfm3_univariate",
        ),
        "timesfm_related_plus_future_vs_univariate": compare_modes(
            result["origins"],
            "timesfm3_related_plus_future",
            "timesfm3_univariate",
        ),
        "timesfm_related_vs_control": compare_modes(
            result["origins"],
            "timesfm3_related_covariates",
            "timesfm3_control_covariates",
        ),
        "graph_vs_own_ridge": compare_modes(
            result["origins"], "graph_diffusion_ridge", "ridge_autoregression"
        ),
        "graph_vs_shuffled_graph": compare_modes(
            result["origins"],
            "graph_diffusion_ridge",
            "shuffled_graph_diffusion_ridge",
        ),
        "timesfm_joint_vs_graph": compare_modes(
            result["origins"], "timesfm3_joint_targets", "graph_diffusion_ridge"
        ),
    }
    result["metadata"]["graph_baseline"] = {
        "description": "Direct multi-horizon ridge on own lags plus DCRNN-style forward/reverse random-walk powers",
        "not_full_dcrnn": True,
        "lags": list(DEFAULT_LAGS),
        "max_diffusion_step": 2,
        "ridge_alpha": alpha,
        "shuffled_graph_seed": 20260902,
        "dcrnn_adjacency_normalized_k": 0.1,
    }
    result_path = Path(result["artifacts"]["results"])
    plot_path = Path(result["artifacts"]["plot"])
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_rolling_mae(result, plot_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-windows", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = PROJECT_ROOT / "data" / "raw" / "metr_la"
    frame = pd.read_hdf(raw_root / "metr-la.h5")
    context_length = 2016
    horizon = 12
    selection_end = leakage_safe_selection_end(
        len(frame), context_length, horizon, args.num_windows, args.step
    )
    panel = load_metr_la(
        raw_root=raw_root,
        context_length=context_length,
        horizon=horizon,
        num_windows=args.num_windows,
        step=args.step,
        selection_end=selection_end,
    )
    if selection_end > panel.origins[0] - panel.context_length:
        raise ValueError("sensor selection overlaps the earliest evaluation context")

    all_values, sensor_ids, adjacency = load_full_metr_graph(raw_root)
    resolved_device = choose_device(args.device)
    started = time.perf_counter()
    forecaster = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path=MODEL_ID,
            per_core_batch_size=16,
            device=resolved_device,
        )
    )
    model_load_seconds = time.perf_counter() - started
    result = run_panel(forecaster, panel, args.output_dir)
    result = add_graph_baselines(
        result,
        panel,
        all_values,
        sensor_ids,
        adjacency,
        alpha=args.ridge_alpha,
    )
    summary = {
        "model": {
            "id": MODEL_ID,
            "upstream_commit": git_revision(PROJECT_ROOT / "upstream-timesfm"),
            "license": "timesfm-non-commercial-license-v1.0",
        },
        "graph_reference": {
            "repository": "https://github.com/liyaguang/DCRNN",
            "commit": DCRNN_COMMIT,
            "baseline_is_full_dcrnn": False,
        },
        "environment": {
            "device": resolved_device,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "model_load_seconds": model_load_seconds,
        "dataset": result["dataset"],
        "targets": result["targets"],
        "num_windows": result["num_windows"],
        "aggregate": result["aggregate"],
        "comparisons": result["comparisons"],
        "artifacts": result["artifacts"],
    }
    summary_path = args.output_dir / "metr_graph_expanded_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
