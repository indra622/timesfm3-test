#!/usr/bin/env python3
"""Compare TimesFM 3 and Chronos-2 zero-shot with DCRNN and STAEformer.

The two tracks intentionally remain separate: the foundation models receive no
METR-LA gradient updates, while the traffic models train on the first half of
the series and use the standard 12-step input / 12-step output protocol.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import pickle
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from metr_graph_followup import (
    compare_modes,
    dcrnn_adjacency,
    leakage_safe_selection_end,
)
from multivariate_followup import (
    OUTPUT_ROOT,
    PROJECT_ROOT,
    aggregate_origins,
    interpolate_zero_markers,
    load_metr_la,
    metrics_for_forecast,
)
from real_data_benchmarks import file_sha256
from run_experiment import choose_device, git_revision

plt.switch_backend("Agg")

CHRONOS_MODEL_ID = "amazon/chronos-2"
CHRONOS_VERSION = "2.3.1"
TORCH_MTS_COMMIT = "2db4de371584067160f9a37f1ae59495699b4a0a"
STAEFORMER_COMMIT = "fc49d39b2f1a8e3cf37b6289d7240680e1690f3f"
CONTEXT_LENGTH = 2016
INPUT_STEPS = 12
HORIZON = 12
NUM_WINDOWS = 40
STEP = 288
SEED = 20260902


@dataclass(frozen=True)
class SupervisedData:
    train: TensorDataset
    validation: TensorDataset
    test_inputs_dcrnn: torch.Tensor
    test_inputs_staeformer: torch.Tensor
    test_targets: torch.Tensor
    mean: float
    std: float
    train_end: int
    validation_end: int


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_panel() -> tuple[Any, pd.DataFrame, int]:
    raw_path = PROJECT_ROOT / "data" / "raw" / "metr_la" / "metr-la.h5"
    frame = pd.read_hdf(raw_path)
    selection_end = leakage_safe_selection_end(
        len(frame), CONTEXT_LENGTH, HORIZON, NUM_WINDOWS, STEP
    )
    panel = load_metr_la(
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        num_windows=NUM_WINDOWS,
        step=STEP,
        selection_end=selection_end,
    )
    if panel.origins[0] - panel.context_length < selection_end:
        raise ValueError("sensor selection overlaps the earliest evaluation context")
    return panel, frame, selection_end


def calendar_features(index: pd.DatetimeIndex, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    tod = ((index.hour * 60 + index.minute) / (24 * 60)).to_numpy(np.float32)
    dow = index.dayofweek.to_numpy(np.float32)
    return (
        np.repeat(tod[:, None], num_nodes, axis=1),
        np.repeat(dow[:, None], num_nodes, axis=1),
    )


def _window_arrays(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    origins: np.ndarray,
    mean: float,
    std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_nodes = values.shape[0]
    tod, dow = calendar_features(timestamps, num_nodes)
    x_dcrnn: list[np.ndarray] = []
    x_staeformer: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for origin in origins:
        speed = values[:, origin - INPUT_STEPS : origin].T
        scaled = (speed - mean) / std
        x_dcrnn.append(
            np.stack([scaled, tod[origin - INPUT_STEPS : origin]], axis=-1)
        )
        x_staeformer.append(
            np.stack(
                [
                    scaled,
                    tod[origin - INPUT_STEPS : origin],
                    dow[origin - INPUT_STEPS : origin],
                ],
                axis=-1,
            )
        )
        targets.append(values[:, origin : origin + HORIZON].T[..., None])
    return (
        np.asarray(x_dcrnn, dtype=np.float32),
        np.asarray(x_staeformer, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def prepare_supervised_data(
    panel: Any,
    timestamps: pd.DatetimeIndex,
    selection_end: int,
    validation_fraction: float = 0.2,
) -> SupervisedData:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    prefix = interpolate_zero_markers(panel.target_values[:, :selection_end])
    train_end = int(selection_end * (1 - validation_fraction))
    validation_end = selection_end
    mean = float(prefix[:, :train_end].mean())
    std = float(prefix[:, :train_end].std())
    if std <= 0:
        raise ValueError("training standard deviation must be positive")

    train_origins = np.arange(INPUT_STEPS, train_end - HORIZON + 1)
    validation_origins = np.arange(train_end, validation_end - HORIZON + 1)
    x_train_dcrnn, x_train_stae, y_train = _window_arrays(
        prefix, timestamps[:selection_end], train_origins, mean, std
    )
    x_val_dcrnn, x_val_stae, y_val = _window_arrays(
        prefix, timestamps[:selection_end], validation_origins, mean, std
    )

    test_dcrnn: list[np.ndarray] = []
    test_stae: list[np.ndarray] = []
    test_targets: list[np.ndarray] = []
    for origin in panel.origins:
        history = interpolate_zero_markers(
            panel.target_values[:, origin - INPUT_STEPS : origin]
        )
        speed = history.T
        scaled = (speed - mean) / std
        origin_index = timestamps[origin - INPUT_STEPS : origin]
        tod, dow = calendar_features(origin_index, len(panel.target_names))
        test_dcrnn.append(np.stack([scaled, tod], axis=-1))
        test_stae.append(np.stack([scaled, tod, dow], axis=-1))
        actual = panel.target_values[:, origin : origin + HORIZON].T[..., None]
        if np.any(actual <= 0):
            raise ValueError("evaluation target contains a zero missing marker")
        test_targets.append(actual)

    train = TensorDataset(
        torch.from_numpy(x_train_dcrnn),
        torch.from_numpy(x_train_stae),
        torch.from_numpy(y_train),
    )
    validation = TensorDataset(
        torch.from_numpy(x_val_dcrnn),
        torch.from_numpy(x_val_stae),
        torch.from_numpy(y_val),
    )
    return SupervisedData(
        train=train,
        validation=validation,
        test_inputs_dcrnn=torch.from_numpy(np.asarray(test_dcrnn, dtype=np.float32)),
        test_inputs_staeformer=torch.from_numpy(
            np.asarray(test_stae, dtype=np.float32)
        ),
        test_targets=torch.from_numpy(np.asarray(test_targets, dtype=np.float32)),
        mean=mean,
        std=std,
        train_end=train_end,
        validation_end=validation_end,
    )


def _chronos_frames(panel: Any, origin: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = origin - panel.context_length
    history = interpolate_zero_markers(panel.target_values[:, start:origin])
    timestamps = pd.DatetimeIndex(panel.timestamps[start:origin])
    wide = pd.DataFrame({"item_id": "metr-panel", "timestamp": timestamps})
    long_parts: list[pd.DataFrame] = []
    for index, name in enumerate(panel.target_names):
        wide[name] = history[index]
        long_parts.append(
            pd.DataFrame(
                {
                    "item_id": name,
                    "timestamp": timestamps,
                    "target": history[index],
                }
            )
        )
    return wide, pd.concat(long_parts, ignore_index=True)


def _chronos_output_array(
    frame: pd.DataFrame,
    names: list[str],
    key: str,
) -> np.ndarray:
    forecasts = []
    identity_column = "target_name" if key == "target_name" else "item_id"
    for name in names:
        rows = frame.loc[frame[identity_column].eq(name), "predictions"]
        if len(rows) != HORIZON:
            raise ValueError(f"unexpected Chronos output for {name}: {len(rows)}")
        forecasts.append(rows.to_numpy(np.float32))
    return np.asarray(forecasts, dtype=np.float32)


def run_chronos_track(panel: Any, device: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    from chronos import Chronos2Pipeline

    load_started = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL_ID,
        device_map=device,
        dtype=torch.float32,
    )
    timings = {"model_load": time.perf_counter() - load_started, "univariate": 0.0, "multivariate": 0.0}
    origin_results: list[dict[str, Any]] = []
    for index, origin in enumerate(panel.origins, start=1):
        wide, long = _chronos_frames(panel, origin)
        actual = panel.target_values[:, origin : origin + panel.horizon]
        history = interpolate_zero_markers(
            panel.target_values[:, origin - panel.context_length : origin]
        )

        started = time.perf_counter()
        univariate_df = pipeline.predict_df(
            long,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=panel.horizon,
            context_length=panel.context_length,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=32,
            cross_learning=False,
            freq="5min",
        )
        timings["univariate"] += time.perf_counter() - started
        univariate = _chronos_output_array(
            univariate_df, panel.target_names, key="item_id"
        )

        started = time.perf_counter()
        multivariate_df = pipeline.predict_df(
            wide,
            id_column="item_id",
            timestamp_column="timestamp",
            target=panel.target_names,
            prediction_length=panel.horizon,
            context_length=panel.context_length,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=32,
            cross_learning=False,
            freq="5min",
        )
        timings["multivariate"] += time.perf_counter() - started
        multivariate = _chronos_output_array(
            multivariate_df, panel.target_names, key="target_name"
        )
        origin_results.append(
            {
                "origin": origin,
                "timestamp": panel.timestamps[origin],
                "metrics": {
                    "chronos2_univariate": metrics_for_forecast(
                        actual, univariate, history
                    ),
                    "chronos2_multivariate": metrics_for_forecast(
                        actual, multivariate, history
                    ),
                },
            }
        )
        print(f"Chronos-2 {index}/{len(panel.origins)}", flush=True)
    return origin_results, timings


def _import_traffic_models() -> tuple[type, type]:
    root = PROJECT_ROOT / "upstream-torch-mts"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    dcrnn_module = importlib.import_module("models.DCRNN")
    staeformer_module = importlib.import_module("models.STAEformer")
    return dcrnn_module.DCRNN, staeformer_module.STAEformer


def build_traffic_models(
    panel: Any,
    dcrnn_device: torch.device,
    staeformer_device: torch.device,
    output_dir: Path,
) -> tuple[torch.nn.Module, torch.nn.Module, Path]:
    DCRNN, STAEformer = _import_traffic_models()
    distances = pd.read_csv(
        PROJECT_ROOT / "data" / "raw" / "metr_la" / "graph" / "distances_la_2012.csv",
        dtype={"from": str, "to": str},
    )
    adjacency = dcrnn_adjacency(distances, panel.target_names)
    cache_dir = output_dir / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_path = cache_dir / "metr_la_target16_adj.pkl"
    with graph_path.open("wb") as handle:
        pickle.dump(
            (
                panel.target_names,
                {name: index for index, name in enumerate(panel.target_names)},
                adjacency,
            ),
            handle,
        )

    dcrnn = DCRNN(
        num_nodes=len(panel.target_names),
        adj_path=str(graph_path),
        device=dcrnn_device,
        input_dim=2,
        output_dim=1,
        seq_len=INPUT_STEPS,
        horizon=HORIZON,
        rnn_units=64,
        num_rnn_layers=2,
        max_diffusion_step=2,
        filter_type="dual_random_walk",
        tf_decay_steps=2000,
        use_teacher_forcing=True,
    ).to(dcrnn_device)
    staeformer = STAEformer(
        num_nodes=len(panel.target_names),
        in_steps=INPUT_STEPS,
        out_steps=HORIZON,
        steps_per_day=288,
        days_per_week=7,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
    ).to(staeformer_device)
    return dcrnn, staeformer, graph_path


def _predict_batch(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    model_name: str,
    mean: float,
    std: float,
    device: torch.device,
) -> torch.Tensor:
    if model_name == "dcrnn":
        normalized = model(inputs.to(device))
    else:
        normalized = model(inputs.to(device))
    return normalized * std + mean


@torch.no_grad()
def validation_mae(
    model: torch.nn.Module,
    loader: DataLoader,
    model_name: str,
    mean: float,
    std: float,
    device: torch.device,
) -> float:
    model.eval()
    absolute_error = 0.0
    count = 0
    feature_index = 0 if model_name == "dcrnn" else 1
    for batch in loader:
        inputs = batch[feature_index]
        targets = batch[2].to(device)
        prediction = _predict_batch(model, inputs, model_name, mean, std, device)
        absolute_error += torch.abs(prediction - targets).sum().item()
        count += targets.numel()
    return absolute_error / count


def train_traffic_model(
    model: torch.nn.Module,
    model_name: str,
    data: SupervisedData,
    device: torch.device,
    max_epochs: int,
    early_stop: int,
    batch_size: int,
    checkpoint_path: Path,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        data.train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        data.validation, batch_size=batch_size, shuffle=False
    )
    feature_index = 0 if model_name == "dcrnn" else 1
    first_batch = next(iter(train_loader))
    if model_name == "dcrnn":
        with torch.no_grad():
            normalized_targets = (first_batch[2].to(device) - data.mean) / data.std
            model(first_batch[0].to(device), normalized_targets, 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, eps=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 30, 40, 50], gamma=0.1
        )
        clip_grad = 5.0
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, weight_decay=0.0003
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 30], gamma=0.1
        )
        clip_grad = None

    started = time.perf_counter()
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    batches_seen = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_absolute_error = 0.0
        train_count = 0
        for batch in train_loader:
            inputs = batch[feature_index].to(device)
            targets = batch[2].to(device)
            optimizer.zero_grad()
            if model_name == "dcrnn":
                normalized_targets = (targets - data.mean) / data.std
                normalized_prediction = model(
                    inputs, normalized_targets, batches_seen
                )
                batches_seen += 1
            else:
                normalized_prediction = model(inputs)
            prediction = normalized_prediction * data.std + data.mean
            loss = torch.mean(torch.abs(prediction - targets))
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_absolute_error += torch.abs(prediction.detach() - targets).sum().item()
            train_count += targets.numel()
        scheduler.step()
        train_mae = train_absolute_error / train_count
        val_mae = validation_mae(
            model, validation_loader, model_name, data.mean, data.std, device
        )
        history.append({"epoch": epoch, "train_mae": train_mae, "validation_mae": val_mae})
        print(
            f"{model_name} epoch {epoch}: train={train_mae:.4f} val={val_mae:.4f}",
            flush=True,
        )
        if val_mae < best_validation - 1e-5:
            best_validation = val_mae
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= early_stop:
                break
    if best_state is None:
        raise RuntimeError(f"{model_name} did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    return {
        "best_epoch": best_epoch,
        "best_validation_mae": best_validation,
        "epochs_run": len(history),
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "checkpoint": str(checkpoint_path),
    }


@torch.no_grad()
def evaluate_traffic_model(
    model: torch.nn.Module,
    model_name: str,
    data: SupervisedData,
    panel: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    model.eval()
    inputs = (
        data.test_inputs_dcrnn
        if model_name == "dcrnn"
        else data.test_inputs_staeformer
    )
    started = time.perf_counter()
    prediction = _predict_batch(
        model, inputs, model_name, data.mean, data.std, device
    ).cpu().numpy()[..., 0]
    inference_seconds = time.perf_counter() - started
    actual = data.test_targets.numpy()[..., 0]
    results: list[dict[str, Any]] = []
    mode = "dcrnn_supervised" if model_name == "dcrnn" else "staeformer_supervised"
    for index, origin in enumerate(panel.origins):
        history = interpolate_zero_markers(
            panel.target_values[:, origin - panel.context_length : origin]
        )
        results.append(
            {
                "origin": origin,
                "timestamp": panel.timestamps[origin],
                "metrics": {
                    mode: metrics_for_forecast(
                        actual[index].T, prediction[index].T, history
                    )
                },
            }
        )
    return results, inference_seconds


def merge_origin_metrics(
    base: list[dict[str, Any]], additions: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    merged = copy.deepcopy(base)
    for collection in additions:
        if [row["origin"] for row in collection] != [row["origin"] for row in merged]:
            raise ValueError("model outputs do not share identical origins")
        for target, source in zip(merged, collection):
            target["metrics"].update(source["metrics"])
    return merged


def load_timesfm_origin_metrics(panel: Any) -> list[dict[str, Any]]:
    path = OUTPUT_ROOT / "metr_la_graph16_rolling40.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["metadata"]["target_sensors"] != panel.target_names:
        raise ValueError("TimesFM artifact target sensors do not match")
    if [row["origin"] for row in result["origins"]] != panel.origins:
        raise ValueError("TimesFM artifact origins do not match")
    keep = {"timesfm3_univariate", "timesfm3_joint_targets", "seasonal_naive"}
    return [
        {
            "origin": row["origin"],
            "timestamp": panel.timestamps[row["origin"]],
            "metrics": {
                name: metrics
                for name, metrics in row["metrics"].items()
                if name in keep
            },
        }
        for row in result["origins"]
    ]


def comparison_summary(origins: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        ("chronos2_multivariate", "chronos2_univariate"),
        ("timesfm3_joint_targets", "timesfm3_univariate"),
        ("timesfm3_joint_targets", "chronos2_multivariate"),
        ("staeformer_supervised", "dcrnn_supervised"),
        ("dcrnn_supervised", "timesfm3_joint_targets"),
        ("staeformer_supervised", "timesfm3_joint_targets"),
    ]
    available = set(origins[0]["metrics"])
    return {
        f"{mode}_vs_{reference}": compare_modes(origins, mode, reference)
        for mode, reference in pairs
        if mode in available and reference in available
    }


def plot_comparison(result: dict[str, Any], path: Path) -> None:
    modes = list(result["aggregate"])
    labels = {
        "seasonal_naive": "Seasonal naive",
        "timesfm3_univariate": "TimesFM 3 UV",
        "timesfm3_joint_targets": "TimesFM 3 MV",
        "chronos2_univariate": "Chronos-2 UV",
        "chronos2_multivariate": "Chronos-2 MV",
        "dcrnn_supervised": "DCRNN trained",
        "staeformer_supervised": "STAEformer trained",
    }
    means = [result["aggregate"][mode]["mae"]["mean"] for mode in modes]
    stds = [result["aggregate"][mode]["mae"]["std"] for mode in modes]
    figure, axes = plt.subplots(2, 1, figsize=(15, 11), constrained_layout=True)
    x = np.arange(len(modes))
    colors = ["#9ca3af" if "naive" in mode else "#2563eb" if "timesfm" in mode else "#d97706" if "chronos" in mode else "#059669" for mode in modes]
    axes[0].bar(x, means, yerr=stds, capsize=5, color=colors)
    axes[0].set_xticks(x, [labels.get(mode, mode) for mode in modes], rotation=25, ha="right")
    axes[0].set_ylabel("MAE (mph)")
    axes[0].set_title("METR-LA 16 sensors × 40 origins")
    axes[0].grid(axis="y", alpha=0.25)
    for mode in modes:
        axes[1].plot(
            [row["metrics"][mode]["mae"] for row in result["origins"]],
            label=labels.get(mode, mode),
            linewidth=1.5,
        )
    axes[1].set_xlabel("Rolling origin")
    axes[1].set_ylabel("MAE (mph)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track", choices=["all", "zero-shot", "supervised"], default="all"
    )
    parser.add_argument(
        "--chronos-device", choices=["mps", "cuda", "cpu"], default="mps"
    )
    parser.add_argument(
        "--dcrnn-device", choices=["mps", "cuda", "cpu"], default="cpu"
    )
    parser.add_argument(
        "--staeformer-device", choices=["mps", "cuda", "cpu"], default="mps"
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    panel, frame, selection_end = build_panel()
    origins = load_timesfm_origin_metrics(panel)
    timings: dict[str, Any] = {}
    training: dict[str, Any] = {}
    additions: list[list[dict[str, Any]]] = []

    if args.track in {"all", "zero-shot"}:
        chronos_results, chronos_timings = run_chronos_track(
            panel, args.chronos_device
        )
        additions.append(chronos_results)
        timings["chronos2"] = chronos_timings

    if args.track in {"all", "supervised"}:
        supervised = prepare_supervised_data(
            panel, pd.DatetimeIndex(frame.index), selection_end
        )
        dcrnn_device = torch.device(choose_device(args.dcrnn_device))
        staeformer_device = torch.device(
            choose_device(args.staeformer_device)
        )
        dcrnn, staeformer, graph_path = build_traffic_models(
            panel, dcrnn_device, staeformer_device, args.output_dir
        )
        dcrnn_training = train_traffic_model(
            dcrnn,
            "dcrnn",
            supervised,
            dcrnn_device,
            args.max_epochs,
            args.early_stop,
            args.batch_size,
            args.output_dir / "checkpoints" / "dcrnn_target16.pt",
        )
        dcrnn_results, dcrnn_inference = evaluate_traffic_model(
            dcrnn, "dcrnn", supervised, panel, dcrnn_device
        )
        training["dcrnn"] = dcrnn_training
        timings["dcrnn_inference"] = dcrnn_inference
        additions.append(dcrnn_results)

        stae_training = train_traffic_model(
            staeformer,
            "staeformer",
            supervised,
            staeformer_device,
            args.max_epochs,
            args.early_stop,
            args.batch_size,
            args.output_dir / "checkpoints" / "staeformer_target16.pt",
        )
        stae_results, stae_inference = evaluate_traffic_model(
            staeformer,
            "staeformer",
            supervised,
            panel,
            staeformer_device,
        )
        training["staeformer"] = stae_training
        timings["staeformer_inference"] = stae_inference
        additions.append(stae_results)
        training["data"] = {
            "training_windows": len(supervised.train),
            "validation_windows": len(supervised.validation),
            "train_end": supervised.train_end,
            "validation_end": supervised.validation_end,
            "input_steps": INPUT_STEPS,
            "output_steps": HORIZON,
            "graph_path": str(graph_path),
        }

    merged = merge_origin_metrics(origins, additions)
    result = {
        "benchmark": "METR-LA target16 rolling40 model comparison",
        "protocol": {
            "selection_end": selection_end,
            "earliest_context_start": panel.origins[0] - panel.context_length,
            "target_sensors": panel.target_names,
            "origins": panel.origins,
            "zero_shot_context_length": panel.context_length,
            "horizon": panel.horizon,
            "supervised_input_steps": INPUT_STEPS,
            "tracks_are_not_one_training_budget": True,
        },
        "models": {
            "timesfm3": {
                "id": "google/timesfm-3.0-pytorch",
                "mode": "zero-shot",
            },
            "chronos2": {
                "id": CHRONOS_MODEL_ID,
                "package_version": CHRONOS_VERSION,
                "mode": "zero-shot",
            },
            "dcrnn": {
                "source": "Torch-MTS faithful PyTorch implementation",
                "torch_mts_commit": TORCH_MTS_COMMIT,
                "mode": "supervised",
            },
            "staeformer": {
                "source": "Torch-MTS by the STAEformer author",
                "torch_mts_commit": TORCH_MTS_COMMIT,
                "official_repo_commit": STAEFORMER_COMMIT,
                "mode": "supervised",
            },
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "data_sha256": file_sha256(
            PROJECT_ROOT / "data" / "raw" / "metr_la" / "metr-la.h5"
        ),
        "source_commits": {
            "torch_mts": git_revision(PROJECT_ROOT / "upstream-torch-mts"),
            "staeformer": git_revision(PROJECT_ROOT / "upstream-staeformer"),
        },
        "timings_seconds": timings,
        "training": training,
        "origins": merged,
        "aggregate": aggregate_origins(merged),
        "comparisons": comparison_summary(merged),
    }
    output_path = args.output_dir / "metr_model_comparison.json"
    plot_path = args.output_dir / "metr_model_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_comparison(result, plot_path)
    print(json.dumps({"result": str(output_path), "plot": str(plot_path), "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
