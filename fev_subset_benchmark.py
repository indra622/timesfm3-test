#!/usr/bin/env python3
"""Reproduce a small multivariate subset of Google's TimesFM 3 FEV run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import fev
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
TIMESFM_ROOT = PROJECT_ROOT / "upstream-timesfm"
FEV_ROOT = PROJECT_ROOT / "upstream-fev"
NOTEBOOK_PATH = (
    TIMESFM_ROOT
    / "timesfm3-usage"
    / "benchmarks"
    / "fev_bench"
    / "fev_bench_timesfm3.ipynb"
)
OFFICIAL_RESULTS_PATH = NOTEBOOK_PATH.parent / "fev_bench_results.csv"
TASKS_PATH = FEV_ROOT / "benchmarks" / "fev_bench" / "tasks.yaml"
DEFAULT_TASKS = ("ETT_1W", "uci_air_quality_1D", "gvar")
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "multivariate" / "fev_subset_external.json"
METRICS = ("SQL", "MASE", "WAPE", "WQL")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def load_official_model_class() -> type:
    """Load cells 1-3 of Google's pinned notebook without running its 100 tasks."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "__name__": "timesfm3_official_fev_wrapper",
        "__file__": str(NOTEBOOK_PATH),
    }
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if len(code_cells) < 3:
        raise RuntimeError("Official FEV notebook no longer contains the expected wrapper cells")
    for cell in code_cells[:3]:
        exec(  # noqa: S102 - executes only the pinned local official notebook
            compile("".join(cell["source"]), str(NOTEBOOK_PATH), "exec"), namespace
        )
    return namespace["TimesFM3FEVModel"]


def select_tasks(task_names: tuple[str, ...]) -> list[fev.Task]:
    benchmark = fev.Benchmark.from_yaml(str(TASKS_PATH))
    by_name = {task.task_name: task for task in benchmark.tasks}
    missing = sorted(set(task_names) - set(by_name))
    if missing:
        raise ValueError(f"Unknown FEV tasks: {missing}")
    tasks = [by_name[name] for name in task_names]
    for task in tasks:
        if len(task.target_columns) <= 1:
            raise ValueError(f"Task is not multivariate: {task.task_name}")
    return tasks


def metric_comparison(local: dict[str, Any], official: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in METRICS:
        local_value = float(local[metric])
        official_value = float(official[metric])
        result[metric] = {
            "local": local_value,
            "official": official_value,
            "absolute_delta": local_value - official_value,
            "relative_delta_pct": 100.0 * (local_value - official_value) / official_value,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_names = tuple(args.tasks)
    official_results = pd.read_csv(OFFICIAL_RESULTS_PATH).set_index("task_name")
    tasks = select_tasks(task_names)
    model_class = load_official_model_class()
    model = model_class(
        checkpoint_path="google/timesfm-3.0-pytorch",
        min_batch=1,
        max_batch=16,
        per_core_batch_size=4,
        max_context_length=15360,
        device=args.device,
    )
    _ = model.forecaster

    started_at = time.time()
    rows: list[dict[str, Any]] = []
    for task in tasks:
        if task.task_name not in official_results.index:
            raise ValueError(f"No Google result exists for {task.task_name}")
        task_started_at = time.time()
        predictions = model.fit_predict(task)
        summary = task.evaluation_summary(
            predictions,
            model_name="TimesFM-3-local-MPS",
            training_time_s=0.0,
            inference_time_s=model.inference_time,
            trained_on_this_dataset=False,
            extra_info={
                "model_class": "TimesFM-3",
                "device": args.device,
                "use_variate_attention": True,
            },
        )
        official = official_results.loc[task.task_name]
        official_fingerprint = str(official["dataset_fingerprint"])
        rows.append(
            {
                "task_name": task.task_name,
                "target_count": len(task.target_columns),
                "horizon": task.horizon,
                "num_windows": task.num_windows,
                "num_forecasts": int(summary["num_forecasts"]),
                "dataset_fingerprint": summary["dataset_fingerprint"],
                "official_dataset_fingerprint": official_fingerprint,
                "dataset_fingerprint_matches_official": (
                    summary["dataset_fingerprint"] == official_fingerprint
                ),
                "metrics": metric_comparison(summary, official),
                "local_wall_time_s": time.time() - task_started_at,
                "local_reported_inference_time_s": float(summary["inference_time_s"]),
                "official_cuda_inference_time_s": float(official["inference_time_s"]),
            }
        )

    abs_relative_sql = [abs(row["metrics"]["SQL"]["relative_delta_pct"]) for row in rows]
    output = {
        "benchmark": "AutoGluon FEV-Bench multivariate subset",
        "scope": {
            "task_count": len(rows),
            "task_names": list(task_names),
            "selection_rule": "Three lightweight multivariate tasks from different domains, fixed before execution",
            "not_full_benchmark": True,
        },
        "provenance": {
            "timesfm_commit": git_revision(TIMESFM_ROOT),
            "fev_tasks_commit": git_revision(FEV_ROOT),
            "fev_runtime_version": fev.__version__,
            "official_notebook_sha256": sha256(NOTEBOOK_PATH),
            "official_results_sha256": sha256(OFFICIAL_RESULTS_PATH),
            "tasks_yaml_sha256": sha256(TASKS_PATH),
        },
        "runtime": {
            "device": args.device,
            "wall_time_s": time.time() - started_at,
        },
        "aggregate": {
            "mean_absolute_sql_relative_delta_pct": float(np.mean(abs_relative_sql)),
            "max_absolute_sql_relative_delta_pct": float(np.max(abs_relative_sql)),
            "dataset_fingerprint_match_count": sum(
                row["dataset_fingerprint_matches_official"] for row in rows
            ),
        },
        "tasks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
