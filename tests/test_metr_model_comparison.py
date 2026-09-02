from __future__ import annotations

import numpy as np
import pandas as pd

from metr_model_comparison import (
    HORIZON,
    INPUT_STEPS,
    _window_arrays,
    build_panel,
    calendar_features,
    load_timesfm_origin_metrics,
    merge_origin_metrics,
)


def test_calendar_features_have_expected_ranges() -> None:
    index = pd.date_range("2026-01-04", periods=4, freq="5min")
    tod, dow = calendar_features(index, num_nodes=3)
    assert tod.shape == (4, 3)
    assert dow.shape == (4, 3)
    assert np.all((tod >= 0) & (tod < 1))
    assert np.all(dow == 6)


def test_window_arrays_preserve_shapes_and_raw_targets() -> None:
    values = np.arange(2 * 50, dtype=np.float32).reshape(2, 50) + 1
    timestamps = pd.date_range("2026-01-01", periods=50, freq="5min")
    origins = np.asarray([INPUT_STEPS, INPUT_STEPS + 1])
    dcrnn, staeformer, targets = _window_arrays(
        values, timestamps, origins, mean=10.0, std=2.0
    )
    assert dcrnn.shape == (2, INPUT_STEPS, 2, 2)
    assert staeformer.shape == (2, INPUT_STEPS, 2, 3)
    assert targets.shape == (2, HORIZON, 2, 1)
    np.testing.assert_array_equal(
        targets[0, :, :, 0], values[:, INPUT_STEPS : INPUT_STEPS + HORIZON].T
    )


def test_merge_origin_metrics_requires_identical_origins() -> None:
    base = [{"origin": 10, "timestamp": "x", "metrics": {"a": {"mae": 1.0}}}]
    addition = [
        {"origin": 10, "timestamp": "x", "metrics": {"b": {"mae": 2.0}}}
    ]
    merged = merge_origin_metrics(base, [addition])
    assert set(merged[0]["metrics"]) == {"a", "b"}


def test_existing_timesfm_artifact_uses_panel_timestamp_schema() -> None:
    panel, _, _ = build_panel()
    rows = load_timesfm_origin_metrics(panel)
    assert rows[0]["timestamp"] == panel.timestamps[rows[0]["origin"]]
