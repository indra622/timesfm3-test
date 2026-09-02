import numpy as np
import pandas as pd
import pytest

from metr_graph_followup import (
    dcrnn_adjacency,
    direct_ridge_forecast,
    dual_random_walk_supports,
    leakage_safe_selection_end,
    moving_block_bootstrap_summary,
    random_walk,
)


def test_leakage_safe_selection_end_precedes_earliest_nominal_context() -> None:
    end = leakage_safe_selection_end(34_272, 2016, 12, 40, 288)

    assert end == 17_136


def test_dcrnn_adjacency_is_directed_and_thresholded() -> None:
    distances = pd.DataFrame(
        {
            "from": ["a", "b", "a"],
            "to": ["b", "a", "c"],
            "cost": [1.0, 2.0, 100.0],
        }
    )

    adjacency = dcrnn_adjacency(distances, ["a", "b", "c"], normalized_k=0.1)

    assert adjacency[0, 1] > adjacency[1, 0] > 0
    assert adjacency[0, 2] == 0
    assert adjacency[1, 2] == 0


def test_random_walk_and_dual_supports() -> None:
    adjacency = np.asarray([[0, 2], [1, 0]], dtype=np.float32)

    walk = random_walk(adjacency)
    supports = dual_random_walk_supports(adjacency, max_diffusion_step=2)

    np.testing.assert_allclose(walk.sum(axis=1), [1, 1])
    assert len(supports) == 4
    assert all(support.shape == (2, 2) for support in supports)


def test_direct_ridge_forecast_has_expected_shape() -> None:
    time = np.arange(420, dtype=np.float32)
    history = np.stack([time, np.roll(time, 1)])
    adjacency = np.asarray([[0, 1], [1, 0]], dtype=np.float32)

    forecast = direct_ridge_forecast(
        history,
        target_indices=[0],
        horizon=4,
        lags=(1, 2, 3),
        supports=dual_random_walk_supports(adjacency),
        alpha=1.0,
    )

    assert forecast.shape == (1, 4)
    assert np.isfinite(forecast).all()


def test_moving_block_bootstrap_is_deterministic() -> None:
    deltas = np.asarray([-2.0, -1.0, 0.5, -0.5])

    first = moving_block_bootstrap_summary(deltas, block_size=2, resamples=200)
    second = moving_block_bootstrap_summary(deltas, block_size=2, resamples=200)

    assert first == second
    assert first["mean_delta"] == pytest.approx(-0.75)
    assert first["win_rate"] == pytest.approx(0.75)
