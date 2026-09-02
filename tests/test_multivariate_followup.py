import numpy as np
import pytest

from multivariate_followup import (
    RollingPanel,
    aggregate_origins,
    interpolate_zero_markers,
    lowest_missing_indices,
    rolling_origins,
)


def test_rolling_origins_compute_required_length():
    origins, total_length = rolling_origins(10, 3, 4, 2)

    assert origins == [10, 12, 14, 16]
    assert total_length == 19


def test_interpolate_zero_markers_uses_context_values():
    values = np.array([[1.0, 0.0, 3.0, 0.0]], dtype=np.float32)

    interpolated = interpolate_zero_markers(values)

    np.testing.assert_allclose(interpolated, [[1.0, 2.0, 3.0, 3.0]])


def test_lowest_missing_indices_uses_prefix_and_fixed_fraction():
    values = np.array(
        [
            [1, 1, 1, 0],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.float32,
    )

    eligible, zero_rate = lowest_missing_indices(
        values, selection_end=3, keep_fraction=0.5
    )

    np.testing.assert_array_equal(eligible, [0, 1])
    np.testing.assert_allclose(zero_rate, [0, 1 / 3, 2 / 3, 1 / 3])


def test_aggregate_origins_reports_paired_win_rate():
    origin_results = [
        {
            "metrics": {
                "timesfm3_univariate": {"mae": 2.0},
                "related": {"mae": 1.0},
            }
        },
        {
            "metrics": {
                "timesfm3_univariate": {"mae": 2.0},
                "related": {"mae": 3.0},
            }
        },
    ]

    aggregate = aggregate_origins(origin_results)

    assert aggregate["related"]["mae"]["mean"] == 2.0
    assert aggregate["related"]["paired_mae_vs_univariate"]["mean_delta"] == 0.0
    assert aggregate["related"]["paired_mae_vs_univariate"]["win_rate"] == 0.5


def test_panel_validation_rejects_mismatched_control_shape():
    panel = RollingPanel(
        name="bad",
        target_values=np.ones((2, 10), dtype=np.float32),
        target_names=["a", "b"],
        related_covariates=np.ones((2, 1, 10), dtype=np.float32),
        control_covariates=np.ones((2, 2, 10), dtype=np.float32),
        past_future_covariates=None,
        timestamps=[str(index) for index in range(10)],
        origins=[8],
        context_length=8,
        horizon=2,
        season_length=2,
        metadata={},
    )

    with pytest.raises(ValueError, match="same shape"):
        panel.validate()
