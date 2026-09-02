import numpy as np
import pytest

from real_data_benchmarks import (
  BenchmarkDataset,
  interval_metrics,
  point_metrics,
  seasonal_naive,
)


def test_generic_seasonal_naive_repeats_requested_period():
  history = np.arange(24, dtype=np.float32).reshape(2, 12)
  prediction = seasonal_naive(history, horizon=6, season_length=4)

  np.testing.assert_array_equal(prediction[:, :4], history[:, -4:])
  np.testing.assert_array_equal(prediction[:, 4:], history[:, -4:-2])


def test_point_metrics_exact_forecast_is_zero():
  history = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
  actual = np.array([[4.0, 5.0], [8.0, 10.0]])
  metrics = point_metrics(actual, actual.copy(), history)

  assert metrics["mae"] == 0
  assert metrics["rmse"] == 0
  assert metrics["wape"] == 0
  assert metrics["mean_rmsse"] == 0


def test_interval_metrics_uses_outer_quantiles():
  actual = np.array([[2.0, 5.0]])
  quantiles = np.zeros((1, 2, 9))
  quantiles[:, :, 0] = [[1.0, 6.0]]
  quantiles[:, :, -1] = [[3.0, 8.0]]

  metrics = interval_metrics(actual, quantiles)

  assert metrics["p10_p90_coverage"] == 0.5
  assert metrics["p10_p90_mean_width"] == 2.0


def test_dataset_validation_rejects_wrong_covariate_length():
  dataset = BenchmarkDataset(
    name="bad",
    values=np.ones((2, 8), dtype=np.float32),
    target_names=["a", "b"],
    context_length=6,
    horizon=2,
    season_length=2,
    past_only_covariates=np.ones((1, 5), dtype=np.float32),
    past_future_covariates=None,
    metadata={},
  )

  with pytest.raises(ValueError, match="past-only"):
    dataset.validate()

