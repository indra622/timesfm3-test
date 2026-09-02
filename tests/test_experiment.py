import numpy as np
import pytest

from run_experiment import (
  forecast_metrics,
  make_synthetic_dataset,
  seasonal_naive,
)


def test_synthetic_dataset_shapes_and_known_future_covariate():
  dataset = make_synthetic_dataset(context_length=70, horizon=14, seed=1)

  assert dataset.history.shape == (2, 70)
  assert dataset.future.shape == (2, 14)
  assert dataset.promotion.shape == (84,)
  assert dataset.foot_traffic.shape == (84,)
  assert dataset.future_promotion.sum() > 0


def test_seasonal_naive_repeats_last_week():
  history = np.arange(28, dtype=np.float32).reshape(2, 14)
  forecast = seasonal_naive(history, horizon=10)

  np.testing.assert_array_equal(forecast[:, :7], history[:, -7:])
  np.testing.assert_array_equal(forecast[:, 7:], history[:, -7:-4])


def test_metrics_are_zero_for_exact_forecast():
  actual = np.array([[1.0, 2.0], [3.0, 4.0]])
  metrics = forecast_metrics(actual, actual.copy())

  assert metrics["mae"] == 0
  assert metrics["rmse"] == 0
  assert metrics["wape"] == 0


def test_metrics_reject_shape_mismatch():
  with pytest.raises(ValueError, match="shape mismatch"):
    forecast_metrics(np.zeros((2, 4)), np.zeros((1, 4)))

