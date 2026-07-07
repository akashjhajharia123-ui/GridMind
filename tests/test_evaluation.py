import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    calculate_metrics,
    seasonal_naive_forecast,
)


def test_calculate_metrics_perfect_predictions():
    y_true = [100.0, 200.0, 300.0]
    y_pred = [100.0, 200.0, 300.0]

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["RMSE"] == pytest.approx(0.0)
    assert metrics["MAPE"] == pytest.approx(0.0)


def test_calculate_metrics_known_values():
    y_true = [100.0, 200.0]
    y_pred = [110.0, 180.0]

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["MAE"] == pytest.approx(15.0)
    assert metrics["RMSE"] == pytest.approx(
        np.sqrt(250.0)
    )
    assert metrics["MAPE"] == pytest.approx(10.0)


def test_calculate_metrics_ignores_zero_targets_for_mape():
    y_true = [0.0, 100.0]
    y_pred = [50.0, 110.0]

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["MAPE"] == pytest.approx(10.0)


def test_calculate_metrics_all_zero_targets_returns_nan_mape():
    y_true = [0.0, 0.0]
    y_pred = [10.0, 20.0]

    metrics = calculate_metrics(y_true, y_pred)

    assert np.isnan(metrics["MAPE"])


def test_calculate_metrics_shape_mismatch_raises_error():
    with pytest.raises(
        ValueError,
        match="must have the same shape",
    ):
        calculate_metrics(
            [100.0, 200.0],
            [100.0],
        )


def test_seasonal_naive_forecast_uses_past_values():
    df = pd.DataFrame({
        "AEP_MW": np.arange(
            200,
            dtype=float,
        )
    })

    forecast = seasonal_naive_forecast(
        df,
        lag_hours=168,
    )

    assert pd.isna(forecast.iloc[167])
    assert forecast.iloc[168] == pytest.approx(0.0)
    assert forecast.iloc[169] == pytest.approx(1.0)


def test_seasonal_naive_invalid_lag_raises_error():
    df = pd.DataFrame({
        "AEP_MW": [100.0, 200.0]
    })

    with pytest.raises(
        ValueError,
        match="lag_hours must be greater than 0",
    ):
        seasonal_naive_forecast(
            df,
            lag_hours=0,
        )