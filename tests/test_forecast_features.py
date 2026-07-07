import numpy as np
import pandas as pd
import pytest

from src.forecast_features import create_forecast_features


def make_hourly_data(n_hours=250):
    return pd.DataFrame({
        "Datetime": pd.date_range(
            start="2024-01-01",
            periods=n_hours,
            freq="h",
        ),
        "AEP_MW": np.arange(
            1000,
            1000 + n_hours,
            dtype=float,
        ),
    })


def test_create_forecast_features_adds_expected_columns():
    df = make_hourly_data()

    result = create_forecast_features(df)

    expected_columns = [
        "hour",
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend",
        "lag_24",
        "lag_48",
        "lag_168",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_mean_168",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_lag_features_use_past_values():
    df = make_hourly_data()

    result = create_forecast_features(df)

    assert result.loc[24, "lag_24"] == df.loc[0, "AEP_MW"]
    assert result.loc[48, "lag_48"] == df.loc[0, "AEP_MW"]
    assert result.loc[168, "lag_168"] == df.loc[0, "AEP_MW"]


def test_rolling_mean_24_is_day_ahead_safe():
    df = make_hourly_data()

    result = create_forecast_features(df)

    expected_mean = df.loc[0:23, "AEP_MW"].mean()

    assert result.loc[47, "rolling_mean_24"] == pytest.approx(
        expected_mean
    )


def test_future_change_does_not_affect_past_features():
    df = make_hourly_data()

    original = create_forecast_features(df)

    modified_df = df.copy()
    modified_df.loc[200:, "AEP_MW"] = 999999.0

    modified = create_forecast_features(modified_df)

    feature_columns = [
        "lag_24",
        "lag_48",
        "lag_168",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_mean_168",
    ]

    pd.testing.assert_frame_equal(
        original.loc[:199, feature_columns],
        modified.loc[:199, feature_columns],
    )


def test_missing_target_column_raises_error():
    df = pd.DataFrame({
        "Datetime": pd.date_range(
            start="2024-01-01",
            periods=10,
            freq="h",
        )
    })

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        create_forecast_features(df)