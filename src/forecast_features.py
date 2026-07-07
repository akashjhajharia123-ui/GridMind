import pandas as pd


LAG_HOURS = [24, 48, 168]


def create_forecast_features(
    df: pd.DataFrame,
    target_column: str = "AEP_MW",
) -> pd.DataFrame:
    """
    Create leakage-safe forecasting features using only past load values.
    """

    result = df.copy()

    required_columns = ["Datetime", target_column]
    missing_columns = [
        col for col in required_columns
        if col not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result["Datetime"] = pd.to_datetime(result["Datetime"])

    # Critical: lags depend on chronological row order
    result = result.sort_values("Datetime").reset_index(drop=True)

    # Future-known calendar features
    result["hour"] = result["Datetime"].dt.hour
    result["day_of_week"] = result["Datetime"].dt.dayofweek
    result["month"] = result["Datetime"].dt.month
    result["day_of_year"] = result["Datetime"].dt.dayofyear
    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)

    # Past-only load features
    for lag in LAG_HOURS:
        result[f"lag_{lag}"] = result[target_column].shift(lag)

    # Past-only rolling statistics.
    # shift(24) is intentional for day-ahead forecasting:
    # every row uses information available at least 24 hours earlier.
    past_load = result[target_column].shift(24)

    result["rolling_mean_24"] = (
        past_load
        .rolling(window=24)
        .mean()
    )

    result["rolling_std_24"] = (
        past_load
        .rolling(window=24)
        .std()
    )

    result["rolling_mean_168"] = (
        past_load
        .rolling(window=168)
        .mean()
    )

    return result