import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def calculate_metrics(
    y_true,
    y_pred,
) -> dict:
    """
    Calculate regression metrics for energy demand forecasts.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape"
        )

    if y_true.size == 0:
        raise ValueError(
            "y_true and y_pred must not be empty"
        )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    non_zero_mask = y_true != 0

    if non_zero_mask.any():
        mape = np.mean(
            np.abs(
                (
                    y_true[non_zero_mask]
                    - y_pred[non_zero_mask]
                )
                / y_true[non_zero_mask]
            )
        ) * 100
    else:
        mape = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


def seasonal_naive_forecast(
    df,
    target_column: str = "AEP_MW",
    lag_hours: int = 168,
):
    """
    Forecast demand using the value from the same hour
    one week earlier by default.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Missing target column: {target_column}"
        )

    if lag_hours <= 0:
        raise ValueError(
            "lag_hours must be greater than 0"
        )

    return df[target_column].shift(lag_hours)