import numpy as np
import pandas as pd


def _to_1d_float_array(values, name):
    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return array


def calculate_residuals(
    actual,
    predicted,
):
    actual_array = _to_1d_float_array(
        actual,
        "actual",
    )

    predicted_array = _to_1d_float_array(
        predicted,
        "predicted",
    )

    if len(actual_array) != len(predicted_array):
        raise ValueError(
            "actual and predicted must have "
            "the same length."
        )

    return actual_array - predicted_array


def fit_residual_threshold(
    residuals,
    quantile=0.99,
):
    if not 0 < quantile < 1:
        raise ValueError(
            "quantile must be between 0 and 1."
        )

    residual_array = _to_1d_float_array(
        residuals,
        "residuals",
    )

    threshold = np.quantile(
        np.abs(residual_array),
        quantile,
    )

    if threshold <= 0:
        raise ValueError(
            "fitted threshold must be positive."
        )

    return float(threshold)


def detect_anomalies(
    actual,
    predicted,
    threshold,
):
    if not np.isfinite(threshold):
        raise ValueError(
            "threshold must be finite and positive."
        )

    if threshold <= 0:
        raise ValueError(
            "threshold must be positive."
        )

    residuals = calculate_residuals(
        actual,
        predicted,
    )

    absolute_residuals = np.abs(
        residuals
    )

    return pd.DataFrame({
        "residual": residuals,
        "absolute_residual":
            absolute_residuals,
        "is_anomaly":
            absolute_residuals > threshold,
    })