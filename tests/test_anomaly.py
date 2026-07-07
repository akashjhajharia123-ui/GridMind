import numpy as np
import pandas as pd
import pytest

from src.anomaly import (
    calculate_residuals,
    fit_residual_threshold,
    detect_anomalies,
)


def test_calculate_residuals_uses_actual_minus_prediction():
    actual = pd.Series([100.0, 120.0, 80.0])
    predicted = pd.Series([90.0, 125.0, 70.0])

    residuals = calculate_residuals(
        actual,
        predicted,
    )

    expected = np.array([
        10.0,
        -5.0,
        10.0,
    ])

    np.testing.assert_allclose(
        residuals,
        expected,
    )


def test_calculate_residuals_rejects_length_mismatch():
    actual = [100.0, 120.0]
    predicted = [95.0]

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        calculate_residuals(
            actual,
            predicted,
        )


def test_fit_residual_threshold_uses_absolute_quantile():
    residuals = np.array([
        -1.0,
        2.0,
        -3.0,
        4.0,
    ])

    threshold = fit_residual_threshold(
        residuals,
        quantile=0.75,
    )

    expected = np.quantile(
        np.abs(residuals),
        0.75,
    )

    assert threshold == pytest.approx(
        expected
    )


def test_fit_residual_threshold_rejects_invalid_quantile():
    residuals = [1.0, 2.0, 3.0]

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        fit_residual_threshold(
            residuals,
            quantile=1.5,
        )


def test_detect_anomalies_flags_large_positive_and_negative_errors():
    actual = pd.Series([
        100.0,
        150.0,
        50.0,
        105.0,
    ])

    predicted = pd.Series([
        100.0,
        100.0,
        100.0,
        100.0,
    ])

    result = detect_anomalies(
        actual,
        predicted,
        threshold=40.0,
    )

    expected_flags = [
        False,
        True,
        True,
        False,
    ]

    assert (
        result["is_anomaly"].tolist()
        == expected_flags
    )

    assert result["residual"].tolist() == [
        0.0,
        50.0,
        -50.0,
        5.0,
    ]

    assert result[
        "absolute_residual"
    ].tolist() == [
        0.0,
        50.0,
        50.0,
        5.0,
    ]


def test_detect_anomalies_rejects_non_positive_threshold():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        detect_anomalies(
            [100.0],
            [90.0],
            threshold=0.0,
        )