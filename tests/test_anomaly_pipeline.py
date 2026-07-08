import pandas as pd
import pytest

from scripts.detect_anomalies import (
    classify_direction,
    classify_severity,
    create_calibration_model,
    validate_prediction_artifact,
)


def test_calibration_model_uses_frozen_configuration():
    model = create_calibration_model()

    params = model.get_params()

    assert params["n_estimators"] == 100
    assert params["max_depth"] == 18
    assert params["min_samples_leaf"] == 2
    assert params["random_state"] == 42
    assert params["n_jobs"] == -1


@pytest.mark.parametrize(
    ("residual", "expected"),
    [
        (100.0, "spike"),
        (-100.0, "drop"),
        (0.0, "neutral"),
    ],
)
def test_classify_direction(
    residual,
    expected,
):
    assert classify_direction(
        residual
    ) == expected


@pytest.mark.parametrize(
    ("score", "is_anomaly", "expected"),
    [
        (0.5, False, "normal"),
        (1.1, True, "moderate"),
        (1.5, True, "high"),
        (1.99, True, "high"),
        (2.0, True, "critical"),
        (3.0, True, "critical"),
    ],
)
def test_classify_severity(
    score,
    is_anomaly,
    expected,
):
    assert classify_severity(
        score,
        is_anomaly,
    ) == expected


def test_prediction_artifact_validation_accepts_match():
    test_df = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2020-01-01 00:00:00",
            "2020-01-01 01:00:00",
        ]),
    })

    predictions_df = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2020-01-01 00:00:00",
            "2020-01-01 01:00:00",
        ]),
        "AEP_MW": [100.0, 110.0],
        "prediction": [101.0, 109.0],
    })

    validate_prediction_artifact(
        predictions_df,
        test_df,
    )


def test_prediction_artifact_validation_rejects_missing_column():
    test_df = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2020-01-01 00:00:00",
        ]),
    })

    predictions_df = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2020-01-01 00:00:00",
        ]),
        "AEP_MW": [100.0],
    })

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_prediction_artifact(
            predictions_df,
            test_df,
        )


def test_prediction_artifact_validation_rejects_timestamp_mismatch():
    test_df = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2020-01-01 00:00:00",
        ]),
    })

    predictions_df = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2020-01-02 00:00:00",
        ]),
        "AEP_MW": [100.0],
        "prediction": [101.0],
    })

    with pytest.raises(
        ValueError,
        match="timestamps do not match",
    ):
        validate_prediction_artifact(
            predictions_df,
            test_df,
        )