from pathlib import Path
import json
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly import (
    calculate_residuals,
    detect_anomalies,
    fit_residual_threshold,
)
from src.data_loader import load_energy_data
from src.forecast_features import create_forecast_features
from src.split import chronological_split


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "AEP_hourly.csv"
)

FINAL_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "final_test_predictions.csv"
)

ANOMALY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "anomalies"
    / "final_test_anomalies.csv"
)

ANOMALY_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "anomaly_summary.json"
)

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "AEP_MW"
THRESHOLD_QUANTILE = 0.99


def save_json(data, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            sort_keys=True,
        )


def create_calibration_model():
    """
    Same frozen Random Forest configuration used
    during forecasting model selection.

    This model is trained on train rows only.
    Validation remains unseen for threshold fitting.
    """
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def classify_direction(residual):
    if residual > 0:
        return "spike"

    if residual < 0:
        return "drop"

    return "neutral"


def classify_severity(
    severity_score,
    is_anomaly,
):
    if not is_anomaly:
        return "normal"

    if severity_score < 1.5:
        return "moderate"

    if severity_score < 2.0:
        return "high"

    return "critical"
def validate_prediction_artifact(
    predictions_df,
    test_df,
):
    required_columns = {
        "Datetime",
        TARGET_COLUMN,
        "prediction",
    }

    missing_columns = (
        required_columns
        - set(predictions_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Final predictions artifact is missing "
            "required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(predictions_df) != len(test_df):
        raise ValueError(
            "Final predictions row count does not "
            "match reserved test split."
        )

    expected_datetimes = (
        test_df["Datetime"]
        .reset_index(drop=True)
    )

    artifact_datetimes = (
        predictions_df["Datetime"]
        .reset_index(drop=True)
    )

    if not expected_datetimes.equals(
        artifact_datetimes
    ):
        raise ValueError(
            "Final predictions timestamps do not "
            "match reserved test split."
        )

def main():
    print("Loading data...")

    df = load_energy_data(DATA_PATH)

    print(
        "Creating leakage-safe forecast features..."
    )

    df = create_forecast_features(
        df,
        target_column=TARGET_COLUMN,
    )

    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).reset_index(drop=True)

    train_df, validation_df, test_df = (
        chronological_split(df)
    )

    print(
        f"Train rows for calibration model: "
        f"{len(train_df):,}"
    )

    print(
        f"Unseen validation rows for "
        f"threshold fitting: "
        f"{len(validation_df):,}"
    )

    print(
        f"Final test rows for scoring only: "
        f"{len(test_df):,}"
    )

    print(
        "\nTraining calibration model "
        "on train rows only..."
    )

    calibration_model = (
        create_calibration_model()
    )

    calibration_model.fit(
        train_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
    )

    print(
        "Predicting unseen validation period..."
    )

    validation_predictions = (
        calibration_model.predict(
            validation_df[FEATURE_COLUMNS]
        )
    )

    validation_residuals = (
        calculate_residuals(
            validation_df[TARGET_COLUMN],
            validation_predictions,
        )
    )

    threshold = fit_residual_threshold(
        validation_residuals,
        quantile=THRESHOLD_QUANTILE,
    )

    print(
        f"Frozen residual threshold "
        f"(q={THRESHOLD_QUANTILE:.2f}): "
        f"{threshold:.4f} MW"
    )

    if not FINAL_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Final predictions artifact not found: "
            f"{FINAL_PREDICTIONS_PATH}"
        )

    print(
        "\nLoading frozen final-test predictions..."
    )

    predictions_df = pd.read_csv(
        FINAL_PREDICTIONS_PATH,
        parse_dates=["Datetime"],
    )

    validate_prediction_artifact(
    predictions_df,
    test_df,
)

    print(
        "Scoring final-test residuals with "
        "frozen validation threshold..."
    )

    anomaly_result = detect_anomalies(
        predictions_df[TARGET_COLUMN],
        predictions_df["prediction"],
        threshold=threshold,
    )

    output_df = predictions_df[
        [
            "Datetime",
            TARGET_COLUMN,
            "prediction",
        ]
    ].copy()

    output_df["residual"] = (
        anomaly_result["residual"]
        .to_numpy()
    )

    output_df["absolute_residual"] = (
        anomaly_result[
            "absolute_residual"
        ].to_numpy()
    )

    output_df["is_anomaly"] = (
        anomaly_result["is_anomaly"]
        .to_numpy()
    )

    output_df["severity_score"] = (
        output_df["absolute_residual"]
        / threshold
    )

    output_df["direction"] = (
        output_df["residual"]
        .map(classify_direction)
    )

    output_df["severity"] = [
        classify_severity(
            severity_score,
            is_anomaly,
        )
        for severity_score, is_anomaly
        in zip(
            output_df["severity_score"],
            output_df["is_anomaly"],
        )
    ]

    anomaly_count = int(
        output_df["is_anomaly"].sum()
    )

    total_rows = int(
        len(output_df)
    )

    anomaly_rate_percent = (
        anomaly_count
        / total_rows
        * 100
    )

    anomaly_only_df = (
        output_df[
            output_df["is_anomaly"]
        ]
        .sort_values(
            "severity_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    spike_count = int(
        (
            anomaly_only_df["direction"]
            == "spike"
        ).sum()
    )

    drop_count = int(
        (
            anomaly_only_df["direction"]
            == "drop"
        ).sum()
    )

    severity_counts = {
        str(key): int(value)
        for key, value
        in anomaly_only_df[
            "severity"
        ]
        .value_counts()
        .to_dict()
        .items()
    }

    top_anomalies = []

    for row in (
        anomaly_only_df
        .head(10)
        .itertuples(index=False)
    ):
        top_anomalies.append({
            "datetime":
                row.Datetime.isoformat(),
            "actual_mw":
                float(row.AEP_MW),
            "prediction_mw":
                float(row.prediction),
            "residual_mw":
                float(row.residual),
            "absolute_residual_mw":
                float(row.absolute_residual),
            "severity_score":
                float(row.severity_score),
            "direction":
                str(row.direction),
            "severity":
                str(row.severity),
        })

    summary = {
        "method":
            "absolute_forecast_residual_quantile",
        "calibration_model":
            "RandomForestRegressor",
        "calibration_model_training_scope":
            "train_only",
        "threshold_source":
            "unseen_validation_residuals",
        "threshold_quantile":
            THRESHOLD_QUANTILE,
        "threshold_mw":
            float(threshold),
        "final_test_usage":
            "scoring_only",
        "total_scored_rows":
            total_rows,
        "anomaly_count":
            anomaly_count,
        "anomaly_rate_percent":
            float(anomaly_rate_percent),
        "spike_count":
            spike_count,
        "drop_count":
            drop_count,
        "severity_counts":
            severity_counts,
        "top_anomalies":
            top_anomalies,
        "threshold_fit_on_final_test":
            False,
        "threshold_fit_on_training_rows":
            False,
        "validation_seen_by_calibration_model":
            False,
    }

    ANOMALY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    anomaly_only_df.to_csv(
        ANOMALY_OUTPUT_PATH,
        index=False,
    )

    save_json(
        summary,
        ANOMALY_SUMMARY_PATH,
    )

    print("\nAnomaly Detection Summary")
    print("-" * 60)

    print(
        f"Threshold: {threshold:.4f} MW"
    )

    print(
        f"Anomalies: "
        f"{anomaly_count:,} / "
        f"{total_rows:,}"
    )

    print(
        f"Anomaly rate: "
        f"{anomaly_rate_percent:.3f}%"
    )

    print(
        f"Spikes: {spike_count:,}"
    )

    print(
        f"Drops: {drop_count:,}"
    )

    print(
        f"Severity counts: "
        f"{severity_counts}"
    )

    if not anomaly_only_df.empty:
        print("\nTop 10 anomalies")
        print("-" * 90)

        print(
            anomaly_only_df[
                [
                    "Datetime",
                    TARGET_COLUMN,
                    "prediction",
                    "residual",
                    "severity_score",
                    "direction",
                    "severity",
                ]
            ]
            .head(10)
            .round(4)
            .to_string(index=False)
        )

    print("\nArtifacts saved:")
    print(ANOMALY_OUTPUT_PATH)
    print(ANOMALY_SUMMARY_PATH)

    print(
        "\nCalibration model was trained "
        "on train rows only."
    )

    print(
        "Threshold was fit on unseen "
        "validation residuals only."
    )

    print(
        "Final test predictions were "
        "used for scoring only."
    )


if __name__ == "__main__":
    main()