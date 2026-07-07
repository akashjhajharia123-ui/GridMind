from pathlib import Path
import json
import sys
import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_energy_data
from src.evaluation import calculate_metrics
from src.forecast_features import create_forecast_features
from src.model_io import save_model
from src.split import chronological_split


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AEP_hourly.csv"

SELECTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "model_selection.json"
)

FINAL_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_final.joblib"
)

FINAL_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "final_test_metrics.json"
)

FINAL_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "final_test_predictions.csv"
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
EXPECTED_SELECTED_MODEL = "Random Forest"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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


def create_final_model():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def main():
    print("Loading frozen model-selection decision...")

    selection = load_json(SELECTION_PATH)

    selected_model = selection.get(
        "selected_model"
    )

    if selected_model != EXPECTED_SELECTED_MODEL:
        raise ValueError(
            "Final evaluation expected "
            f"{EXPECTED_SELECTED_MODEL}, "
            f"but selection artifact contains "
            f"{selected_model}."
        )

    if selection.get(
        "test_set_evaluated"
    ) is not False:
        raise ValueError(
            "Selection artifact does not confirm "
            "an untouched test set."
        )

    print(
        f"Frozen selected model: {selected_model}"
    )

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

    train_df, val_df, test_df = (
        chronological_split(df)
    )

    development_df = pd.concat(
        [train_df, val_df],
        ignore_index=True,
    )

    print(
        f"Development rows: "
        f"{len(development_df):,}"
    )

    print(
        f"Reserved test rows: "
        f"{len(test_df):,}"
    )

    X_development = development_df[
        FEATURE_COLUMNS
    ]

    y_development = development_df[
        TARGET_COLUMN
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    print(
        "\nTraining frozen Random Forest "
        "on full development data..."
    )

    model = create_final_model()

    train_start = time.perf_counter()

    model.fit(
        X_development,
        y_development,
    )

    train_seconds = (
        time.perf_counter() - train_start
    )

    print(
        "Evaluating reserved test set..."
    )

    predict_start = time.perf_counter()

    test_predictions = model.predict(
        X_test
    )

    predict_seconds = (
        time.perf_counter() - predict_start
    )

    model_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    baseline_predictions = (
        test_df["lag_168"].to_numpy()
    )

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions,
    )

    model_mape = float(
        model_metrics["MAPE"]
    )

    baseline_mape = float(
        baseline_metrics["MAPE"]
    )

    relative_improvement = (
        (baseline_mape - model_mape)
        / baseline_mape
        * 100
    )

    prediction_df = test_df[
        ["Datetime", TARGET_COLUMN]
    ].copy()

    prediction_df["prediction"] = (
        test_predictions
    )

    prediction_df["residual"] = (
        prediction_df[TARGET_COLUMN]
        - prediction_df["prediction"]
    )

    prediction_df["absolute_error"] = (
        prediction_df["residual"].abs()
    )

    FINAL_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df.to_csv(
        FINAL_PREDICTIONS_PATH,
        index=False,
    )

    saved_model_path = save_model(
        model,
        FINAL_MODEL_PATH,
    )

    final_artifact = {
        "selected_model": selected_model,
        "training_scope":
            "train_plus_validation",
        "development_rows":
            int(len(development_df)),
        "test_rows":
            int(len(test_df)),
        "random_forest": {
            key: float(value)
            for key, value
            in model_metrics.items()
        },
        "seasonal_naive": {
            key: float(value)
            for key, value
            in baseline_metrics.items()
        },
        "relative_mape_improvement_percent":
            float(relative_improvement),
        "train_seconds":
            float(train_seconds),
        "predict_seconds":
            float(predict_seconds),
        "test_set_evaluated": True,
    }

    save_json(
        final_artifact,
        FINAL_METRICS_PATH,
    )

    results_df = pd.DataFrame({
        "Seasonal Naive": baseline_metrics,
        "Random Forest": model_metrics,
    }).T

    print("\nFinal Test Results")
    print("-" * 60)

    print(
        results_df
        .round(4)
        .to_string()
    )

    print(
        f"\nFinal MAPE improvement vs baseline: "
        f"{relative_improvement:.2f}%"
    )

    print(
        f"\nTraining time: "
        f"{train_seconds:.2f} seconds"
    )

    print(
        f"Prediction time: "
        f"{predict_seconds:.4f} seconds"
    )

    print("\nFinal artifacts saved:")
    print(saved_model_path)
    print(FINAL_METRICS_PATH)
    print(FINAL_PREDICTIONS_PATH)

    print(
        "\nReserved test set has now been "
        "evaluated and is considered consumed."
    )


if __name__ == "__main__":
    main()