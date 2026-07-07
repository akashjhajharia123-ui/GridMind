from pathlib import Path
import json
import sys

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
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"
METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "validation_metrics.json"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "feature_metadata.json"
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


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            sort_keys=True,
        )


def main():
    print("Loading data...")
    df = load_energy_data(DATA_PATH)

    print("Creating leakage-safe forecast features...")
    df = create_forecast_features(
        df,
        target_column=TARGET_COLUMN,
    )

    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).reset_index(drop=True)

    print(f"Usable rows: {len(df):,}")

    train_df, val_df, test_df = chronological_split(df)

    print(
        f"Train: {len(train_df):,} | "
        f"Validation: {len(val_df):,} | "
        f"Test: {len(test_df):,}"
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    print("Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Predicting validation period...")
    val_pred = model.predict(X_val)

    ml_metrics = calculate_metrics(
        y_val,
        val_pred,
    )

    baseline_pred = val_df["lag_168"]

    baseline_metrics = calculate_metrics(
        y_val,
        baseline_pred,
    )

    baseline_mape = baseline_metrics["MAPE"]
    ml_mape = ml_metrics["MAPE"]

    improvement = (
        (baseline_mape - ml_mape)
        / baseline_mape
        * 100
    )

    print("\nValidation Results")
    print("-" * 45)

    results = pd.DataFrame({
        "Seasonal Naive": baseline_metrics,
        "Random Forest": ml_metrics,
    }).T

    print(results.round(3))

    print(
        f"\nMAPE improvement vs baseline: "
        f"{improvement:.2f}%"
    )

    print("\nSaving model and experiment artifacts...")

    saved_model_path = save_model(
        model,
        MODEL_PATH,
    )

    metrics_artifact = {
        "seasonal_naive": {
            key: float(value)
            for key, value in baseline_metrics.items()
        },
        "random_forest": {
            key: float(value)
            for key, value in ml_metrics.items()
        },
        "mape_improvement_percent": float(improvement),
    }

    metadata_artifact = {
        "model_type": "RandomForestRegressor",
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "test_set_evaluated": False,
        "random_state": 42,
    }

    save_json(
        metrics_artifact,
        METRICS_PATH,
    )

    save_json(
        metadata_artifact,
        METADATA_PATH,
    )

    print(f"Model saved to: {saved_model_path}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")

    print(
        "\nTest set remains untouched for final evaluation."
    )


if __name__ == "__main__":
    main()