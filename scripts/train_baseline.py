from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Allow imports from project root when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_energy_data
from src.evaluation import calculate_metrics
from src.forecast_features import create_forecast_features
from src.split import chronological_split


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AEP_hourly.csv"

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


def main():
    print("Loading data...")
    df = load_energy_data(DATA_PATH)

    print("Creating leakage-safe forecast features...")
    df = create_forecast_features(
        df,
        target_column=TARGET_COLUMN,
    )

    # Remove only rows unavailable because of historical lags/rolling windows
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

    # Fair baseline on exactly the same validation rows.
    # lag_168 is same-hour-last-week and was created before splitting.
    baseline_pred = val_df["lag_168"]

    baseline_metrics = calculate_metrics(
        y_val,
        baseline_pred,
    )

    print("\nValidation Results")
    print("-" * 45)

    results = pd.DataFrame({
        "Seasonal Naive": baseline_metrics,
        "Random Forest": ml_metrics,
    }).T

    print(results.round(3))

    baseline_mape = baseline_metrics["MAPE"]
    ml_mape = ml_metrics["MAPE"]

    improvement = (
        (baseline_mape - ml_mape)
        / baseline_mape
        * 100
    )

    print(
        f"\nMAPE improvement vs baseline: "
        f"{improvement:.2f}%"
    )

    # Test set intentionally untouched for now.
    print(
        "\nTest set remains untouched for final evaluation."
    )


if __name__ == "__main__":
    main()