from pathlib import Path
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_energy_data
from src.forecast_features import create_forecast_features
from src.model_io import load_model
from src.split import chronological_split


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AEP_hourly.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"

IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "feature_importance.csv"
)

DIAGNOSTICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "validation_diagnostics.json"
)

WORST_ERRORS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "worst_validation_errors.csv"
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

    train_df, val_df, test_df = chronological_split(df)

    print(
        f"Train: {len(train_df):,} | "
        f"Validation: {len(val_df):,} | "
        f"Test reserved: {len(test_df):,}"
    )

    print("Loading saved model...")
    model = load_model(MODEL_PATH)

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    print("Predicting validation period...")
    predictions = model.predict(X_val)

    diagnostics_df = val_df[
        ["Datetime", TARGET_COLUMN]
    ].copy()

    diagnostics_df["prediction"] = predictions

    diagnostics_df["residual"] = (
        diagnostics_df[TARGET_COLUMN]
        - diagnostics_df["prediction"]
    )

    diagnostics_df["absolute_error"] = (
        diagnostics_df["residual"].abs()
    )

    diagnostics_df["absolute_percentage_error"] = (
        diagnostics_df["absolute_error"]
        / diagnostics_df[TARGET_COLUMN].abs()
        * 100
    )

    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    IMPORTANCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    worst_errors = diagnostics_df.nlargest(
        10,
        "absolute_error",
    )

    worst_errors.to_csv(
        WORST_ERRORS_PATH,
        index=False,
    )

    diagnostics = {
        "validation_rows": int(len(val_df)),
        "mean_residual": float(
            diagnostics_df["residual"].mean()
        ),
        "median_residual": float(
            diagnostics_df["residual"].median()
        ),
        "residual_std": float(
            diagnostics_df["residual"].std()
        ),
        "mean_absolute_error": float(
            diagnostics_df["absolute_error"].mean()
        ),
        "median_absolute_error": float(
            diagnostics_df["absolute_error"].median()
        ),
        "p95_absolute_error": float(
            diagnostics_df["absolute_error"].quantile(0.95)
        ),
        "max_absolute_error": float(
            diagnostics_df["absolute_error"].max()
        ),
        "mean_absolute_percentage_error": float(
            diagnostics_df[
                "absolute_percentage_error"
            ].mean()
        ),
        "test_set_evaluated": False,
    }

    save_json(
        diagnostics,
        DIAGNOSTICS_PATH,
    )

    print("\nTop Feature Importances")
    print("-" * 45)
    print(
        feature_importance.head(10).to_string(
            index=False
        )
    )

    print("\nValidation Residual Diagnostics")
    print("-" * 45)

    for key, value in diagnostics.items():
        print(f"{key}: {value}")

    print("\nWorst 10 Validation Errors")
    print("-" * 45)

    print(
        worst_errors[
            [
                "Datetime",
                TARGET_COLUMN,
                "prediction",
                "absolute_error",
            ]
        ].to_string(index=False)
    )

    print("\nArtifacts saved:")
    print(IMPORTANCE_PATH)
    print(DIAGNOSTICS_PATH)
    print(WORST_ERRORS_PATH)

    print(
        "\nFinal test set remains untouched."
    )


if __name__ == "__main__":
    main()