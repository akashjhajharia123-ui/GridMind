from pathlib import Path
import json
import sys
import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_energy_data
from src.evaluation import calculate_metrics
from src.forecast_features import create_forecast_features
from src.model_io import load_model, save_model
from src.split import chronological_split


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AEP_hourly.csv"

RF_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest.joblib"
)

CHALLENGER_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hist_gradient_boosting.joblib"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "model_comparison.json"
)

TABLE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "model_comparison.csv"
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


def timed_predict(model, X):
    start = time.perf_counter()
    predictions = model.predict(X)
    elapsed = time.perf_counter() - start

    return predictions, elapsed


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

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    results = {}

    print("\nEvaluating Seasonal Naive baseline...")

    baseline_start = time.perf_counter()
    baseline_pred = val_df["lag_168"].to_numpy()
    baseline_predict_time = (
        time.perf_counter() - baseline_start
    )

    baseline_metrics = calculate_metrics(
        y_val,
        baseline_pred,
    )

    results["Seasonal Naive"] = {
        **{
            key: float(value)
            for key, value in baseline_metrics.items()
        },
        "train_seconds": 0.0,
        "predict_seconds": float(
            baseline_predict_time
        ),
    }

    print("Loading saved Random Forest...")

    random_forest = load_model(
        RF_MODEL_PATH
    )

    rf_pred, rf_predict_time = timed_predict(
        random_forest,
        X_val,
    )

    rf_metrics = calculate_metrics(
        y_val,
        rf_pred,
    )

    results["Random Forest"] = {
        **{
            key: float(value)
            for key, value in rf_metrics.items()
        },
        "train_seconds": None,
        "predict_seconds": float(
            rf_predict_time
        ),
    }

    print("Training HistGradientBoosting challenger...")

    challenger = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )

    train_start = time.perf_counter()

    challenger.fit(
        X_train,
        y_train,
    )

    challenger_train_time = (
        time.perf_counter() - train_start
    )

    challenger_pred, challenger_predict_time = (
        timed_predict(
            challenger,
            X_val,
        )
    )

    challenger_metrics = calculate_metrics(
        y_val,
        challenger_pred,
    )

    results["HistGradientBoosting"] = {
        **{
            key: float(value)
            for key, value in challenger_metrics.items()
        },
        "train_seconds": float(
            challenger_train_time
        ),
        "predict_seconds": float(
            challenger_predict_time
        ),
    }

    comparison_df = pd.DataFrame(
        results
    ).T

    comparison_df = comparison_df.sort_values(
        "MAPE",
        ascending=True,
    )

    winner = comparison_df.index[0]

    print("\nModel Comparison")
    print("-" * 75)

    print(
        comparison_df.round(4).to_string()
    )

    print(f"\nValidation winner: {winner}")

    if winner == "HistGradientBoosting":
        saved_path = save_model(
            challenger,
            CHALLENGER_MODEL_PATH,
        )

        print(
            f"Winning challenger saved to: "
            f"{saved_path}"
        )
    else:
        print(
            "Challenger did not beat the current "
            "validation winner."
        )

    best_mape = float(
        comparison_df.iloc[0]["MAPE"]
    )

    baseline_mape = float(
        results["Seasonal Naive"]["MAPE"]
    )

    relative_improvement = (
        (baseline_mape - best_mape)
        / baseline_mape
        * 100
    )

    artifact = {
        "models": results,
        "validation_winner": winner,
        "winner_mape": best_mape,
        "winner_relative_mape_improvement_percent": float(
            relative_improvement
        ),
        "test_set_evaluated": False,
    }

    save_json(
        artifact,
        RESULTS_PATH,
    )

    TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        TABLE_PATH,
        index=True,
        index_label="model",
    )

    print(
        f"\nWinner improvement vs baseline: "
        f"{relative_improvement:.2f}%"
    )

    print("\nArtifacts saved:")
    print(RESULTS_PATH)
    print(TABLE_PATH)

    print(
        "\nFinal test set remains untouched."
    )


if __name__ == "__main__":
    main()