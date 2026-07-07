from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_energy_data
from src.evaluation import calculate_metrics
from src.forecast_features import create_forecast_features
from src.split import chronological_split


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AEP_hourly.csv"

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "backtest_results.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "backtest_summary.json"
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

N_FOLDS = 4
MIN_TRAIN_FRACTION = 0.50


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


def build_expanding_folds(
    development_df,
    n_folds=N_FOLDS,
    min_train_fraction=MIN_TRAIN_FRACTION,
):
    n_rows = len(development_df)

    min_train_size = int(
        n_rows * min_train_fraction
    )

    remaining_rows = (
        n_rows - min_train_size
    )

    fold_size = (
        remaining_rows // n_folds
    )

    if fold_size <= 0:
        raise ValueError(
            "Not enough rows to create backtest folds."
        )

    folds = []

    for fold_number in range(1, n_folds + 1):
        train_end = (
            min_train_size
            + (fold_number - 1) * fold_size
        )

        validation_end = (
            train_end + fold_size
        )

        if fold_number == n_folds:
            validation_end = n_rows

        train_fold = development_df.iloc[
            :train_end
        ].copy()

        validation_fold = development_df.iloc[
            train_end:validation_end
        ].copy()

        folds.append(
            (
                fold_number,
                train_fold,
                validation_fold,
            )
        )

    return folds


def create_random_forest():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def create_hist_gradient_boosting():
    return HistGradientBoostingRegressor(
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


def evaluate_predictions(
    actual,
    predictions,
):
    metrics = calculate_metrics(
        actual,
        predictions,
    )

    return {
        key: float(value)
        for key, value in metrics.items()
    }


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
        f"Final test rows reserved: "
        f"{len(test_df):,}"
    )

    folds = build_expanding_folds(
        development_df
    )

    all_results = []

    for (
        fold_number,
        fold_train,
        fold_validation,
    ) in folds:
        print(
            f"\nFold {fold_number}/{N_FOLDS}"
        )

        print(
            f"Train: {len(fold_train):,} | "
            f"Validation: "
            f"{len(fold_validation):,}"
        )

        X_train = fold_train[
            FEATURE_COLUMNS
        ]

        y_train = fold_train[
            TARGET_COLUMN
        ]

        X_validation = fold_validation[
            FEATURE_COLUMNS
        ]

        y_validation = fold_validation[
            TARGET_COLUMN
        ]

        baseline_predictions = (
            fold_validation["lag_168"]
            .to_numpy()
        )

        baseline_metrics = (
            evaluate_predictions(
                y_validation,
                baseline_predictions,
            )
        )

        all_results.append({
            "fold": fold_number,
            "model": "Seasonal Naive",
            **baseline_metrics,
            "train_seconds": 0.0,
            "predict_seconds": 0.0,
        })

        models = {
            "Random Forest":
                create_random_forest(),
            "HistGradientBoosting":
                create_hist_gradient_boosting(),
        }

        for model_name, model in models.items():
            print(
                f"Training {model_name}..."
            )

            train_start = (
                time.perf_counter()
            )

            model.fit(
                X_train,
                y_train,
            )

            train_seconds = (
                time.perf_counter()
                - train_start
            )

            predict_start = (
                time.perf_counter()
            )

            predictions = model.predict(
                X_validation
            )

            predict_seconds = (
                time.perf_counter()
                - predict_start
            )

            metrics = evaluate_predictions(
                y_validation,
                predictions,
            )

            all_results.append({
                "fold": fold_number,
                "model": model_name,
                **metrics,
                "train_seconds":
                    float(train_seconds),
                "predict_seconds":
                    float(predict_seconds),
            })

            print(
                f"{model_name} MAPE: "
                f"{metrics['MAPE']:.4f}%"
            )

    results_df = pd.DataFrame(
        all_results
    )

    summary_df = (
        results_df
        .groupby("model")
        .agg({
            "MAE": ["mean", "std"],
            "RMSE": ["mean", "std"],
            "MAPE": ["mean", "std"],
            "train_seconds": ["mean"],
            "predict_seconds": ["mean"],
        })
    )

    mean_mape = (
        results_df
        .groupby("model")["MAPE"]
        .mean()
        .sort_values()
    )

    winner = mean_mape.index[0]

    print("\nFold-wise Backtest Results")
    print("-" * 90)

    print(
        results_df[
            [
                "fold",
                "model",
                "MAE",
                "RMSE",
                "MAPE",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nAggregate Backtest Summary")
    print("-" * 90)

    print(
        summary_df
        .round(4)
        .to_string()
    )

    print(
        f"\nBacktest winner by mean MAPE: "
        f"{winner}"
    )

    summary_artifact = {
        "n_folds": N_FOLDS,
        "min_train_fraction":
            MIN_TRAIN_FRACTION,
        "development_rows":
            int(len(development_df)),
        "reserved_test_rows":
            int(len(test_df)),
        "winner_by_mean_mape":
            winner,
        "mean_mape_by_model": {
            model: float(value)
            for model, value
            in mean_mape.items()
        },
        "mape_std_by_model": {
            model: float(value)
            for model, value
            in results_df
            .groupby("model")["MAPE"]
            .std()
            .items()
        },
        "test_set_evaluated": False,
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    save_json(
        summary_artifact,
        SUMMARY_PATH,
    )

    print("\nArtifacts saved:")
    print(RESULTS_PATH)
    print(SUMMARY_PATH)

    print(
        "\nFinal test set remains untouched."
    )


if __name__ == "__main__":
    main()