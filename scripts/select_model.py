from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPARISON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "model_comparison.json"
)

BACKTEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "backtest_summary.json"
)

SELECTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "model_selection.json"
)

CANDIDATE_MODELS = [
    "Random Forest",
    "HistGradientBoosting",
]

BACKTEST_MAPE_TOLERANCE = 0.05


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


def main():
    print("Loading experiment artifacts...")

    comparison = load_json(
        COMPARISON_PATH
    )

    backtest = load_json(
        BACKTEST_PATH
    )

    if comparison.get(
        "test_set_evaluated"
    ) is not False:
        raise ValueError(
            "Comparison artifact does not confirm "
            "an untouched test set."
        )

    if backtest.get(
        "test_set_evaluated"
    ) is not False:
        raise ValueError(
            "Backtest artifact does not confirm "
            "an untouched test set."
        )

    validation_models = comparison["models"]

    backtest_mean_mape = (
        backtest["mean_mape_by_model"]
    )

    backtest_mape_std = (
        backtest["mape_std_by_model"]
    )

    eligible_models = [
        model
        for model in CANDIDATE_MODELS
        if model in validation_models
        and model in backtest_mean_mape
        and model in backtest_mape_std
    ]

    if len(eligible_models) < 2:
        raise ValueError(
            "Not enough candidate models "
            "for model selection."
        )

    best_backtest_mape = min(
        backtest_mean_mape[model]
        for model in eligible_models
    )

    near_best_models = [
        model
        for model in eligible_models
        if (
            backtest_mean_mape[model]
            - best_backtest_mape
        )
        <= BACKTEST_MAPE_TOLERANCE
    ]

    validation_ranked = sorted(
        near_best_models,
        key=lambda model: (
            validation_models[model]["MAPE"],
            backtest_mean_mape[model],
            backtest_mape_std[model],
        ),
    )

    selected_model = validation_ranked[0]

    runner_up = next(
        model
        for model in eligible_models
        if model != selected_model
    )

    selected_validation_mape = float(
        validation_models[
            selected_model
        ]["MAPE"]
    )

    runner_up_validation_mape = float(
        validation_models[
            runner_up
        ]["MAPE"]
    )

    selected_backtest_mape = float(
        backtest_mean_mape[
            selected_model
        ]
    )

    runner_up_backtest_mape = float(
        backtest_mean_mape[
            runner_up
        ]
    )

    selected_backtest_std = float(
        backtest_mape_std[
            selected_model
        ]
    )

    runner_up_backtest_std = float(
        backtest_mape_std[
            runner_up
        ]
    )

    reason = (
        f"{selected_model} selected because all models "
        f"within {BACKTEST_MAPE_TOLERANCE:.3f} percentage "
        "points of the best mean backtest MAPE were treated "
        "as practically competitive; among those models, "
        f"{selected_model} had the best holdout validation "
        "MAPE. The final test set remained untouched."
    )

    selection_artifact = {
        "selected_model": selected_model,
        "runner_up": runner_up,
        "policy": {
            "primary_evidence":
                "expanding_window_mean_mape",
            "near_tie_tolerance_percentage_points":
                BACKTEST_MAPE_TOLERANCE,
            "tie_breaker":
                "holdout_validation_mape",
            "secondary_tie_breaker":
                "backtest_mape_std",
        },
        "selected_model_metrics": {
            "validation_mape":
                selected_validation_mape,
            "backtest_mean_mape":
                selected_backtest_mape,
            "backtest_mape_std":
                selected_backtest_std,
        },
        "runner_up_metrics": {
            "validation_mape":
                runner_up_validation_mape,
            "backtest_mean_mape":
                runner_up_backtest_mape,
            "backtest_mape_std":
                runner_up_backtest_std,
        },
        "reason": reason,
        "test_set_evaluated": False,
    }

    save_json(
        selection_artifact,
        SELECTION_PATH,
    )

    print("\nModel Selection Decision")
    print("-" * 60)

    print(
        f"Selected model: {selected_model}"
    )

    print(
        f"Runner-up: {runner_up}"
    )

    print(
        f"\n{selected_model} validation MAPE: "
        f"{selected_validation_mape:.4f}%"
    )

    print(
        f"{runner_up} validation MAPE: "
        f"{runner_up_validation_mape:.4f}%"
    )

    print(
        f"\n{selected_model} backtest mean MAPE: "
        f"{selected_backtest_mape:.4f}%"
    )

    print(
        f"{runner_up} backtest mean MAPE: "
        f"{runner_up_backtest_mape:.4f}%"
    )

    print(
        f"\n{selected_model} backtest MAPE std: "
        f"{selected_backtest_std:.4f}"
    )

    print(
        f"{runner_up} backtest MAPE std: "
        f"{runner_up_backtest_std:.4f}"
    )

    print("\nSelection reason:")
    print(reason)

    print(
        f"\nSelection artifact saved to: "
        f"{SELECTION_PATH}"
    )

    print(
        "\nFinal test set remains untouched."
    )


if __name__ == "__main__":
    main()