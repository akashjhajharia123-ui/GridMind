from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
]


def train_model(
    df: pd.DataFrame,
    target_column: str = "AEP_MW",
) -> RandomForestRegressor:
    """
    Train an energy demand forecasting model.
    """

    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Missing target column: {target_column}"
        )

    X = df[FEATURE_COLUMNS]
    y = df[target_column]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    return model


def predict_demand(
    model: RandomForestRegressor,
    df: pd.DataFrame,
):
    """
    Predict energy demand.
    """

    X = df[FEATURE_COLUMNS]

    return model.predict(X)


def save_model(
    model: RandomForestRegressor,
    path: str = "models/gridmind_model.joblib",
) -> None:
    """
    Save trained model to disk.
    """

    model_path = Path(path)
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, model_path)