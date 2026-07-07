import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the Datetime column.
    """

    result = df.copy()

    if "Datetime" not in result.columns:
        raise ValueError("Missing required column: Datetime")

    result["Datetime"] = pd.to_datetime(result["Datetime"])

    result["hour"] = result["Datetime"].dt.hour
    result["day_of_week"] = result["Datetime"].dt.dayofweek
    result["month"] = result["Datetime"].dt.month
    result["day_of_year"] = result["Datetime"].dt.dayofyear
    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)

    return result