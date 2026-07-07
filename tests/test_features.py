import pandas as pd
import pytest

from src.features import create_time_features


def test_create_time_features():
    df = pd.DataFrame({
        "Datetime": [
            "2024-01-01 14:00:00",
            "2024-01-06 09:00:00",
        ],
        "AEP_MW": [12000, 13000],
    })

    result = create_time_features(df)

    assert result.loc[0, "hour"] == 14
    assert result.loc[0, "month"] == 1
    assert result.loc[0, "day_of_week"] == 0
    assert result.loc[0, "is_weekend"] == 0

    assert result.loc[1, "hour"] == 9
    assert result.loc[1, "is_weekend"] == 1


def test_create_time_features_missing_datetime():
    df = pd.DataFrame({
        "AEP_MW": [12000, 13000]
    })

    with pytest.raises(
        ValueError,
        match="Missing required column: Datetime"
    ):
        create_time_features(df)