import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    """
    Split time-series data chronologically into
    train, validation, and test sets.
    """

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "train_ratio + val_ratio must be less than 1"
        )

    if "Datetime" not in df.columns:
        raise ValueError(
            "Missing required column: Datetime"
        )

    result = df.copy()

    result["Datetime"] = pd.to_datetime(
        result["Datetime"]
    )

    result = (
        result
        .sort_values("Datetime")
        .reset_index(drop=True)
    )

    n_rows = len(result)

    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)

    train_df = (
        result.iloc[:train_end]
        .copy()
        .reset_index(drop=True)
    )

    val_df = (
        result.iloc[train_end:val_end]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        result.iloc[val_end:]
        .copy()
        .reset_index(drop=True)
    )

    return train_df, val_df, test_df