import pandas as pd
import pytest

from src.split import chronological_split


def make_sample_data(n_rows=100):
    return pd.DataFrame({
        "Datetime": pd.date_range(
            start="2024-01-01",
            periods=n_rows,
            freq="h",
        ),
        "AEP_MW": range(n_rows),
    })


def test_chronological_split_sizes():
    df = make_sample_data(100)

    train_df, val_df, test_df = chronological_split(df)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15


def test_chronological_split_preserves_time_order():
    df = make_sample_data(100)

    train_df, val_df, test_df = chronological_split(df)

    assert train_df["Datetime"].max() < val_df["Datetime"].min()
    assert val_df["Datetime"].max() < test_df["Datetime"].min()


def test_chronological_split_sorts_unsorted_input():
    df = make_sample_data(100)

    shuffled_df = df.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    train_df, val_df, test_df = chronological_split(
        shuffled_df
    )

    assert train_df["Datetime"].is_monotonic_increasing
    assert val_df["Datetime"].is_monotonic_increasing
    assert test_df["Datetime"].is_monotonic_increasing


def test_split_contains_every_row_exactly_once():
    df = make_sample_data(101)

    train_df, val_df, test_df = chronological_split(df)

    assert (
        len(train_df)
        + len(val_df)
        + len(test_df)
        == len(df)
    )


def test_invalid_ratios_raise_error():
    df = make_sample_data()

    with pytest.raises(ValueError):
        chronological_split(
            df,
            train_ratio=0.90,
            val_ratio=0.20,
        )


def test_missing_datetime_raises_error():
    df = pd.DataFrame({
        "AEP_MW": [1000, 1100, 1200]
    })

    with pytest.raises(
        ValueError,
        match="Missing required column: Datetime",
    ):
        chronological_split(df)