from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_energy_data


def test_load_energy_data_sorts_and_removes_duplicates(tmp_path: Path):
    test_file = tmp_path / "energy.csv"

    input_df = pd.DataFrame(
        {
            "Datetime": [
                "2024-01-01 02:00:00",
                "2024-01-01 01:00:00",
                "2024-01-01 01:00:00",
            ],
            "AEP_MW": [1200, 1000, 1000],
        }
    )
    input_df.to_csv(test_file, index=False)

    result = load_energy_data(test_file)

    assert len(result) == 2
    assert result["Datetime"].is_monotonic_increasing
    assert result["Datetime"].duplicated().sum() == 0


def test_load_energy_data_raises_for_missing_file(tmp_path: Path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_energy_data(missing_file)


def test_load_energy_data_raises_for_missing_columns(tmp_path: Path):
    test_file = tmp_path / "bad_energy.csv"

    bad_df = pd.DataFrame(
        {
            "Datetime": ["2024-01-01 00:00:00"],
            "WrongColumn": [1000],
        }
    )
    bad_df.to_csv(test_file, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_energy_data(test_file)