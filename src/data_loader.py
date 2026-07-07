from pathlib import Path

import pandas as pd


def load_energy_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load hourly energy-consumption data from a CSV file.

    Expected columns:
    - Datetime
    - AEP_MW
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path.resolve()}"
        )

    df = pd.read_csv(file_path)

    required_columns = {"Datetime", "AEP_MW"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["Datetime"] = pd.to_datetime(
        df["Datetime"],
        errors="raise"
    )

    df = (
        df.sort_values("Datetime")
        .drop_duplicates(subset="Datetime")
        .reset_index(drop=True)
    )

    return df