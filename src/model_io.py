from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, path: str | Path) -> Path:
    """
    Save a trained model to disk.

    Parent directories are created automatically.
    Returns the resolved output path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)

    return output_path.resolve()


def load_model(path: str | Path) -> Any:
    """
    Load a previously saved model from disk.

    Raises FileNotFoundError if the model file does not exist.
    """
    model_path = Path(path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)