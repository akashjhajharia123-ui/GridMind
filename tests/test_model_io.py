import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.model_io import load_model, save_model


def test_save_model_creates_file(tmp_path):
    model = RandomForestRegressor(
        n_estimators=5,
        random_state=42,
    )

    model_path = tmp_path / "nested" / "model.joblib"

    saved_path = save_model(model, model_path)

    assert model_path.exists()
    assert model_path.is_file()
    assert saved_path == model_path.resolve()


def test_loaded_model_preserves_predictions(tmp_path):
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
    ])

    y = np.array([
        10.0,
        20.0,
        30.0,
        40.0,
    ])

    model = RandomForestRegressor(
        n_estimators=10,
        random_state=42,
    )
    model.fit(X, y)

    expected_predictions = model.predict(X)

    model_path = tmp_path / "model.joblib"
    save_model(model, model_path)

    loaded_model = load_model(model_path)
    actual_predictions = loaded_model.predict(X)

    np.testing.assert_allclose(
        actual_predictions,
        expected_predictions,
    )


def test_load_model_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(
        FileNotFoundError,
        match="Model file not found",
    ):
        load_model(missing_path)