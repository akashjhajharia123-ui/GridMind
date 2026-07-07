import pandas as pd

from src.model import (
    FEATURE_COLUMNS,
    predict_demand,
    save_model,
    train_model,
)


def make_sample_data():
    return pd.DataFrame({
        "hour": [0, 6, 12, 18, 23],
        "day_of_week": [0, 1, 2, 3, 4],
        "month": [1, 1, 1, 1, 1],
        "day_of_year": [1, 2, 3, 4, 5],
        "is_weekend": [0, 0, 0, 0, 0],
        "AEP_MW": [10000, 11000, 14000, 15000, 12000],
    })


def test_train_model():
    df = make_sample_data()

    model = train_model(df)

    assert model is not None
    assert hasattr(model, "predict")


def test_predict_demand():
    df = make_sample_data()

    model = train_model(df)
    predictions = predict_demand(model, df)

    assert len(predictions) == len(df)
    assert all(prediction > 0 for prediction in predictions)


def test_save_model(tmp_path):
    df = make_sample_data()
    model = train_model(df)

    model_path = tmp_path / "test_model.joblib"

    save_model(model, str(model_path))

    assert model_path.exists()