import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

MODEL_PATH = "models/heatwave_model.pkl"
DATA_PATH = "data/Rajasthan_Heatwave_2006_2025.csv"


def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_is_created_successfully():
    model = joblib.load(MODEL_PATH)

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "model" in model.named_steps


def test_model_produces_valid_prediction():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    features = [
        "WIND_U10",
        "WIND_V10",
        "MSLP",
        "BLH",
        "GEOP",
        "TEMP2M",
        "TMAX",
        "TMIN",
        "DEW2M",
        "CLOUD",
        "RAIN",
        "SRAD",
        "EVAP",
        "SOILT1",
        "SOILM1",
        "LAI",
        "LAT",
        "LON",
        "YEAR",
        "MONTH",
        "DAY",
        "DISTRICT",
    ]

    sample = df[features].iloc[[0]]

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]
