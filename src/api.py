from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

MODEL_PATH = "models/heatwave_model.pkl"
FRONTEND_PATH = Path("frontend/index.html")

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Heatwave Prediction API")

#CI-CD
class WeatherInput(BaseModel):
    WIND_U10: float
    WIND_V10: float
    MSLP: float
    BLH: float
    GEOP: float
    TEMP2M: float
    TMAX: float
    TMIN: float
    DEW2M: float
    CLOUD: float
    RAIN: float
    SRAD: float
    EVAP: float
    SOILT1: float
    SOILM1: float
    LAI: float
    LAT: float
    LON: float
    YEAR: int
    MONTH: int
    DAY: int
    DISTRICT: str


@app.get("/")
def home():
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: WeatherInput):
    input_data = pd.DataFrame([data.model_dump()])

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    return {
        "heatwave_prediction": prediction,
        "heatwave_probability": round(probability, 4),
    }
