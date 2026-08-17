import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import f1_score

DATA_PATH = "data/Rajasthan_Heatwave_2006_2025.csv"
MODEL_PATH = "models/heatwave_model.pkl"
F1_THRESHOLD = float(os.getenv("F1_THRESHOLD", "0.60"))

FEATURES = [
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


def main():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    df["DATE"] = pd.to_datetime(
        df[["YEAR", "MONTH", "DAY"]].rename(
            columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}
        )
    )

    df = df.sort_values(["DATE", "DISTRICT"])

    df["TARGET"] = df.groupby("DISTRICT")["HEATWAVE"].shift(-1)

    df = df.dropna(subset=["TARGET"])
    df["TARGET"] = df["TARGET"].astype(int)

    test_mask = df["YEAR"] >= 2022

    X_test = df.loc[test_mask, FEATURES]
    y_test = df.loc[test_mask, "TARGET"]

    predictions = model.predict(X_test)

    f1 = f1_score(y_test, predictions)

    print(f"Test F1: {f1:.4f}")
    print(f"Required F1: {F1_THRESHOLD:.4f}")

    if f1 < F1_THRESHOLD:
        print("MODEL VALIDATION FAILED")
        sys.exit(1)

    print("MODEL VALIDATION PASSED")


if __name__ == "__main__":
    main()
