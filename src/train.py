import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
]

CATEGORICAL_FEATURES = ["DISTRICT"]


def load_data():
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

    X = df[FEATURES + CATEGORICAL_FEATURES]
    y = df["TARGET"]

    train_mask = df["YEAR"] <= 2018
    validation_mask = (df["YEAR"] >= 2019) & (df["YEAR"] <= 2021)
    test_mask = df["YEAR"] >= 2022

    return (
        X.loc[train_mask],
        X.loc[validation_mask],
        X.loc[test_mask],
        y.loc[train_mask],
        y.loc[validation_mask],
        y.loc[test_mask],
    )


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "f1": f1_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def main():
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = load_data()

    model = build_pipeline()

    mlflow.set_experiment("heatwave-monitoring")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        validation_metrics = evaluate(
            model,
            X_validation,
            y_validation,
        )

        test_metrics = evaluate(
            model,
            X_test,
            y_test,
        )

        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "n_estimators": 200,
                "class_weight": "balanced",
                "random_state": 42,
                "f1_threshold": F1_THRESHOLD,
            }
        )

        mlflow.log_metrics(
            {
                "validation_f1": validation_metrics["f1"],
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "validation_roc_auc": validation_metrics["roc_auc"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
        )

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print("\nValidation Results")
        print("------------------")
        for name, value in validation_metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nTest Results")
        print("------------")
        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")

        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"F1 threshold: {F1_THRESHOLD}")


if __name__ == "__main__":
    main()
