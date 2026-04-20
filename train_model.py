import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("INDIA_AQI_COMPLETE_20251126.csv")
MODEL_PATH = Path("aqi_model.joblib")
METRICS_PATH = Path("metrics.json")

TARGET = "US_AQI"
FEATURES = [
    "Temp_2m_C",
    "Humidity_Percent",
    "Wind_Speed_10m_kmh",
    "Pressure_MSL_hPa",
    "UV_Index",
    "PM2_5_ugm3",
    "PM10_ugm3",
    "CO_ugm3",
    "NO2_ugm3",
    "SO2_ugm3",
    "O3_ugm3",
    "NH3_ugm3",
    "Dust_ugm3",
]

CATEGORICAL_FEATURES = []

NUMERICAL_FEATURES = [
    col for col in FEATURES if col not in CATEGORICAL_FEATURES
]


def to_aqi_band(values: np.ndarray) -> np.ndarray:
    bins = [-np.inf, 50, 100, 150, 200, 300, np.inf]
    labels = np.array(["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"])
    idx = np.digitize(values, bins, right=True) - 1
    return labels[idx]


def main() -> None:
    cols = FEATURES + [TARGET]
    df = pd.read_csv(DATA_PATH, usecols=cols)

    df = df.dropna(subset=[TARGET])
    # Keep training practical for local machines while retaining a large sample.
    sample_size = min(60_000, len(df))
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    X = df[FEATURES]
    y = pd.to_numeric(df[TARGET], errors="coerce")

    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0.0,
                    keep_empty_features=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=60,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    y_test_band = to_aqi_band(y_test.to_numpy())
    y_pred_band = to_aqi_band(y_pred)

    cls_accuracy = float(accuracy_score(y_test_band, y_pred_band))
    cls_precision = float(precision_score(y_test_band, y_pred_band, average="weighted", zero_division=0))
    cls_recall = float(recall_score(y_test_band, y_pred_band, average="weighted", zero_division=0))
    cls_f1 = float(f1_score(y_test_band, y_pred_band, average="weighted", zero_division=0))

    metrics = {
        "dataset_rows_used": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "regression": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        },
        "classification_from_regression": {
            "aqi_bands": ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"],
            "accuracy": cls_accuracy,
            "precision_weighted": cls_precision,
            "recall_weighted": cls_recall,
            "f1_weighted": cls_f1,
        },
    }

    joblib.dump({"pipeline": pipeline, "features": FEATURES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Model saved to", MODEL_PATH)
    print("Metrics saved to", METRICS_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
