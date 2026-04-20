import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

MODEL_PATH = Path("aqi_model.joblib")
METRICS_PATH = Path("metrics.json")

app = Flask(__name__)

loaded = joblib.load(MODEL_PATH)
pipeline = loaded["pipeline"]
FEATURES = loaded["features"]

CATEGORICAL_FEATURES = []
NUMERICAL_FEATURES = [col for col in FEATURES if col not in CATEGORICAL_FEATURES]

print("\n=== APP INITIALIZATION ===")
print("Loaded model features:", FEATURES)
print("Categorical features:", CATEGORICAL_FEATURES)
print("Numerical features:", NUMERICAL_FEATURES)
print("=== END INIT ===\n")

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
else:
    metrics = {}


def to_float_or_nan(value: str):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    submitted = {}
    error = None

    if request.method == "POST":
        submitted = {
            "Temp_2m_C": to_float_or_nan(request.form.get("Temp_2m_C", "")),
            "Humidity_Percent": to_float_or_nan(request.form.get("Humidity_Percent", "")),
            "Wind_Speed_10m_kmh": to_float_or_nan(request.form.get("Wind_Speed_10m_kmh", "")),
            "Pressure_MSL_hPa": to_float_or_nan(request.form.get("Pressure_MSL_hPa", "")),
            "UV_Index": to_float_or_nan(request.form.get("UV_Index", "")),
            "PM2_5_ugm3": to_float_or_nan(request.form.get("PM2_5_ugm3", "")),
            "PM10_ugm3": to_float_or_nan(request.form.get("PM10_ugm3", "")),
            "CO_ugm3": to_float_or_nan(request.form.get("CO_ugm3", "")),
            "NO2_ugm3": to_float_or_nan(request.form.get("NO2_ugm3", "")),
            "SO2_ugm3": to_float_or_nan(request.form.get("SO2_ugm3", "")),
            "O3_ugm3": to_float_or_nan(request.form.get("O3_ugm3", "")),
            "NH3_ugm3": to_float_or_nan(request.form.get("NH3_ugm3", "")),
            "Dust_ugm3": to_float_or_nan(request.form.get("Dust_ugm3", "")),
        }

        try:
            frame = pd.DataFrame([submitted], columns=FEATURES)
            print(f"\n=== PREDICTION DEBUG (attempt) ===")
            print(f"Submitted keys: {list(submitted.keys())}")
            print(f"Expected FEATURES: {FEATURES}")
            print(f"Keys match? {set(submitted.keys()) == set(FEATURES)}")
            
            print(f"\nDataFrame dtypes BEFORE processing:")
            for col in FEATURES:
                print(f"  {col}: {frame[col].dtype}")
            
            # All features are numeric, just ensure they're float64
            frame = frame.astype("float64")
            
            print(f"\nDataFrame dtypes AFTER float64 cast:")
            for col in FEATURES:
                print(f"  {col}: {frame[col].dtype}")
            
            print(f"\nCalling pipeline.predict()...")
            prediction = float(pipeline.predict(frame)[0])
            print(f"Prediction successful: {prediction}")
            print("=== END PREDICTION DEBUG ===\n")
            
        except Exception as exc:
            import traceback
            tb_str = traceback.format_exc()
            print("\n=== PREDICTION ERROR ===")
            print(f"Exception type: {type(exc).__name__}")
            print(f"Exception message: {str(exc)}")
            print(f"\nFull traceback:\n{tb_str}")
            try:
                print("\nDataFrame at error time:")
                print(frame)
                print("\nDataFrame dtypes at error:")
                for col in FEATURES:
                    print(f"  {col}: {frame[col].dtype}")
            except Exception as inner_exc:
                print(f"Could not print DataFrame: {inner_exc}")
            print("=== END ERROR DEBUG ===\n")
            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        submitted=submitted,
        error=error,
        metrics=metrics,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
