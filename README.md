# AQI Prediction App

This project trains a machine learning model to predict **US_AQI** from location, time, weather, and pollutant/gas features, then serves predictions via a localhost Flask web app.

## 1) Train the model

```powershell
cd c:\Users\rmuki\OneDrive\Desktop\dsci
.\.venv\Scripts\python.exe train_model.py
```

This creates:
- `aqi_model.joblib` (trained preprocessing + model pipeline)
- `metrics.json` (regression + classification metrics)

## 2) Run the web app

```powershell
cd c:\Users\rmuki\OneDrive\Desktop\dsci
.\.venv\Scripts\python.exe app.py
```

Open:
- http://127.0.0.1:5000/

## Inputs expected
- Place: `City`, `State`
- Time: `Month`, `Hour`, `Time_of_Day`, `Season`
- Weather: `Temp_2m_C`, `Humidity_Percent`, `Wind_Speed_10m_kmh`, `Pressure_MSL_hPa`, `UV_Index`
- Pollutants/gases: `PM2_5_ugm3`, `PM10_ugm3`, `CO_ugm3`, `NO2_ugm3`, `SO2_ugm3`, `O3_ugm3`, `NH3_ugm3`, `Dust_ugm3`

## Reported metrics
- Regression: MAE, RMSE, R2
- Classification-style (AQI bands derived from predicted AQI): Accuracy, Precision (weighted), Recall (weighted), F1 (weighted)
