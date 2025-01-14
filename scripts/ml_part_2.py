import os
import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.utils.missing_values import fill_missing_values
from autogluon.timeseries import TimeSeriesPredictor
from pycaret.time_series import *

# Define directories
print("Current working directory:", os.getcwd())
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
merge_dir = os.path.join(data_dir, 'merge')
input_file = os.path.join(base_dir, 'data', 'merge', 'allyears.csv')

print(input_file)

# Load dataset
data = pd.read_csv(input_file)
data['hour'] = data['hour'].astype(str).str.zfill(2)
data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['hour'] + ':00:00')
data = data[['datetime', 'electric']].rename(columns={'datetime': 'ds', 'electric': 'y'})

# Ensure there are no duplicates
data = data.drop_duplicates(subset='ds')

# Darts
try:
    print("Running Darts...")
    # Fill missing dates for consistency
    data = data.set_index('ds')
    data = fill_missing_values(data, freq='H')
    data.reset_index(inplace=True)

    # Convert to Darts TimeSeries
    series = TimeSeries.from_dataframe(data, time_col="ds", value_col="y", fill_missing_dates=True, freq="H")

    # Train a Darts model
    model_darts = ExponentialSmoothing()
    model_darts.fit(series)
    forecast_darts = model_darts.predict(14 * 24)  # Predict next 14 days (hourly)

    print("Darts forecast completed.")
except Exception as e:
    print(f"Darts error: {e}")

# AutoGluon
try:
    print("Running AutoGluon...")
    predictor = TimeSeriesPredictor(label="y")
    predictor.fit(data)
    forecast_autogluon = predictor.predict(14 * 24)  # Predict next 14 days
    print("AutoGluon forecast completed.")
except Exception as e:
    print(f"AutoGluon error: {e}")

# PyCaret
try:
    print("Running PyCaret...")
    setup(data=data, target='y', session_id=123)
    best_model = compare_models()
    future_pycaret = predict_model(best_model, fh=14 * 24)
    print("PyCaret forecast completed.")
except Exception as e:
    print(f"PyCaret error: {e}")

# Save results
output_file = os.path.join(merge_dir, 'future_14_day_predictions.csv')
try:
    results = pd.DataFrame({
        'predict_at': pd.Timestamp.now(),
        'Darts': forecast_darts.pd_series() if 'forecast_darts' in locals() else None,
        'AutoGluon': forecast_autogluon if 'forecast_autogluon' in locals() else None,
        'PyCaret': future_pycaret if 'future_pycaret' in locals() else None
    })
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
except Exception as e:
    print(f"Error saving results: {e}")
