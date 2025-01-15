import os
import pandas as pd
from darts import TimeSeries
from darts.models import Theta
from fbprophet import Prophet  # Change from NeuralProphet to Prophet
from pycaret.time_series import setup, compare_models, predict_model
import warnings
import logging
import shutil
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import time
import numpy as np
from datetime import datetime
from io import StringIO
import sys

# Define a helper function for MAPE calculation
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Suppress warnings and reduce log verbosity
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("fbprophet").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

print("Starting script without warnings...")
print("Current working directory:", os.getcwd())

# Define directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
merge_dir = os.path.join(data_dir, 'merge')
evaluation_dir = os.path.join(base_dir, 'evaluation')
training_folder = os.path.join(base_dir, 'training')

# Ensure necessary directories exist
os.makedirs(merge_dir, exist_ok=True)
os.makedirs(evaluation_dir, exist_ok=True)
os.makedirs(training_folder, exist_ok=True)

# Load the dataset
input_file = os.path.join(merge_dir, 'allyears.csv')
print(f"Reading input file: {input_file}")
data = pd.read_csv(input_file, parse_dates=['ds'])

if 'id' not in data.columns:
    data['ID'] = range(1, len(data) + 1)

# Rename columns for compatibility
data = data.rename(columns={'datetime': 'ds', 'value': 'y'})
selected_columns = ['ID', 'ds', 'y', 'temp', 'feelslike', 'humidity', 'windspeed', 
                    'cloudcover', 'solaradiation', 'precip', 'preciptype', 'is_holiday']
data = data.drop_duplicates(subset='ds')
print("Preview of prepared data:")
print(data.head())

# Validate dataset
if data['ds'].isnull().any() or data['y'].isnull().any():
    raise ValueError("Input data contains missing values in 'ds' or 'y'.")
if data.empty or 'ds' not in data.columns or 'y' not in data.columns:
    raise ValueError("Input data is empty or missing required columns ('ds', 'y').")
print(f"Input data validated. {len(data)} rows loaded.")
print(f"Input file size: {os.path.getsize(input_file)} bytes")
print(f"Columns in dataset: {data.columns.tolist()}")

# Initialize PyCaret log capture
pycaret_log = []
original_stdout = sys.stdout
sys.stdout = StringIO()

# Helper function to save data to CSV
def save_csv_append(file_path, df):
    if df.empty:
        print(f"No data to append for {file_path}")
        return
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        df = pd.concat([existing_data, df], ignore_index=True)
    df.to_csv(file_path, index=False)
    print(f"Data appended to {file_path}. Total rows: {len(df)}")

predicted_at = datetime.now()

# Forecasting with Darts Theta Model
try:
    print("Running Darts with Theta Model...")
    series = TimeSeries.from_dataframe(data[['ds', 'y']], time_col="ds", value_cols="y", freq="H")
    model_darts_theta = Theta()
    model_darts_theta.fit(series)
    forecast_darts_theta = model_darts_theta.predict(14 * 24)
    darts_forecast = forecast_darts_theta.pd_dataframe().reset_index()
    if darts_forecast.empty:
        raise ValueError("Darts Theta forecast resulted in an empty DataFrame.")
    darts_forecast['predicted_at'] = predicted_at
    darts_results_file = os.path.join(evaluation_dir, 'darts_theta_forecast.csv')
    save_csv_append(darts_results_file, darts_forecast)
    print("Darts Theta forecast completed.")
except Exception as e:
    print(f"Darts Theta error: {e}")

# Forecasting with Prophet
try:
    print("Running Prophet...")
    model_prophet = Prophet()
    model_prophet.fit(data[['ds', 'y']])
    future = model_prophet.make_future_dataframe(periods=14 * 24, freq='H')
    forecast_prophet = model_prophet.predict(future)
    if forecast_prophet.empty:
        raise ValueError("Prophet forecast resulted in an empty DataFrame.")
    forecast_prophet['predicted_at'] = predicted_at
    prophet_results_file = os.path.join(evaluation_dir, 'prophet_forecast.csv')
    save_csv_append(prophet_results_file, forecast_prophet)
    print(f"Prophet forecast saved to {prophet_results_file}.")
except Exception as e:
    print(f"Prophet error: {e}")

# Forecasting with PyCaret
try:
    print("Running PyCaret...")
    sys.stdout = StringIO()  # Redirect stdout to capture PyCaret logs
    setup(data=data, target='y', session_id=123, fold=2, fh=48)
    best_model = compare_models()
    future_pycaret = predict_model(best_model, fh=14 * 24)
    pycaret_output = sys.stdout.getvalue()
    sys.stdout = sys.__stdout__  # Restore stdout
    if future_pycaret.empty or 'Label' not in future_pycaret.columns:
        raise ValueError("PyCaret forecast is empty or missing 'Label' column.")
    future_pycaret['predicted_at'] = predicted_at
    pycaret_results_file = os.path.join(evaluation_dir, 'pycaret_forecast.csv')
    save_csv_append(pycaret_results_file, future_pycaret)
    print("PyCaret forecast completed.")
except Exception as e:
    print(f"PyCaret error: {e}")

# Generate summary of results
try:
    print("Generating summary of results...")
    predict_for = pd.date_range(start=data['ds'].max(), periods=14 * 24, freq='H')
    summary = pd.DataFrame({'predict_at': predicted_at, 'predict_for': predict_for})
    if 'forecast_darts_theta' in locals() and not forecast_darts_theta.empty:
        summary['Darts Theta'] = forecast_darts_theta.pd_series().values[:len(predict_for)]
    if 'forecast_prophet' in locals() and not forecast_prophet.empty:
        summary['Prophet'] = forecast_prophet['yhat'].values[:len(predict_for)]
    if 'future_pycaret' in locals() and not future_pycaret.empty and 'Label' in future_pycaret.columns:
        summary['PyCaret'] = future_pycaret['Label'].values[:len(predict_for)]
    summary_file = os.path.join(evaluation_dir, 'summary_forecast.csv')
    save_csv_append(summary_file, summary)
    print(f"Summary results saved to {summary_file}.")
except Exception as e:
    print(f"Error saving summary results: {e}")

# Clean up unnecessary files/folders
for log_dir in [os.path.join(base_dir, 'lightning_logs'), os.path.join(base_dir, 'catboost_info')]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"Deleted {log_dir}")
