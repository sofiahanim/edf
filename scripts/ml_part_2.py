import os
import pandas as pd
from darts import TimeSeries
from darts.models import Theta
from neuralprophet import NeuralProphet
from pycaret.time_series import setup, compare_models, predict_model
import warnings
import logging
import shutil
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import time
import numpy as np
from datetime import datetime
import sys
from io import StringIO

# Define a helper function for MAPE calculation
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("neuralprophet").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

print("Starting script without warnings...")
print("Current working directory:", os.getcwd())
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

# Rename columns for compatibility with models:
# 'datetime' to 'ds' (time column required by models)
# 'value' to 'y' (target variable for forecasting)
data = data.rename(columns={'datetime': 'ds', 'value': 'y'})
selected_columns = ['ds', 'y', 'temp', 'feelslike', 'humidity', 'windspeed', 
                    'cloudcover', 'solaradiation', 'precip', 'preciptype', 'is_holiday']
data = data.drop_duplicates(subset='ds')
print("Preview of prepared data:")
print(data.head())

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

def save_csv_append(file_path, df):
    if df.empty:
        print(f"No data to append for {file_path}")
        return
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        df = pd.concat([existing_data, df], ignore_index=True)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}. Total rows: {len(df)}")


if not all(col in data.columns for col in selected_columns) or data.empty:
    raise ValueError("Dataset is empty or missing required columns ('ds', 'y').")
if data.isnull().any().any():
    raise ValueError("Dataset contains missing values.")

print(f"Input data validated. {len(data)} rows loaded.")
print(f"Columns in dataset: {data.columns.tolist()}")

predicted_at = datetime.now()
# Forecasting with Darts Theta
try:
    print("Running Darts with Theta Model...")
    series = TimeSeries.from_dataframe(data[['ds', 'y']], time_col="ds", value_cols="y", freq="H")
    model_darts_theta = Theta()
    model_darts_theta.fit(series)
    forecast_darts_theta = model_darts_theta.predict(14 * 24)
    darts_forecast = forecast_darts_theta.pd_dataframe().reset_index()
    darts_forecast['predicted_at'] = predicted_at
    darts_results_file = os.path.join(evaluation_dir, 'darts_theta_forecast.csv')
    save_csv_append(darts_results_file, darts_forecast)
    print("Darts Theta forecast completed.")
except Exception as e:
    print(f"Darts Theta error: {e}")

# Forecasting with NeuralProphet
try:
    print("Running NeuralProphet...")
    model_np = NeuralProphet(batch_size=32, epochs=30)
    
    # Fit the model with the ID column
    model_np.fit(data[['ds', 'y', 'series_id']], freq="H")
    
    # Create a future dataframe with the ID column
    future = model_np.make_future_dataframe(data[['ds', 'y', 'series_id']], periods=14 * 24)
    if future.empty:
        raise ValueError("NeuralProphet future dataframe is empty.")
    
    # Predict and add 'predicted_at'
    forecast_np = model_np.predict(future)
    if forecast_np.empty:
        raise ValueError("NeuralProphet forecast resulted in an empty DataFrame.")
    forecast_np['predicted_at'] = predicted_at

    # Save forecast
    neuralprophet_results_file = os.path.join(evaluation_dir, 'neuralprophet_forecast.csv')
    save_csv_append(neuralprophet_results_file, forecast_np)
    print(f"NeuralProphet forecast saved to {neuralprophet_results_file}.")
except Exception as e:
    print(f"NeuralProphet error: {e}")

# Forecasting with PyCaret
try:
    print("Running PyCaret...")
    sys.stdout = StringIO()  # Redirect stdout for PyCaret logs
    setup(data=data, target='y', session_id=123, fold=2, fh=48)
    best_model = compare_models()
    future_pycaret = predict_model(best_model, fh=14 * 24)
    sys.stdout = sys.__stdout__  # Restore stdout
    future_pycaret['predicted_at'] = predicted_at
    pycaret_results_file = os.path.join(evaluation_dir, 'pycaret_forecast.csv')
    save_csv_append(pycaret_results_file, future_pycaret)
    print("PyCaret forecast completed.")
except Exception as e:
    print(f"PyCaret error: {e}")

# Generate Summary
try:
    print("Generating summary...")
    predict_for = pd.date_range(start=data['ds'].max(), periods=14 * 24, freq='H')
    summary = pd.DataFrame({'predict_at': predicted_at, 'predict_for': predict_for})
    if 'forecast_darts_theta' in locals():
        summary['Darts Theta'] = forecast_darts_theta.pd_series().values[:len(predict_for)]
    if 'forecast_np' in locals():
        summary['NeuralProphet'] = forecast_np['yhat1'].values[:len(predict_for)]
    if 'future_pycaret' in locals() and 'Label' in future_pycaret.columns:
        summary['PyCaret'] = future_pycaret['Label'].values[:len(predict_for)]
    summary_file = os.path.join(evaluation_dir, 'summary_forecast.csv')
    save_csv_append(summary_file, summary)
    print("Summary saved.")
except Exception as e:
    print(f"Error saving summary: {e}")

# Generate training information
try:
    print("Collecting training information...")
    training_info = {
        "Model": ["Darts Theta", "NeuralProphet", "PyCaret"],
        "Parameters": ["Default", "batch_size=32, epochs=30", "auto-optimized"],
        "Metrics": [None, None, None],
        "Training Duration (seconds)": [None, None, None],
        "Details": ["", "", pycaret_output[:1000] if 'pycaret_output' in locals() else ""]
    }

    # Validate true_values
    forecast_horizon = 14 * 24  # Forecast horizon: 14 days hourly
    true_values = data['y'][-forecast_horizon:].values
    if len(true_values) < forecast_horizon:
        raise ValueError(f"Insufficient data for forecast horizon. Required: {forecast_horizon}, Available: {len(true_values)}")

    # Metrics for Darts Theta
    if 'forecast_darts_theta' in locals():
        start_time = time.time()
        predicted_values_darts = forecast_darts_theta.pd_series().values[:len(true_values)]
        mae_darts = mean_absolute_error(true_values, predicted_values_darts)
        rmse_darts = root_mean_squared_error(true_values, predicted_values_darts, squared=False)
        mape_darts = mean_absolute_percentage_error(true_values, predicted_values_darts)
        end_time = time.time()
        training_info["Metrics"][0] = f"MAPE: {mape_darts:.2f}%, MAE: {mae_darts:.2f}, RMSE: {rmse_darts:.2f}"
        training_info["Training Duration (seconds)"][0] = end_time - start_time

    # Metrics for NeuralProphet
    if 'forecast_np' in locals():
        start_time = time.time()
        predicted_values_np = forecast_np['yhat1'].values[:len(true_values)]
        mae_np = mean_absolute_error(true_values, predicted_values_np)
        rmse_np = root_mean_squared_error(true_values, predicted_values_np, squared=False)
        mape_np = mean_absolute_percentage_error(true_values, predicted_values_np)
        end_time = time.time()
        training_info["Metrics"][1] = f"MAPE: {mape_np:.2f}%, MAE: {mae_np:.2f}, RMSE: {rmse_np:.2f}"
        training_info["Training Duration (seconds)"][1] = end_time - start_time
    # Metrics for PyCaret
    if 'future_pycaret' in locals() and 'Label' in future_pycaret.columns:
        # Add predicted_at and save forecast
        future_pycaret['predicted_at'] = predicted_at
        save_csv_append(pycaret_results_file, future_pycaret)
        
        # Calculate metrics
        start_time = time.time()
        predicted_values_pycaret = future_pycaret['Label'].values[:len(true_values)]
        mae_pycaret = mean_absolute_error(true_values, predicted_values_pycaret)
        rmse_pycaret = root_mean_squared_error(true_values, predicted_values_pycaret, squared=False)
        mape_pycaret = mean_absolute_percentage_error(true_values, predicted_values_pycaret)
        end_time = time.time()
        
        # Log metrics and training duration
        training_info["Metrics"][2] = f"MAPE: {mape_pycaret:.2f}%, MAE: {mae_pycaret:.2f}, RMSE: {rmse_pycaret:.2f}"
        training_info["Training Duration (seconds)"][2] = end_time - start_time


    # Save training information
    training_info_df = pd.DataFrame(training_info)
    training_info_file = os.path.join(training_folder, 'training_info.csv')
    save_csv_append(training_info_file, training_info_df)
    print(f"Training information saved to {training_info_file}")
except Exception as e:
    print(f"Error saving training information: {e}")


# Clean-up
for log_dir in [os.path.join(base_dir, 'lightning_logs'), os.path.join(base_dir, 'catboost_info')]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"Deleted {log_dir}")