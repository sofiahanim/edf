import os
import pandas as pd
from darts import TimeSeries
from darts.models import Theta
from neuralprophet import NeuralProphet
from pycaret.time_series import setup, compare_models, predict_model
import warnings
import logging
import shutil
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import numpy as np

# Define a helper function for MAPE calculation
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# Suppress warnings
warnings.filterwarnings("ignore")

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress logging for specific libraries
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("neuralprophet").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

print("Starting script without warnings...")

# Define directories
print("Current working directory:", os.getcwd())
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
merge_dir = os.path.join(data_dir, 'merge')
evaluation_dir = os.path.join(data_dir, 'evaluation')
training_folder = os.path.join(base_dir, 'training')

# Ensure necessary directories exist
os.makedirs(merge_dir, exist_ok=True)
os.makedirs(evaluation_dir, exist_ok=True)
os.makedirs(training_folder, exist_ok=True)

# Load the dataset
input_file = os.path.join(merge_dir, 'allyears.csv')
print(f"Reading input file: {input_file}")
data = pd.read_csv(input_file, parse_dates=['datetime'])

# Rename columns for compatibility with models:
# 'datetime' to 'ds' (time column required by models)
# 'value' to 'y' (target variable for forecasting)
data = data.rename(columns={'datetime': 'ds', 'value': 'y'})

# Select only the necessary columns for forecasting:
# 'ds' - Timestamp
# 'y' - Target variable (e.g., electricity demand)
# Other columns are features that may influence the target variable.
selected_columns = ['ds', 'y', 'temp', 'feelslike', 'humidity', 'windspeed', 
                    'cloudcover', 'solaradiation', 'precip', 'is_holiday']
data = data[selected_columns]

# Remove duplicate timestamps
data = data.drop_duplicates(subset='ds')

# Preview data
print("Preview of prepared data:")
print(data.head())

# Forecasting with Darts Theta Model
try:
    print("Running Darts with Theta Model...")
    series = TimeSeries.from_dataframe(data[['ds', 'y']], time_col="ds", value_cols="y", freq="H")
    model_darts_theta = Theta()
    model_darts_theta.fit(series)
    forecast_darts_theta = model_darts_theta.predict(14 * 24)
    darts_results_file = os.path.join(evaluation_dir, 'darts_theta_forecast.csv')
    forecast_darts_theta.pd_series().to_csv(darts_results_file, header=True)
    print("Darts Theta forecast completed.")
except Exception as e:
    print(f"Darts Theta error: {e} - Ensure the input data format is compatible with Darts.")

# Forecasting with NeuralProphet
try:
    print("Running NeuralProphet...")
    neuralprophet_data = data[['ds', 'y']]
    model_np = NeuralProphet(batch_size=32, epochs=30)
    model_np.fit(neuralprophet_data, freq="H")
    future = model_np.make_future_dataframe(neuralprophet_data, periods=14 * 24)
    forecast_np = model_np.predict(future)
    neuralprophet_results_file = os.path.join(evaluation_dir, 'neuralprophet_forecast.csv')
    forecast_np.to_csv(neuralprophet_results_file, index=False)
    print("NeuralProphet forecast completed.")
except Exception as e:
    print(f"NeuralProphet error: {e} - Check data compatibility with NeuralProphet.")


# Forecasting with PyCaret
try:
    print("Running PyCaret...")
    setup(data=data, target='y', session_id=123, fold=2, fh=48)
    best_model = compare_models()
    future_pycaret = predict_model(best_model, fh=14 * 24)
    pycaret_results_file = os.path.join(evaluation_dir, 'pycaret_forecast.csv')
    if isinstance(future_pycaret, pd.DataFrame):
        future_pycaret.to_csv(pycaret_results_file, index=False)
    print("PyCaret forecast completed.")
except Exception as e:
    print(f"PyCaret error: {e} - Check data compatibility with PyCaret.")


# Generate summary of results
try:
    print("Generating summary of results...")
    summary_file = os.path.join(evaluation_dir, 'summary_forecast.csv')
    predict_for = pd.date_range(start=data['ds'].max(), periods=14 * 24, freq='H')

    results = pd.DataFrame({'predict_at': pd.Timestamp.now(), 'predict_for': predict_for})

    # Add forecasts to the summary
    if 'forecast_darts_theta' in locals():
        results['Darts Theta'] = forecast_darts_theta.pd_series().values[:len(predict_for)]
    if 'forecast_np' in locals():
        results['NeuralProphet'] = forecast_np['yhat1'].values[:len(predict_for)]
    if 'future_pycaret' in locals() and 'Label' in future_pycaret.columns:
        results['PyCaret'] = future_pycaret['Label'].values[:len(predict_for)]

    results.to_csv(summary_file, index=False)
    print(f"Summary results saved to {summary_file} with {len(results)} rows.")
except Exception as e:
    print(f"Error saving summary results: {e}")



try:
    print("Collecting training information...")

        # Initialize training_info dictionary
    training_info = {
        "Model": ["Darts Theta", "NeuralProphet", "PyCaret"],
        "Parameters": [
            "Default", 
            "batch_size=32, epochs=30", 
            "auto-optimized"
        ],
        "Metrics": [None, None, None],  # Metrics to be calculated dynamically
        "Training Duration (seconds)": [None, None, None],
        "Epochs": [None, 30, None],  # Add epochs if applicable
    }
    
    # Define the true values for the test period (last 14 days in the dataset)
    forecast_horizon = 14 * 24  # 14 days of hourly data
    true_values = data['y'][-forecast_horizon:].values  # Last 14 days

    # Validate the length of true_values
    if len(true_values) < forecast_horizon:
        raise ValueError(f"Insufficient data for the forecast horizon. Required: {forecast_horizon}, Available: {len(true_values)}")

    if 'forecast_darts_theta' in locals():
        start_time = time.time()
        predicted_values_darts = forecast_darts_theta.pd_series().values[:len(true_values)]
        mae_darts = mean_absolute_error(true_values, predicted_values_darts)
        rmse_darts = mean_squared_error(true_values, predicted_values_darts, squared=False)
        mape_darts = mean_absolute_percentage_error(true_values, predicted_values_darts)
        end_time = time.time()
        training_info["Metrics"][0] = f"MAPE: {mape_darts:.2f}%, MAE: {mae_darts:.2f}, RMSE: {rmse_darts:.2f}"
        training_info["Training Duration (seconds)"][0] = end_time - start_time


    # Measure time and calculate metrics for NeuralProphet
    if 'forecast_np' in locals():
        start_time = time.time()
        predicted_values_np = forecast_np['yhat1'].values[:len(true_values)]
        mae_np = mean_absolute_error(true_values, predicted_values_np)
        rmse_np = mean_squared_error(true_values, predicted_values_np, squared=False)
        mape_np = mean_absolute_percentage_error(true_values, predicted_values_np)
        end_time = time.time()
        training_info["Metrics"][1] = f"MAPE: {mape_np:.2f}%, MAE: {mae_np:.2f}, RMSE: {rmse_np:.2f}"
        training_info["Training Duration (seconds)"][1] = end_time - start_time

    # Measure time and calculate metrics for PyCaret
    if 'future_pycaret' in locals() and 'Label' in future_pycaret.columns:
        start_time = time.time()
        predicted_values_pycaret = future_pycaret['Label'].values[:len(true_values)]
        mae_pycaret = mean_absolute_error(true_values, predicted_values_pycaret)
        rmse_pycaret = mean_squared_error(true_values, predicted_values_pycaret, squared=False)
        mape_pycaret = mean_absolute_percentage_error(true_values, predicted_values_pycaret)
        end_time = time.time()
        training_info["Metrics"][2] = f"MAPE: {mape_pycaret:.2f}%, MAE: {mae_pycaret:.2f}, RMSE: {rmse_pycaret:.2f}"
        training_info["Training Duration (seconds)"][2] = end_time - start_time

    # Convert to DataFrame and save
    training_info_df = pd.DataFrame(training_info)
    training_info_file = os.path.join(training_folder, 'training_info.csv')
    training_info_df.to_csv(training_info_file, index=False)
    print(f"Training information saved to {training_info_file}")
except Exception as e:
    print(f"Error saving training information: {e}")

# Remove unnecessary folders
for log_dir in [os.path.join(base_dir, 'lightning_logs'), os.path.join(base_dir, 'catboost_info')]:
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"Deleted {log_dir}")
