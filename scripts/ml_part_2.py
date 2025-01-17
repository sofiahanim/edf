import os
import pandas as pd
from darts import TimeSeries
from darts.models import Theta
from prophet import Prophet
import warnings
import logging
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import time
import numpy as np
import pickle
from joblib import dump
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sktime.forecasting.base import ForecastingHorizon
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from cmdstanpy import install_cmdstan

install_cmdstan()


logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# Suppress warnings and reduce log verbosity
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

# Define directories
base_dir = os.path.abspath('.')
data_dir = os.path.join(base_dir, 'data', 'merge')
evaluation_dir = os.path.join(base_dir, 'evaluation')
training_dir = os.path.join(base_dir, 'training')
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(evaluation_dir, exist_ok=True)
os.makedirs(training_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Error logging
error_log_file = os.path.join(training_dir, 'error_log.txt')
def log_error(message):
    with open(error_log_file, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")


def append_to_csv(file_path, df):
    """
    Appends data to a CSV file. If the file is empty or doesn't exist, it creates a new one
    with the provided DataFrame structure.
    """
    try:
        # Check if the file exists
        if os.path.exists(file_path):
            try:
                existing_data = pd.read_csv(file_path)

                # Handle empty or corrupted files
                if existing_data.empty or len(existing_data.columns) == 0:
                    print(f"File {file_path} is empty or corrupted. Creating a new one.")
                    existing_data = pd.DataFrame(columns=df.columns)

            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                print(f"File {file_path} is empty or corrupted. Creating a new one.")
                existing_data = pd.DataFrame(columns=df.columns)

            # Concatenate new data with existing data and deduplicate
            df = pd.concat([existing_data, df]).drop_duplicates(ignore_index=True)

        # Write to the file (overwrite if creating new)
        df.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error appending to file {file_path}: {e}")
        log_error(f"Error appending to file {file_path}: {e}")



def calculate_metrics(model, y_true, y_pred, is_future=False):
    if is_future:  # No actual values for future
        return {
            "Model": model,
            "Generated_At": datetime.now().isoformat(),
            "MAE": None,
            "MAPE": None,
            "RMSE": None,
            "MSE": None,
            "R²": None,
            "MBE": None
        }
    return {
        "Model": model,
        "Generated_At": datetime.now().isoformat(),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MSE": mean_squared_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "MBE": np.mean(y_pred - y_true)
    }

def ensure_columns(df, required_columns):
    """
    Ensures the required columns exist in the DataFrame. If not, adds them with default None or empty values.
    """
    for col in required_columns:
        if col not in df.columns:
            df[col] = None  # Default value if the column is missing
    return df

def validate_file(file_path, required_columns=None):
    """
    Validates a file. If the file is empty or corrupted, it creates a new file
    with the required columns if specified.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Creating a new file.")
        if required_columns:
            pd.DataFrame(columns=required_columns).to_csv(file_path, index=False)
        return False

    try:
        df = pd.read_csv(file_path)
        if df.empty or len(df.columns) == 0:
            print(f"File {file_path} is empty or corrupted. Creating a new file.")
            if required_columns:
                pd.DataFrame(columns=required_columns).to_csv(file_path, index=False)
            return False
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        print(f"File {file_path} is empty or corrupted. Creating a new file.")
        if required_columns:
            pd.DataFrame(columns=required_columns).to_csv(file_path, index=False)
        return False

    return True

def generate_and_save_summary(model_name, model_file, iteration, timestamp):
    """
    Generates and saves a summary report for a given model.
    """
    required_columns = ['Model','Predicted', 'Actual', 'Type', 'Iteration']
    if not validate_file(model_file, required_columns):
        print(f"Skipping summary generation for {model_name} due to invalid file.")
        return

    try:
        data = pd.read_csv(model_file)

        # Ensure required columns exist
        data = ensure_columns(data, required_columns)

        # Aggregate summary metrics
        summary = (
            data.groupby(['Iteration', 'Type'])[['Actual', 'Predicted']]
            .agg(['mean', 'min', 'max'])
            .reset_index()
        )

        # Flatten multi-index columns
        summary.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in summary.columns
        ]

        # Add model name and timestamp
        summary['Model'] = model_name  # This should add the 'Model' column
        summary['Generated_At'] = timestamp

        # Append the summary to a consolidated file
        summary_file = os.path.join(evaluation_dir, 'summary_report.csv')
        append_to_csv(summary_file, summary)

        print(f"Summary for {model_name} appended to {summary_file}")
    except Exception as e:
        print(f"Error generating summary for {model_name}: {e}")
        log_error(f"Error generating summary for {model_name}: {e}")

input_file = os.path.join(data_dir, 'allyears.csv')
data = pd.read_csv(input_file)
data['ds'] = pd.to_datetime(data['ds'])
data = data[['ds', 'y']].drop_duplicates(subset='ds')

# Validations
if data.isnull().any().any():
    raise ValueError("Dataset contains missing values.")

if data.empty:
    raise ValueError("Dataset is empty or incorrectly loaded.")

if data['ds'].isna().any() or data['y'].isna().any():
    raise ValueError("Dataset contains missing or invalid values.")

if len(data['ds'].unique()) != len(data['ds']):
    raise ValueError("Timestamp column contains duplicate entries.")

# Logging dataset info
print(f"Dataset loaded with {len(data)} rows.")

# Data splitting for training, validation, and testing
historical_cutoff = len(data) - 336  # Exclude last 14 days for future testing
train_cutoff = int(0.9 * historical_cutoff)  # 90% of historical data for training

train_data = data[:train_cutoff]
validation_data = data[train_cutoff:historical_cutoff]
historical_test_data = data[historical_cutoff:]

# Define target and predictors
target = 'y'  # Replace 'y' with the actual target column name if different
predictors = [col for col in train_data.columns if col != target]

# Consolidate validation metrics
validation_files = [
    os.path.join(validation_dir, 'theta_validation_metrics.csv'),
    os.path.join(validation_dir, 'prophet_validation_metrics.csv'),
    os.path.join(validation_dir, 'gbr_validation_metrics.csv')
]

all_validation_metrics = []

for file_path in validation_files:
    if validate_file(file_path, required_columns=['Model', 'MAE', 'MAPE', 'RMSE', 'MSE', 'R²', 'MBE', 'Generated_At']):
        df = pd.read_csv(file_path)
        all_validation_metrics.append(df)


if all_validation_metrics:
    consolidated_validation = pd.concat(all_validation_metrics, ignore_index=True)
    consolidated_file = os.path.join(validation_dir, 'consolidated_validation_metrics.csv')
    consolidated_validation.to_csv(consolidated_file, index=False)
    print(f"Consolidated validation metrics saved to {consolidated_file}")

# Darts Theta Validation
try:
    train_series = TimeSeries.from_dataframe(train_data, time_col="ds", value_cols="y")
    validation_series = TimeSeries.from_dataframe(validation_data, time_col="ds", value_cols="y")

    model_theta = Theta()
    model_theta.fit(train_series)

    validation_prediction = model_theta.predict(len(validation_series))
    validation_metrics_theta = calculate_metrics(
        "Darts Theta",
        validation_series.values().flatten(),
        validation_prediction.values().flatten()
    )
    validation_metrics_theta['Model'] = "Darts Theta"
    validation_metrics_theta['Generated_At'] = datetime.now().isoformat()
   
    print("Validation Metrics (Theta):", validation_metrics_theta)
    # Save validation metrics
    theta_validation_file = os.path.join(validation_dir, 'theta_validation_metrics.csv')
    theta_validation_df = pd.DataFrame([validation_metrics_theta])
    append_to_csv(theta_validation_file, theta_validation_df)
    print(f"Theta validation metrics saved to {theta_validation_file}")

except Exception as e:
    print(f"Error during Darts Theta validation: {e}")
    log_error(f"Error during Darts Theta validation: {e}")


# Prophet Validation
try:
    model_prophet = Prophet()  # Instantiate a new object for each iteration
    model_prophet.fit(train_data)

    validation_future = validation_data[['ds']]
    validation_forecast = model_prophet.predict(validation_future)

    validation_actual = validation_data['y'].values
    validation_predicted = validation_forecast['yhat'].values
    validation_metrics_prophet = calculate_metrics("Prophet",validation_actual, validation_predicted)
    print("Validation Metrics (Prophet):", validation_metrics_prophet)

    validation_metrics_prophet['Model'] = "Prophet"
    validation_metrics_prophet['Generated_At'] = datetime.now().isoformat()
    # Save validation metrics
    prophet_validation_file = os.path.join(validation_dir, 'prophet_validation_metrics.csv')
    prophet_validation_df = pd.DataFrame([validation_metrics_prophet])
    append_to_csv(prophet_validation_file, prophet_validation_df)
    print(f"Prophet validation metrics saved to {prophet_validation_file}")

except Exception as e:
    print(f"Error during Prophet validation: {e}")
    log_error(f"Error during Prophet validation: {e}")

# Sktime ThetaForecaster Validation
try:  
    # Split the data for training and validation
    train_series, validation_series = temporal_train_test_split(data['y'], test_size=336)  # Last 14 days for validation
    fh = ForecastingHorizon(validation_series.index, is_relative=False)

    # Initialize and train ThetaForecaster
    theta_forecaster = ThetaForecaster(sp=24)  # Assuming daily seasonality (hourly data with 24-hour cycle)
    theta_forecaster.fit(train_series)

    # Predict on the validation set
    validation_predicted = theta_forecaster.predict(fh)

    # Calculate Validation Metrics
    validation_metrics_sktime = {
        "Model": "ThetaForecaster",
        "Generated_At": datetime.now().isoformat(),
        "MAE": mean_absolute_error(validation_series, validation_predicted),
        "MAPE": mean_absolute_percentage_error(validation_series, validation_predicted),
        "RMSE": np.sqrt(mean_squared_error(validation_series, validation_predicted)),
        "MSE": mean_squared_error(validation_series, validation_predicted),
    }

    print("Validation Metrics (ThetaForecaster):", validation_metrics_sktime)

    # Save Validation Metrics for ThetaForecaster
    sktime_validation_file = os.path.join(validation_dir, 'sktime_theta_validation_metrics.csv')
    sktime_validation_df = pd.DataFrame([validation_metrics_sktime])
    append_to_csv(sktime_validation_file, sktime_validation_df)
    print(f"ThetaForecaster validation metrics saved to {sktime_validation_file}")

except Exception as e:
    print(f"Error during ThetaForecaster validation: {e}")
    log_error(f"Error during ThetaForecaster validation: {e}")

if data['ds'].isna().any() or data['y'].isna().any():
    raise ValueError("Dataset contains missing or invalid values.")

if len(data['ds'].unique()) != len(data['ds']):
    raise ValueError("Timestamp column contains duplicate entries.")

def save_metrics(model_name, metrics):
    metrics_file = os.path.join(training_dir, f"{model_name.lower()}_metrics.csv")
    metrics_df = pd.DataFrame([metrics])
    append_to_csv(metrics_file, metrics_df)

# Training loop
from tqdm import tqdm

training_info = []

for i in tqdm(range(10), desc="Processing Models"):
    time.sleep(1)  # Simulate processing


for iteration in tqdm(range(1, 4), desc="Training Iterations"):
    print(f"Starting training iteration {iteration}...")
    #predicted_at = datetime.now()
    predicted_at = datetime.now().isoformat()

    for model_name, prediction_file in {
        "Darts Theta": os.path.join(evaluation_dir, 'darts_theta_predictions.csv'),
        "H2O": os.path.join(evaluation_dir, 'gbr_predictions.csv'),
        "Prophet": os.path.join(evaluation_dir, 'prophet_predictions.csv')
    }.items():

        if model_name == "H2O":
            try:
                print("\nTraining GradientBoostingRegressor Model...")
                start_time = time.time()

                # Validate dataset
                if data is None or data.empty:
                    raise ValueError("Dataset is empty or invalid.")
                if target not in data.columns:
                    raise ValueError(f"Target column '{target}' is missing in the dataset.")
                if not all(col in data.columns for col in predictors):
                    raise ValueError("One or more predictor columns are missing in the dataset.")

                # Split the data into training and testing datasets
                X = data[predictors]
                y = data[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

                # Initialize and train the GradientBoostingRegressor model
                gbm = GradientBoostingRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
                gbm.fit(X_train, y_train)

                # Historical predictions and metrics
                train_predicted = gbm.predict(X_train)
                historical_metrics = calculate_metrics(
                    "GradientBoostingRegressor",
                    y_train.values,
                    train_predicted
                )

                # Prepare historical DataFrame
                historical_df = pd.DataFrame({
                    "Actual": y_train.values,
                    "Predicted": train_predicted,
                    "Type": "Historical"
                })

                # Future predictions
                test_predicted = gbm.predict(X_test)
                future_df = pd.DataFrame({
                    "Actual": y_test.values,
                    "Predicted": test_predicted,
                    "Type": "Future"
                })

                # Combine historical and future results
                combined_df = pd.concat([historical_df, future_df], ignore_index=True)
                combined_df['Iteration'] = iteration

                # Debugging logs to verify data correctness
                print("Train actual sample:", y_train.head())
                print("Train predicted sample:", train_predicted[:5])
                print("Combined DataFrame sample:", combined_df.head())

                end_time = time.time()
                metrics = {
                    "Model": "GradientBoostingRegressor",
                    "Iteration": iteration,
                    "MAE": historical_metrics['MAE'],
                    "MAPE": historical_metrics['MAPE'],
                    "RMSE": historical_metrics['RMSE'],
                    "MSE": historical_metrics['MSE'],
                    "Time_Taken": end_time - start_time,  # Ensure consistency
                    "Training_Rows": len(X_train),
                    "Test_Rows": len(X_test),
                    "Predicted_At": predicted_at
                }

                save_metrics("GradientBoostingRegressor", metrics)
                training_info.append(metrics)

                # Aggregate results
                aggregated_df = combined_df.groupby('Type')[['Actual', 'Predicted']].agg(['mean', 'min', 'max']).reset_index()

                # Flatten multi-level column names
                aggregated_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in aggregated_df.columns]

                # Add iteration and timestamp
                aggregated_df['Model'] = "GradientBoostingRegressor"
                aggregated_df['Iteration'] = iteration
                aggregated_df['Generated_At'] = predicted_at

                # Save aggregated results
                append_to_csv(prediction_file, aggregated_df)

                print("GradientBoostingRegressor Model completed successfully.")
                # Call generate_and_save_summary for each model after saving predictions

                generate_and_save_summary("GradientBoostingRegressor", prediction_file, iteration, predicted_at)

            except Exception as e:
                print(f"GradientBoostingRegressor failed: {e}")
                log_error(f"GradientBoostingRegressor Model failed: {e}")


        elif model_name == "Darts Theta":
            # Theta Model
            # Darts Theta Model
            try:
                print("Running Darts Theta Model...")
                start_time = time.time()

                # Prepare TimeSeries object
                series = TimeSeries.from_dataframe(data, time_col="ds", value_cols="y", freq="H")

                if series is None or len(series) == 0:
                    raise ValueError("Darts TimeSeries is invalid or empty. Check the input data.")

                np.random.seed(42)
                # Train the model
                model_theta = Theta()
                model_theta.fit(series[:-336])  # Exclude the last 336 hours (14 days)

                # Forecast for the next 336 hours
                # forecast_theta = model_theta.predict(336)
                # Darts Theta prediction
                forecast_theta = model_theta.predict(336)
                if forecast_theta is None:
                    raise ValueError("Darts Theta prediction returned None. Check the input series or model.")


                # Historical metrics calculation
                historical_actual = series[-336:]
                historical_predicted = forecast_theta
                historical_metrics = calculate_metrics( "Darts Theta",historical_actual.values().flatten(), historical_predicted.values().flatten())

                # Combine historical and future data
                historical_df = pd.DataFrame({
                    "Actual": historical_actual.values().flatten(),
                    "Predicted": historical_predicted.values().flatten(),
                    "Type": "Historical"
                })

                future_df = pd.DataFrame({
                    "Actual": [None] * 336,  # Future actual values are not available
                    "Predicted": forecast_theta.values().flatten(),
                    "Type": "Future"
                })

                combined_df = pd.concat([historical_df, future_df], ignore_index=True)
                combined_df['Iteration'] = iteration
                
                end_time = time.time()

                # Save metrics and predictions
                metrics = {
                    "Model": "Darts Theta",
                    "Iteration": iteration,
                    "MAE": historical_metrics['MAE'],
                    "MAPE": historical_metrics['MAPE'],
                    "RMSE": historical_metrics['RMSE'],
                    "MSE": historical_metrics['MSE'],
                    "Time_Taken": end_time - start_time,
                    "Predicted_At": predicted_at
                }
                save_metrics("Darts Theta", metrics)
                training_info.append(metrics)
                        
                # Aggregate results
                aggregated_df = combined_df.groupby('Type')[['Actual', 'Predicted']].agg(['mean', 'min', 'max']).reset_index()

                # Flatten multi-level column names
                aggregated_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in aggregated_df.columns]

                # Add iteration and timestamp
                aggregated_df['Model'] = "Darts Theta"
                aggregated_df['Iteration'] = iteration
                aggregated_df['Generated_At'] = predicted_at

                # Save aggregated results
                append_to_csv(prediction_file, aggregated_df)

                # Save aggregated results
                append_to_csv(prediction_file, aggregated_df)
                # Call generate_and_save_summary for each model after saving predictions
                generate_and_save_summary("Darts Theta", prediction_file, iteration, predicted_at)                
            except Exception as e:
                print(f"Darts Theta Model failed: {e}")
                log_error(f"Darts Theta Model failed: {e}")

        elif model_name == "Prophet":
            # Prophet Model
            try:
                print("Running Prophet Model...")

                start_time = time.time()
                # Fit model on historical data
                model_prophet = Prophet()
                model_prophet.fit(data[:-336])  # Exclude the last 336 rows for historical training

                # Forecast for the next 336 hours (future)
                #future = model_prophet.make_future_dataframe(periods=336, freq='H')
                #forecast = model_prophet.predict(future)

                # Generate future DataFrame
                future = model_prophet.make_future_dataframe(periods=336, freq='H')
                if future is None or future.empty:
                    raise ValueError("Prophet future DataFrame is empty or None. Check the input data.")

                # Prophet prediction
                forecast = model_prophet.predict(future)
                if forecast is None or forecast.empty:
                    raise ValueError("Prophet prediction returned None or an empty DataFrame. Check the model and input data.")

                # Historical metrics computation
                historical_actual = data[-336:]['y']
                historical_predicted = forecast['yhat'][:len(historical_actual)]
            

                # Align lengths of actual and predicted data
                min_length = min(len(historical_actual), len(historical_predicted))
                historical_actual = historical_actual.iloc[:min_length]
                historical_predicted = historical_predicted.iloc[:min_length]

                historical_metrics = calculate_metrics("Prophet",historical_actual, historical_predicted)

                # Combine data for historical and future predictions
                historical_df = pd.DataFrame({
                    "Actual": historical_actual.values,
                    "Predicted": historical_predicted.values,
                    "Type": "Historical"
                })

                # Future predictions (ensure proper handling if future length mismatches)
                future_predictions = forecast['yhat'][-336:]
                if len(future_predictions) < 336:
                    future_predictions = pd.Series([None] * 336, name='Predicted')

                future_df = pd.DataFrame({
                    "Actual": [None] * len(future_predictions),  # No actual values available for future
                    "Predicted": future_predictions.values,
                    "Type": "Future"
                })

                combined_df = pd.concat([historical_df, future_df], ignore_index=True)
                combined_df['Iteration'] = iteration
                end_time = time.time()

                # Save metrics and predictions
                metrics = {
                    "Model": "Prophet",
                    "Iteration": iteration,
                    "MAE": historical_metrics['MAE'],
                    "MAPE": historical_metrics['MAPE'],
                    "RMSE": historical_metrics['RMSE'],
                    "MSE": historical_metrics['MSE'],
                    "R²": historical_metrics['R²'],
                    "MBE": historical_metrics['MBE'],
                    "Time_Taken": end_time - start_time,
                    "Training_Rows": len(data[:-336]),
                    "Test_Rows": 336,
                    "Predicted_At": predicted_at
                }
                save_metrics("Prophet", metrics)
                training_info.append(metrics)

                # Aggregate results
                aggregated_df = combined_df.groupby('Type')[['Actual', 'Predicted']].agg(['mean', 'min', 'max']).reset_index()

                # Flatten multi-level column names
                aggregated_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in aggregated_df.columns]

                # Add iteration and timestamp
                aggregated_df['Model'] = "Prophet"
                aggregated_df['Iteration'] = iteration
                aggregated_df['Generated_At'] = predicted_at

                # Save aggregated results
                append_to_csv(prediction_file, aggregated_df)
                # Call generate_and_save_summary for each model after saving predictions
                generate_and_save_summary("Prophet", prediction_file, iteration, predicted_at)
                print(f"Prophet model saved for iteration {iteration}")

            except Exception as e:
                print(f"Prophet Model failed: {e}")
                log_error(f"Prophet Model failed: {e}")
        
# Save training info
try:
    training_info_df = pd.DataFrame(training_info)
    if not training_info_df.empty:  # Ensure the DataFrame has content
        training_info_file = os.path.join(training_dir, 'training_info.csv')
        append_to_csv(training_info_file, training_info_df)
        print(f"Training info saved to {training_info_file}")
    else:
        print("No training information to save. Skipping.")
except Exception as e:
    print(f"Error saving training info: {e}")
    log_error(f"Error saving training info: {e}")


# Final Logging and Cleanup
print("All forecasts and training iterations completed.")

# Generate a summary report
metrics_files = [
    os.path.join(training_dir, 'gbr_metrics.csv'),
    os.path.join(training_dir, 'darts_theta_metrics.csv'),
    os.path.join(training_dir, 'prophet_metrics.csv')
]


# Generate a summary report
summary_file = os.path.join(evaluation_dir, 'summary_report.csv')
all_metrics = []
for file_path in [
    os.path.join(evaluation_dir, 'gbr_predictions.csv'),
    os.path.join(evaluation_dir, 'darts_theta_predictions.csv'),
    os.path.join(evaluation_dir, 'prophet_predictions.csv')
]:
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                all_metrics.append(df)
        except pd.errors.EmptyDataError:
            print(f"File {file_path} is empty. Skipping...")

if all_metrics:
    summary_df = pd.concat(all_metrics, ignore_index=True)
    summary_df.to_csv(summary_file, index=False)
    print(f"Consolidated summary report saved to {summary_file}")
else:
    print("No metrics files found or all files are empty.")

print("Execution completed.")
