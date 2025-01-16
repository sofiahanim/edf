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
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator

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
os.makedirs(evaluation_dir, exist_ok=True)
os.makedirs(training_dir, exist_ok=True)

# Error logging
error_log_file = os.path.join(training_dir, 'error_log.txt')
def log_error(message):
    with open(error_log_file, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")

def append_to_csv(file_path, df):
    if os.path.exists(file_path):
        try:
            existing_data = pd.read_csv(file_path)
            if existing_data.empty:
                existing_data = pd.DataFrame(columns=df.columns)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            print(f"File {file_path} is corrupted or empty. Creating a new one.")
            existing_data = pd.DataFrame(columns=df.columns)
        
        df = pd.concat([existing_data, df]).drop_duplicates(ignore_index=True)
    df.to_csv(file_path, index=False)


def calculate_metrics(y_true, y_pred, is_future=False):
    if is_future:  # No actual values for future
        return {
            "MAE": None,
            "MAPE": None,
            "RMSE": None,
            "MSE": None,
            "R²": None,
            "MBE": None
        }
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MSE": mean_squared_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "MBE": np.mean(y_pred - y_true)
    }

def generate_and_save_summary(model_name, model_file, iteration, timestamp):
    try:
        if not os.path.exists(model_file):
            print(f"No predictions found for {model_name}. Skipping summary generation.")
            return

        # Load data
        data = pd.read_csv(model_file)

        # Ensure required columns exist
        if not {'Actual', 'Predicted', 'Type', 'Iteration'}.issubset(data.columns):
            print(f"Skipping {model_name} - Required columns missing.")
            return

        # Calculate summary metrics
        summary = (
            data.groupby(['Iteration', 'Type'])[['Actual', 'Predicted']]
            .agg(['mean', 'min', 'max'])
            .reset_index()
        )

        # Flatten multi-index columns
        summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns]

        # Add timestamp
        summary['Generated_At'] = timestamp

        # Append or create a summary file
        summary_file = os.path.join(evaluation_dir, f"{model_name.lower()}_summary.csv")
        append_to_csv(summary_file, summary)
        print(f"Summary for {model_name} appended to {summary_file}")
    except Exception as e:
        print(f"Error generating summary for {model_name}: {e}")

# Load dataset
input_file = os.path.join(data_dir, 'allyears.csv')
data = pd.read_csv(input_file)


# Data preparation
data['ds'] = pd.to_datetime(data['ds'])
data = data.rename(columns={'y': 'y'})
data = data[['ds', 'y']]
data = data.drop_duplicates(subset='ds')

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

# Logging dataset info
print(f"Dataset loaded with {len(data)} rows.")
# Initialize H2O
h2o.init()

try:
    h2o_data = h2o.import_file(input_file)
    # After importing data
    print(h2o_data)
except Exception as e:
    print(f"Error loading dataset: {e}")



# Define predictors and target
target = 'y'
predictors = [col for col in h2o_data.columns if col not in [target, 'date']]


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

for iteration in tqdm(range(1, 4), desc="Training Iterations"):
    print(f"Starting training iteration {iteration}...")
    #predicted_at = datetime.now()
    predicted_at = datetime.now().isoformat()

    for model_name, prediction_file in {
        "Darts Theta": os.path.join(evaluation_dir, 'darts_theta_predictions.csv'),
        "H2O": os.path.join(evaluation_dir, 'h2o_predictions.csv'),
        "Prophet": os.path.join(evaluation_dir, 'prophet_predictions.csv')
    }.items():
        if model_name == "H2O":
            # H2O GBM Model
            try:
                print("\nTraining H2O GBM Model...")
                start_time = time.time()

                # Validate dataset
                if h2o_data is None or h2o_data.nrows == 0:
                    raise ValueError("H2O dataset is empty or invalid.")
                if target not in h2o_data.columns:
                    raise ValueError(f"Target column '{target}' is missing in H2O data.")
                if not all(col in h2o_data.columns for col in predictors):
                    raise ValueError("One or more predictor columns are missing in H2O data.")

                # Split the data into training and testing datasets
                train, test = h2o_data.split_frame(ratios=[0.8], seed=1234)

                # Initialize and train the H2O GBM model
                h2o_gbm = H2OGradientBoostingEstimator(ntrees=50, max_depth=5, learn_rate=0.1)
                h2o_gbm.train(x=predictors, y=target, training_frame=train)

                # Historical predictions and metrics
                train_actual = train[target].as_data_frame()
                train_predicted = h2o_gbm.predict(train).as_data_frame()
                historical_metrics = calculate_metrics(
                    train_actual.values.flatten(),
                    train_predicted.values.flatten()
                )

                # Prepare historical DataFrame
                historical_df = pd.DataFrame({
                    "Actual": train_actual.values.flatten(),
                    "Predicted": train_predicted.values.flatten(),
                    "Type": "Historical"
                })

                # Future predictions
                test_predicted = h2o_gbm.predict(test).as_data_frame()
                future_df = pd.DataFrame({
                    "Actual": [None] * len(test_predicted),  # No actual values for future
                    "Predicted": test_predicted.values.flatten(),
                    "Type": "Future"
                })

                # Combine historical and future results
                combined_df = pd.concat([historical_df, future_df], ignore_index=True)
                combined_df['Iteration'] = iteration

                # Debugging logs to verify data correctness
                print("Train actual sample:", train_actual.head())
                print("Train predicted sample:", train_predicted.head())
                print("Combined DataFrame sample:", combined_df.head())
                
                end_time = time.time()
                metrics = {
                    "Model": "H2O GBM",
                    "Iteration": iteration,
                    "MAE": historical_metrics['MAE'],
                    "MAPE": historical_metrics['MAPE'],
                    "RMSE": historical_metrics['RMSE'],
                    "MSE": historical_metrics['MSE'],
                    "Time_Taken": end_time - start_time,  # Ensure consistency
                    "Training_Rows": train.nrows,
                    "Test_Rows": test.nrows,
                    "Predicted_At": predicted_at
                }

                save_metrics("H2O", metrics)
                training_info.append(metrics)
                
                # Aggregate results
                aggregated_df = combined_df.groupby('Type')[['Actual', 'Predicted']].agg(['mean', 'min', 'max']).reset_index()

                # Flatten multi-level column names
                aggregated_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in aggregated_df.columns]

                # Add iteration and timestamp
                aggregated_df['Iteration'] = iteration
                aggregated_df['Generated_At'] = predicted_at

                # Save aggregated results
                append_to_csv(prediction_file, aggregated_df)
                
                print("H2O GBM Model completed successfully.")

            except Exception as e:
                print(f"H2O GBM failed: {e}")
                log_error(f"H2O GBM Model failed: {e}")


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
                historical_metrics = calculate_metrics(historical_actual.values().flatten(), historical_predicted.values().flatten())

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
                aggregated_df['Iteration'] = iteration
                aggregated_df['Generated_At'] = predicted_at

                # Save aggregated results
                append_to_csv(prediction_file, aggregated_df)

                # Save aggregated results
                append_to_csv(prediction_file, aggregated_df)

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

                historical_metrics = calculate_metrics(historical_actual, historical_predicted)

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
                aggregated_df['Iteration'] = iteration
                aggregated_df['Generated_At'] = predicted_at

                # Save aggregated results
                append_to_csv(prediction_file, aggregated_df)

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

try:
    print("Cleaning up H2O resources...")
    h2o.remove_all()
    h2o.cluster().shutdown(prompt=False)
    print("H2O cluster shutdown completed.")
except Exception as e:
    print(f"Error during H2O cleanup: {e}")

# Generate a summary report
metrics_files = [
    os.path.join(training_dir, 'h2o_metrics.csv'),
    os.path.join(training_dir, 'darts_theta_metrics.csv'),
    os.path.join(training_dir, 'prophet_metrics.csv')
]


# Generate a summary report
summary_file = os.path.join(evaluation_dir, 'summary_report.csv')
all_metrics = []
for file_path in [
    os.path.join(evaluation_dir, 'h2o_predictions.csv'),
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
