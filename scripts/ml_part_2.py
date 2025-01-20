import os
import traceback
import pandas as pd
from darts import TimeSeries
from darts.models import Theta
from prophet import Prophet
import warnings
import logging
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime, timedelta
import numpy as np
import statsforecast

############################################################################################################

# Prerequisites: Initial Setup
print(statsforecast.__version__)

# Suppress warnings and reduce log verbosity
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

# Log error messages
def log_error(message):
    with open(error_log_file, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")

# Log hyperparameter tuning details
def log_hyperparams(message):
    with open(hyperparam_log_file, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")

# Define directories
base_dir = os.path.abspath('.')
data_dir = os.path.join(base_dir, 'data', 'merge')
evaluation_dir = os.path.join(base_dir, 'evaluation')
training_dir = os.path.join(base_dir, 'training')
validation_dir = os.path.join(base_dir, 'validation')

for directory in [evaluation_dir, training_dir, validation_dir]:
    os.makedirs(directory, exist_ok=True)

hyperparam_log_file = os.path.join(training_dir, 'hyperparam_log.txt')
error_log_file = os.path.join(training_dir, 'error_log.txt')

# Validation files list for consolidated metrics
validation_files = [
    os.path.join(validation_dir, 'theta_validation_metrics.csv'),
    os.path.join(validation_dir, 'prophet_validation_metrics.csv'),
    os.path.join(validation_dir, 'gbr_validation_metrics.csv')
]

def append_to_csv(file_path, df, model_name=None):
    """
    Appends data to a CSV file, adding timestamp and model name if not already present,
    and ensures no duplicate rows are added.
    """
    try:
        if "Generated_At" not in df.columns:
            df["Generated_At"] = datetime.now().isoformat()
        if model_name and "Model" not in df.columns:
            df["Model"] = model_name

        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path)
            df = pd.concat([existing_data, df]).drop_duplicates(subset=["Model", "Generated_At"], ignore_index=True)
        df.to_csv(file_path, index=False)

    except Exception as e:
        log_error(f"Error appending to file {file_path}: {e}")

# Validate Dataset with Specific Error Messages
def validate_dataset(df):
    """
    Validates the input dataset for required columns, missing values, and duplicate timestamps.

    Parameters:
        df (pd.DataFrame): Input dataframe.

    Raises:
        ValueError: If validation fails for columns, missing values, or duplicate timestamps.
    """
    if 'y' not in df.columns or 'ds' not in df.columns:
        raise ValueError("Dataset must contain 'ds' and 'y' columns.")
    if df.isnull().any().any():
        missing_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Dataset contains missing values in columns: {missing_cols}")
    if len(df['ds'].unique()) != len(df['ds']):
        raise ValueError("Timestamp column contains duplicate entries.")

# Validate file structure
def validate_file(file_path, required_columns=None):
    if not os.path.exists(file_path):
        if required_columns:
            pd.DataFrame(columns=required_columns).to_csv(file_path, index=False)
        return False
    try:
        df = pd.read_csv(file_path)
        if df.empty or len(df.columns) == 0:
            if required_columns:
                pd.DataFrame(columns=required_columns).to_csv(file_path, index=False)
            return False
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        if required_columns:
            pd.DataFrame(columns=required_columns).to_csv(file_path, index=False)
        return False
    return True

# Append training details to training_info.csv
def append_training_details(model_name, iteration, params, metrics):
    """
    Appends training details (parameters and metrics) to the training_info.csv file.
    """
    training_details = {
        "Model": model_name,
        "Iteration": iteration,
        "Parameters": str(params),  # Convert parameters to string for consistent logging
        "MAE": metrics.get("MAE"),
        "MAPE": metrics.get("MAPE"),
        "RMSE": metrics.get("RMSE"),
        "MSE": metrics.get("MSE"),
        "R²": metrics.get("R²"),
        "MBE": metrics.get("MBE"),
        "Generated_At": datetime.now().isoformat(),
    }
    training_file = os.path.join(training_dir, 'training_info.csv')
    training_df = pd.DataFrame([training_details])
    append_to_csv(training_file, training_df, model_name=model_name)

# Save metrics to validation CSV files
def save_metrics(model_name, metrics, file_path):
    metrics_df = pd.DataFrame([metrics])
    append_to_csv(file_path, metrics_df, model_name=model_name)


############################################################################################################

# Prerequisites: Data Preprocessing

# Dataset loading and validation
input_file = os.path.join(data_dir, 'allyears.csv')

try:
    data = pd.read_csv(input_file).drop_duplicates(subset=['ds'], keep='first')
    data['ds'] = pd.to_datetime(data['ds'])
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data.fillna(data.mean(), inplace=True)
    """    
    # Fill missing numeric values with column means
    numeric_columns = data.select_dtypes(include=[np.number]).fillna(data.mean())

    if not numeric_columns.empty:
        for column in numeric_columns.columns:
            data[column].fillna(data[column].mean(), inplace=True)
    """        
    validate_dataset(data)
except FileNotFoundError:
    log_error(f"File not found: {input_file}")
    raise
except pd.errors.ParserError as e:
    log_error(f"Error parsing file {input_file}: {e}")
    raise

current_date = datetime.now()
history_end = current_date - timedelta(days=1)
past_14_days_start = current_date - timedelta(days=14)

# Historical Data
history_data = data.copy()
train_cutoff = int(0.9 * len(history_data))
train_data = history_data[:train_cutoff]
validation_data = history_data[train_cutoff:]
log_error(f"Validation data shape: {validation_data.shape}")

# Validate splits
if train_data.empty or validation_data.empty:
    raise ValueError("Training or validation data is empty. Check historical data.")

# Past and Future Data
past_14_days_data = data[data['ds'] >= past_14_days_start]
future_data = pd.DataFrame({
    "ds": pd.date_range(start=history_end + timedelta(days=1), periods=14, freq="D"),
    "y": np.nan
})

# Preprocess numeric features
train_features = train_data.drop(columns=['ds', 'y'], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
train_target = train_data['y']
validation_features = validation_data.drop(columns=['ds', 'y'], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
validation_target = validation_data['y']
past_14_features = past_14_days_data.drop(columns=['ds', 'y'], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
future_features = future_data.drop(columns=['ds', 'y'], errors='ignore').select_dtypes(include=[np.number]).fillna(0)

############################################################################################################

# Section 1: Hyperparameter tuning for GradientBoostingRegressor

def tune_gradient_boosting(train_features, train_target, param_grid):
    """
    Tunes the GradientBoostingRegressor using GridSearchCV.

    Parameters:
        train_features (pd.DataFrame): Training features.
        train_target (pd.Series): Training target values.
        param_grid (dict): Hyperparameter grid for tuning.

    Returns:
        GradientBoostingRegressor: Best model from GridSearchCV, or None if tuning fails.
    """
    try:
        if train_features.empty or train_target.empty:
            log_error("Training data is empty. Hyperparameter tuning skipped.")
            return None
        print(f"Tuning GradientBoostingRegressor... Training data: {train_features.shape[0]} rows.")
        gbr = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gbr, param_grid, cv=3, scoring="neg_mean_squared_error")
        grid_search.fit(train_features, train_target)
        log_hyperparams(f"GradientBoostingRegressor Best Params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        log_error(f"Error during GradientBoostingRegressor tuning: {e}")
        return None

# Define hyperparameter grids
initial_params = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.05],
}
expanded_params = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
}
fallback_params = {
    "n_estimators": [10, 20],
    "max_depth": [2, 3],
    "learning_rate": [0.005, 0.01],
}

# Attempt Hyperparameter Tuning with Each Grid
param_grids = [initial_params, expanded_params, fallback_params]
gbm = None
for params in param_grids:
    gbm = tune_gradient_boosting(train_features, train_target, params)
    if gbm is not None:
        break  # Exit loop if a model is successfully tuned

# Log if tuning fails
if gbm is None:
    log_error("GradientBoostingRegressor tuning failed. No model available.")

############################################################################################################

# Section 2: Model Training and Validation

# Validate validation data
if validation_data.empty:
    log_error("Validation data is empty. Check training and history split.")
    raise ValueError("Validation data is empty.")

# Log dataset details
print(f"Dataset loaded with {len(data)} rows.")
print(f"History: {len(history_data)} rows, Future: {len(future_data)} rows.")

# Function to calculate evaluation metrics
def calculate_metrics(model, y_true, y_pred, is_future=False):
    """
    Calculate performance metrics for a given model's predictions.

    Parameters:
        model (str): Name of the model.
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.
        is_future (bool): If true, metrics are placeholders (used for future predictions).

    Returns:
        dict: Metrics including MAE, MAPE, RMSE, MSE, R², and MBE.
    """
    if is_future:
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
        "MBE": np.mean(y_pred - y_true),
    }

# Train and validate models
try:
    # 1. GradientBoostingRegressor Validation
    if gbm:
        print("GradientBoostingRegressor is ready for validation.")
        validation_predicted = gbm.predict(validation_features)
        validation_data['Predicted'] = validation_predicted
        validation_metrics = calculate_metrics("GradientBoostingRegressor", validation_data['y'], validation_predicted)

        # Save metrics to gbr_validation_metrics.csv
        
        append_training_details("GradientBoostingRegressor", 1, gbm.get_params(), validation_metrics)
        append_to_csv(os.path.join(validation_dir, 'gbr_validation_metrics.csv'), pd.DataFrame([validation_metrics]))
        
        gbr_validation_file = os.path.join(validation_dir, 'gbr_validation_metrics.csv')
        save_metrics("GradientBoostingRegressor", validation_metrics, gbr_validation_file)
        print(f"Validation metrics for GradientBoostingRegressor saved to {gbr_validation_file}.")

        if validation_metrics:
            print(f"GradientBoostingRegressor validated successfully. Metrics: {validation_metrics}.")
            log_hyperparams(f"GradientBoostingRegressor validated successfully with metrics: {validation_metrics}.")
            
    # 2. Darts Theta Validation
    validation_series = TimeSeries.from_dataframe(validation_data, time_col="ds", value_cols="y")
    if not train_features.empty:
        try:
            train_series = TimeSeries.from_dataframe(train_data, time_col="ds", value_cols="y")
            model_theta = Theta()
            model_theta.fit(train_series)

            if len(validation_series) > 0:
                validation_prediction = model_theta.predict(len(validation_series))
                validation_data['Theta_Predicted'] = validation_prediction.values().flatten()
                validation_metrics_theta = calculate_metrics(
                    "Darts Theta",
                    validation_series.values().flatten(),
                    validation_prediction.values().flatten()
                )
                theta_params = {"Theta Value": model_theta.theta, "Seasonality Mode": model_theta.season_mode}
                validation_metrics_theta.update({"Parameters": theta_params})
                append_training_details("Darts Theta", 1, theta_params, validation_metrics_theta)
                append_to_csv(os.path.join(validation_dir, 'theta_validation_metrics.csv'), pd.DataFrame([validation_metrics_theta]))
                # Save metrics to theta_validation_metrics.csv
                theta_validation_file = os.path.join(validation_dir, 'theta_validation_metrics.csv')
                save_metrics("Darts Theta", validation_metrics_theta, theta_validation_file)
                print(f"Validation metrics for Darts Theta saved to {theta_validation_file}.")

            else:
                log_error("Validation series is empty for Darts Theta.")
        except Exception as e:
            log_error(f"Error during Theta model validation: {e}\n{traceback.format_exc()}")
    else:
        log_error("No data available for Darts Theta training.")
        print("Training data is empty. Skipping Darts Theta training.")

   
    # 3. Prophet Validation
    if not train_features.empty:
        changepoint_prior_scale = 0.05  # Default value
        try:
            model_prophet = Prophet(
                seasonality_mode="multiplicative",
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=changepoint_prior_scale
            )
            model_prophet.fit(train_data)

            validation_forecast = model_prophet.predict(validation_data[['ds']])
            validation_data['Prophet_Predicted'] = validation_forecast['yhat'].values

            validation_metrics_prophet = calculate_metrics("Prophet", validation_data['y'], validation_forecast['yhat'].values)
            prophet_params = {"Changepoint Prior Scale": changepoint_prior_scale, "Seasonality Mode": "multiplicative"}
            validation_metrics_prophet.update({"Parameters": prophet_params})
            append_training_details("Prophet", 1, prophet_params, validation_metrics_prophet)
            append_to_csv(os.path.join(validation_dir, 'prophet_validation_metrics.csv'), pd.DataFrame([validation_metrics_prophet]))

            # Save metrics to prophet_validation_metrics.csv
            prophet_validation_file = os.path.join(validation_dir, 'prophet_validation_metrics.csv')
            save_metrics("Prophet", validation_metrics_prophet, prophet_validation_file)
            print(f"Validation metrics for Prophet saved to {prophet_validation_file}.")


        except Exception as e:
            log_error(f"Error during Prophet validation: {e}\n{traceback.format_exc()}")
    else:
        log_error("No data available for Prophet training.")
        print("Training data is empty. Skipping Prophet training.")

except Exception as e:
    log_error(f"Error during model training or validation: {e}\n{traceback.format_exc()}")
    raise

############################################################################################################

# Section 3: Past and Future Predictions with Consolidation Metrics 

# 1. Past Predictions for the Last 14 Days
if not past_14_features.empty:
    if gbm is not None:
        past_predictions = gbm.predict(past_14_features)
        past_14_days_data['Predicted'] = past_predictions

        # Calculate metrics for past 14 days
        past_14_metrics = calculate_metrics("GradientBoostingRegressor", past_14_days_data['y'], past_predictions)
        print(f"Past 14-Day Metrics: {past_14_metrics}")
    else:
        log_error("GradientBoostingRegressor model is not initialized. Skipping past predictions.")
else:
    log_error("No valid features for past 14-day predictions.")

# 2. Future Predictions Using All Models
try:
    if future_data.empty:
        log_error("Future data is empty. Skipping future predictions.")
        print("Future data is empty. Predictions skipped.")
    else:
        # 2.a. Darts Theta Future Predictions
        # Check if Model Exists Before Training
        if 'model_theta' not in globals():
            model_theta = Theta()
            train_series = TimeSeries.from_dataframe(train_data, time_col="ds", value_cols="y")
            model_theta.fit(train_series)

        future_series = TimeSeries.from_dataframe(future_data, time_col="ds", value_cols="y")
        theta_predictions = model_theta.predict(len(future_series))
        theta_future_file = os.path.join(evaluation_dir, 'theta_predictions.csv')
        theta_predictions_df = pd.DataFrame({
            "ds": future_data["ds"].values,
            "Predicted": theta_predictions.values().flatten(),
            "Model": "Darts Theta",
            "Generated_At": datetime.now().isoformat()
        })
        append_to_csv(theta_future_file, theta_predictions_df, model_name="Darts Theta")
        print(f"Darts Theta future predictions saved to {theta_future_file}")

        # 2.b. Prophet Future Predictions
        if 'model_prophet' not in globals():
            print("Reinitializing Prophet model for future predictions.")
            model_prophet = Prophet(
                seasonality_mode="multiplicative",
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model_prophet.fit(train_data)

        prophet_forecast = model_prophet.predict(future_data[['ds']])
        prophet_future_file = os.path.join(evaluation_dir, 'prophet_predictions.csv')
        prophet_predictions_df = prophet_forecast[['ds', 'yhat']].rename(columns={"yhat": "Predicted"})
        prophet_predictions_df["Model"] = "Prophet"
        prophet_predictions_df["Generated_At"] = datetime.now().isoformat()
        append_to_csv(prophet_future_file, prophet_predictions_df, model_name="Prophet")
        print(f"Prophet future predictions saved to {prophet_future_file}")

        # 2.c. GradientBoostingRegressor Future Predictions
        if 'gbm' in globals() and gbm is not None:
            try:
                # Generate predictions
                future_predictions = gbm.predict(future_features)
                # Update future_data with predictions
                future_data['Predicted'] = future_predictions
                future_data['Actual'] = None  # Placeholder for actual values
                future_data['Generated_At'] = datetime.now().isoformat()  # Timestamp for predictions
                
                # Save predictions to a CSV file
                gbr_future_file = os.path.join(evaluation_dir, 'gbr_predictions.csv')
                gbr_predictions_df = future_data[['ds', 'Predicted', 'Actual', 'Generated_At']]
                append_to_csv(gbr_future_file, gbr_predictions_df, model_name="GradientBoostingRegressor")

                
                print(f"GradientBoostingRegressor future predictions saved to {gbr_future_file}")
            except Exception as e:
                log_error(f"Error during GradientBoostingRegressor future predictions: {e}")
                print(f"Error: {e}")
        else:
            log_error("GradientBoostingRegressor model is not initialized. Skipping future predictions.")
            print("GradientBoostingRegressor model is not initialized. Skipping future predictions.")

except Exception as e:
    log_error(f"Error generating future predictions: {e}")
    print(f"Error during future predictions: {e}")

# 3. Consolidate Validation Metrics and Generate Summary Report
try:
    consolidated_validation_file = os.path.join(validation_dir, 'consolidated_validation_metrics.csv')
    all_metrics = []
    for file_path in validation_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    all_metrics.append(df)
                else:
                    log_error(f"Validation file {file_path} is empty.")
            except Exception as e:
                log_error(f"Error reading validation file {file_path}: {e}")
        else:
            log_error(f"Validation file {file_path} does not exist.")

    # Consolidation: Skip if No Metrics Available
    if not all_metrics:
        log_error("No metrics available for consolidation or summary generation.")
        print("Skipping consolidation due to missing metrics.")
    else:
        # Perform consolidation and summary report generation
        consolidated_validation_file = os.path.join(validation_dir, 'consolidated_validation_metrics.csv')
        consolidated_validation_df = pd.concat(all_metrics, ignore_index=True)
        consolidated_validation_df = consolidated_validation_df.drop_duplicates(subset=["Model", "Generated_At"], ignore_index=True)
        #consolidated_validation_df.to_csv(consolidated_validation_file, index=False)
        append_to_csv(consolidated_validation_file, consolidated_validation_df)
        print(f"Consolidated validation metrics saved to {consolidated_validation_file}")

    summary_file = os.path.join(evaluation_dir, 'summary_report.csv')
    if all_metrics:
        summary_df = pd.concat(all_metrics, ignore_index=True)
        summary_df = summary_df.drop_duplicates(subset=["Model", "Generated_At"], ignore_index=True)
        append_to_csv(summary_file, summary_df)
        print(f"Summary report saved to {summary_file}")
    else:
        log_error("No metrics found for summary report.")
except Exception as e:
    log_error(f"Error during consolidation or summary generation: {e}")
    print(f"Error during consolidation or summary generation: {e}")

############################################################################################################

# Final Log
print("All sections completed successfully.")
