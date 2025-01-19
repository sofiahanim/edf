import os
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

hyperparam_log_file = os.path.join(training_dir, 'hyperparam_log.txt')
error_log_file = os.path.join(training_dir, 'error_log.txt')

# Validation files list for consolidated metrics
validation_files = [
    os.path.join(validation_dir, 'theta_validation_metrics.csv'),
    os.path.join(validation_dir, 'prophet_validation_metrics.csv'),
    os.path.join(validation_dir, 'gbr_validation_metrics.csv')
]


# Log error messages
def log_error(message):
    with open(error_log_file, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")

# Log hyperparameter tuning details
def log_hyperparams(message):
    with open(hyperparam_log_file, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")


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
            try:
                existing_data = pd.read_csv(file_path)
                if existing_data.empty or len(existing_data.columns) == 0:
                    existing_data = pd.DataFrame(columns=df.columns)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                existing_data = pd.DataFrame(columns=df.columns)
            df = pd.concat([existing_data, df]).drop_duplicates(subset=["Model", "Generated_At"], ignore_index=True)
        df.to_csv(file_path, index=False)
    except Exception as e:
        log_error(f"Error appending to file {file_path}: {e}")

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

# Calculate evaluation metrics
def calculate_metrics(model, y_true, y_pred, is_future=False):
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
    result = {
        "Model": model,
        "Generated_At": datetime.now().isoformat(),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MSE": mean_squared_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "MBE": np.mean(y_pred - y_true),
    }

    if model == "GradientBoostingRegressor":
        result["Parameters"] = str(best_gbr_params)
    elif model == "Darts Theta":
        result["Parameters"] = str({"Theta Value": model_theta.theta, "Seasonality Mode": model_theta.season_mode})
    elif model == "Prophet":
        result["Parameters"] = str({"Changepoint Prior Scale": 0.05, "Seasonality Mode": "multiplicative"})

    return result


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

# Load dataset
input_file = os.path.join(data_dir, 'allyears.csv')
data = pd.read_csv(input_file)
data = data.drop_duplicates(subset=['ds'], keep='first')

# Ensure 'ds' column is datetime and target 'y' is numeric
data['ds'] = pd.to_datetime(data['ds'])
if 'y' in data.columns:
    data['y'] = pd.to_numeric(data['y'], errors='coerce')

# Fill missing numeric values with column means
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Convert and validate dataset
data['ds'] = pd.to_datetime(data['ds'])
if 'y' not in data.columns or 'ds' not in data.columns:
    raise ValueError("Dataset must contain 'ds' and 'y' columns.")
if data.isnull().any().any():
    raise ValueError("Dataset contains missing values.")
if len(data['ds'].unique()) != len(data['ds']):
    raise ValueError("Timestamp column contains duplicate entries.")

current_date = datetime.now()
history_end = current_date - timedelta(days=1)
history_start = history_end - timedelta(days=30)
future_start = current_date
future_end = current_date + timedelta(days=14)

# All rows are historical
history_data = data.copy()

# Generate placeholder rows for future predictions
future_data = pd.DataFrame({
    "ds": pd.date_range(start=history_data['ds'].max() + timedelta(days=1), end=future_end, freq="D"),
    "y": np.nan  # Use NaN for missing target values
})

# Preprocess numeric features in future data
future_features = future_data.drop(columns=['ds', 'y'], errors='ignore').select_dtypes(include=[np.number])
future_features = future_features.fillna(future_features.mean())

log_error("All rows in the dataset are historical. Placeholder rows created for future predictions.")
print(f"Info: Placeholder rows created for {len(future_data)} future rows.")


# Split historical data into training and validation
train_cutoff = int(0.9 * len(history_data))
train_data = history_data[:train_cutoff]
validation_data = history_data[train_cutoff:]

# Prepare training features and target
train_features = train_data.drop(columns=['ds', 'y'], errors='ignore').select_dtypes(include=[np.number])
train_features = train_features.fillna(train_features.mean())
train_target = train_data['y']

# Prepare validation features and target
validation_features = validation_data.drop(columns=['ds', 'y'], errors='ignore').select_dtypes(include=[np.number])
validation_features = validation_features.fillna(validation_features.mean())
validation_target = validation_data['y']

# Hyperparameter tuning for GradientBoostingRegressor
try:
    gbr_params = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
    }
    gbr = GradientBoostingRegressor(random_state=42)
    
    if train_features.empty or train_target.empty:
        log_error("Training data is empty. GradientBoostingRegressor tuning skipped.")
    else:
        print(f"Training data shape: {train_features.shape}, Target shape: {train_target.shape}")
        grid_search = GridSearchCV(gbr, gbr_params, cv=3, scoring="neg_mean_squared_error")
        try:
            grid_search.fit(train_features, train_target)
            best_gbr_params = grid_search.best_params_
            log_hyperparams(f"GradientBoostingRegressor Best Params: {best_gbr_params}")
        except Exception as e:
            log_error(f"Error during GradientBoostingRegressor tuning: {e}")

    best_gbr_score = -grid_search.best_score_
    log_hyperparams(f"GradientBoostingRegressor Best Params: {best_gbr_params}, Best Score: {best_gbr_score}")
    print(f"GradientBoostingRegressor Best Params: {best_gbr_params}, Best Score: {best_gbr_score}")

    gbm = grid_search.best_estimator_
except Exception as e:
    log_error(f"Error during GradientBoostingRegressor hyperparameter tuning: {e}")

if future_features.empty:
    log_error("No valid features for GradientBoostingRegressor predictions.")
    print("Future features are empty, skipping GradientBoostingRegressor predictions.")
else:
    gbr_predictions = gbm.predict(future_features)
    gbr_predictions_df = pd.DataFrame({
        "ds": future_data["ds"],
        "Predicted": gbr_predictions,
        "Model": "GradientBoostingRegressor",
        "Generated_At": datetime.now().isoformat()
    })
    append_to_csv(os.path.join(evaluation_dir, 'gbr_predictions.csv'), gbr_predictions_df, model_name="GradientBoostingRegressor")
    print(f"GradientBoostingRegressor future predictions saved to {os.path.join(evaluation_dir, 'gbr_predictions.csv')}")

# Validate validation data
if validation_data.empty:
    log_error("Validation data is empty. Check training and history split.")
    raise ValueError("Validation data is empty.")

# Log dataset details
print(f"Dataset loaded with {len(data)} rows.")
print(f"History: {len(history_data)} rows, Future: {len(future_data)} rows.")


# Train and validate models
try:
    # GradientBoostingRegressor
    #validation_predicted = gbm.predict(validation_data.drop(columns=['ds', 'y']))
    validation_predicted = gbm.predict(validation_features)

    validation_data['Predicted'] = validation_predicted
    validation_metrics = calculate_metrics("GradientBoostingRegressor", validation_data['y'], validation_predicted)
    append_training_details("GradientBoostingRegressor", 1, best_gbr_params, validation_metrics)
    append_to_csv(os.path.join(validation_dir, 'gbr_validation_metrics.csv'), pd.DataFrame([validation_metrics]))

    # Darts Theta
    validation_series = TimeSeries.from_dataframe(validation_data, time_col="ds", value_cols="y")
    if train_features.empty:
        log_error("No data available for Darts Theta training.")
        print("Training data is empty, skipping Darts Theta training.")
    else:
        try:
            train_series = TimeSeries.from_dataframe(train_data, time_col="ds", value_cols="y")
            validation_series = TimeSeries.from_dataframe(validation_data, time_col="ds", value_cols="y")
            if len(train_series) == 0 or len(validation_series) == 0:
                raise ValueError("Train or Validation series is empty.")
            model_theta = Theta()
            model_theta.fit(train_series)
            validation_prediction = model_theta.predict(len(validation_series))
        except Exception as e:
            log_error(f"Error during Theta model training or validation: {e}")


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

    # Prophet
    if train_features.empty:
        log_error("No data available for Prophet training.")
        print("Training data is empty, skipping Prophet training.")
    else:
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
            log_hyperparams(f"Prophet Model: Changepoint Prior Scale: {changepoint_prior_scale}")
        except Exception as e:
            log_error(f"Error during Prophet training: {e}")


    validation_forecast = model_prophet.predict(validation_data[['ds']])
    validation_data['Prophet_Predicted'] = validation_forecast['yhat'].values
    validation_metrics_prophet = calculate_metrics("Prophet", validation_data['y'], validation_forecast['yhat'].values)
    prophet_params = {"Changepoint Prior Scale": 0.05, "Seasonality Mode": "multiplicative"}
    validation_metrics_prophet.update({"Parameters": prophet_params})
    append_training_details("Prophet", 1, prophet_params, validation_metrics_prophet)
    append_to_csv(os.path.join(validation_dir, 'prophet_validation_metrics.csv'), pd.DataFrame([validation_metrics_prophet]))

except Exception as e:
    log_error(f"Error during model training or validation: {e}")

# Predict for future data using all models
try:
    # Ensure model_theta is defined
    if 'model_theta' not in globals():
        print("Reinitializing and training Theta model for future predictions.")
        model_theta = Theta()
        train_series = TimeSeries.from_dataframe(train_data, time_col="ds", value_cols="y")
        model_theta.fit(train_series)

    # Darts Theta Future Predictions
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
    print(f"Train Series Length: {len(train_series)}, Future Series Length: {len(future_series)}")

    # Ensure model_prophet is defined
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

    # Prophet Future Predictions
    prophet_forecast = model_prophet.predict(future_data[['ds']])
    prophet_future_file = os.path.join(evaluation_dir, 'prophet_predictions.csv')
    prophet_predictions_df = prophet_forecast[['ds', 'yhat']].rename(columns={"yhat": "Predicted"})
    prophet_predictions_df["Model"] = "Prophet"
    prophet_predictions_df["Generated_At"] = datetime.now().isoformat()
    append_to_csv(prophet_future_file, prophet_predictions_df, model_name="Prophet")
    print(f"Prophet future predictions saved to {prophet_future_file}")

    # Ensure gbm (GradientBoostingRegressor) is defined
    if 'gbm' not in globals():
        print("Reinitializing and training GradientBoostingRegressor for future predictions.")
        gbr = GradientBoostingRegressor(random_state=42)
        gbr_params = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
        }
        if train_features.empty or train_target.empty:
            log_error("Training data is empty. GradientBoostingRegressor tuning skipped.")
            print("Training data is empty, skipping GradientBoostingRegressor tuning.")
        else:
            grid_search = GridSearchCV(gbr, gbr_params, cv=3, scoring="neg_mean_squared_error")
            grid_search.fit(train_features, train_target)


        gbm = grid_search.best_estimator_

    # GradientBoostingRegressor Future Predictions
    if future_features.empty:
        log_error("No valid features for GradientBoostingRegressor predictions.")
        print("Future features are empty, skipping GradientBoostingRegressor predictions.")
    else:
        gbr_predictions = gbm.predict(future_features)
        gbr_future_file = os.path.join(evaluation_dir, 'gbr_predictions.csv')
        gbr_predictions_df = pd.DataFrame({
            "ds": future_data["ds"],
            "Predicted": gbr_predictions,
            "Model": "GradientBoostingRegressor",
            "Generated_At": datetime.now().isoformat()
        })
        append_to_csv(gbr_future_file, gbr_predictions_df, model_name="GradientBoostingRegressor")
        print(f"GradientBoostingRegressor future predictions saved to {gbr_future_file}")

except Exception as e:
    log_error(f"Error generating future predictions: {e}")
    print(f"Error during future predictions: {e}")

# Consolidated Validation Metrics
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

    if all_metrics:
        try:
            consolidated_validation_df = pd.concat(all_metrics, ignore_index=True)
            consolidated_validation_df = consolidated_validation_df.drop_duplicates(subset=["Model", "Generated_At"], ignore_index=True)
            consolidated_validation_df.to_csv(consolidated_validation_file, index=False)
            print(f"Consolidated validation metrics saved to {consolidated_validation_file}")
        except Exception as e:
            log_error(f"Error consolidating validation metrics: {e}")
            print(f"Error consolidating validation metrics: {e}")
    else:
        print("No validation metrics available to consolidate.")
        log_error("No validation metrics available for consolidation.")

except Exception as e:
    log_error(f"Unexpected error in consolidating validation metrics: {e}")
    print(f"Unexpected error: {e}")


# Generate Summary Report
try:
    summary_file = os.path.join(evaluation_dir, 'summary_report.csv')
    if all_metrics:
        print(f"Consolidating {len(all_metrics)} metrics into summary.")
        try:
            summary_df = pd.concat(all_metrics, ignore_index=True)
            summary_df = summary_df.drop_duplicates(subset=["Model", "Generated_At"], ignore_index=True)
            summary_df.to_csv(summary_file, index=False)
            print(f"Summary report saved to {summary_file}")
        except Exception as e:
            log_error(f"Error generating summary report: {e}")
    else:
        print("No metrics found to generate a summary report.")
        log_error("No metrics found for summary report.")
        
except Exception as e:
    log_error(f"Error generating summary report: {e}")
    print(f"Error: {e}")

# Final Log
print("All processes completed successfully.")
