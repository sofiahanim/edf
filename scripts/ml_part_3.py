import os
import pandas as pd
import logging
import requests
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===== Logging Setup ===== #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Define Directories ===== #
base_dir = os.path.abspath('.')
evaluation_dir = os.path.join(base_dir, 'evaluation')
reports_dir = os.path.join(base_dir, 'reports')

for dir_path in [evaluation_dir, reports_dir]:
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Checked or created directory: {dir_path}")

# ===== Helper Functions ===== #
def truncate_to_hour(dt):
    """Truncate a datetime object to the hour (set minutes and seconds to 0)."""
    return dt.replace(minute=0, second=0, microsecond=0)

def fetch_data_for_date(base_url, date):
    """Try to fetch data for a specific date."""
    api_url = f"{base_url}/{date.strftime('%Y-%m-%d')}"
    response = requests.get(api_url)
    if response.status_code == 200 and "data" in response.json() and response.json()["data"]:
        return response.json()["data"]
    return None

def determine_latest_date(base_url):
    """Determine the latest available date by checking consecutive days backward."""
    current_date = datetime.now()
    for days_back in range(30):  # Check up to 30 days back
        check_date = current_date - timedelta(days=days_back)
        logger.info(f"Checking for data on {check_date.strftime('%Y-%m-%d')}...")
        data = fetch_data_for_date(base_url, check_date)
        if data:
            logger.info(f"Data found for {check_date.strftime('%Y-%m-%d')}")
            latest_time = max(
                datetime.strptime(record[0]["stringValue"], "%Y-%m-%d %H:%M:%S") for record in data
            )
            return truncate_to_hour(latest_time)
    raise ValueError("No data found for the past 30 days.")

def fetch_actual_data(base_url, start_date, end_date):
    """Fetch actual demand data between start_date and end_date."""
    api_url = f"{base_url}/{start_date.strftime('%Y-%m-%d')}to{end_date.strftime('%Y-%m-%d')}"
    logger.info(f"Fetching actual data from {start_date} to {end_date}")
    response = requests.get(api_url)
    response.raise_for_status()
    raw_data = response.json()["data"]
    flattened_data = [
        {"ds": pd.to_datetime(record[0]["stringValue"]),
         "actual": record[1]["longValue"]}
        for record in raw_data
    ]
    return pd.DataFrame(flattened_data)

def merge_with_actual(predictions_df, actual_df):
    """Merge predictions with actual data based on timestamps."""
    predictions_df["ds"] = pd.to_datetime(predictions_df["ds"])
    actual_df["ds"] = pd.to_datetime(actual_df["ds"])
    merged_df = pd.merge(predictions_df, actual_df, on="ds", how="inner")
    logger.info(f"Merged {len(merged_df)} records between predictions and actual data.")
    return merged_df

def calculate_metrics(predictions, model_name):
    """Calculate MAE and RMSE metrics."""
    if predictions.empty:
        logger.warning(f"No data to calculate metrics for {model_name}.")
        return {"Model": model_name, "MAE": None, "RMSE": None}
    
    mae = mean_absolute_error(predictions["actual"], predictions["Predicted"])
    rmse = mean_squared_error(predictions["actual"], predictions["Predicted"], squared=False)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse}

# ===== Main Execution ===== #
try:
    # Step 1: Determine the latest available date
    base_url = "https://api.electforecast.de/demand/2025"
    latest_date = determine_latest_date(base_url)
    logger.info(f"Latest available data date: {latest_date}")

    # Step 2: Calculate start_date and end_date
    end_date = latest_date
    start_date = truncate_to_hour(end_date - timedelta(days=14))

    # Step 3: Fetch actual data
    actual_data = fetch_actual_data(base_url, start_date, end_date)


    # Step 4: Fetch actual data
    actual_data = fetch_actual_data(base_url, start_date, end_date)
    print("Actual Data:")
    print(actual_data)

    # Step 5: Load predictions
    gbr_predictions = pd.read_csv(os.path.join(evaluation_dir, "gbr_future_future_predictions.csv"))
    prophet_predictions = pd.read_csv(os.path.join(evaluation_dir, "prophet_future_future_predictions.csv"))
    theta_predictions = pd.read_csv(os.path.join(evaluation_dir, "theta_future_future_predictions.csv"))

    # Print predictions for debugging
    print("GBR Predictions:")
    print(gbr_predictions[["ds", "Predicted"]].head())
    print("Prophet Predictions:")
    print(prophet_predictions[["ds", "Predicted"]].head())
    print("Theta Predictions:")
    print(theta_predictions[["ds", "Predicted"]].head())

    # Merge predictions with actual data
    gbr_predictions = merge_with_actual(gbr_predictions, actual_data)
    prophet_predictions = merge_with_actual(prophet_predictions, actual_data)
    theta_predictions = merge_with_actual(theta_predictions, actual_data)

    # Print merged data for debugging
    print("Merged GBR Predictions:")
    print(gbr_predictions)
    print("Merged Prophet Predictions:")
    print(prophet_predictions)
    print("Merged Theta Predictions:")
    print(theta_predictions)

    # Step 4: Load predictions
    gbr_predictions = pd.read_csv(os.path.join(evaluation_dir, "gbr_future_predictions.csv"))
    prophet_predictions = pd.read_csv(os.path.join(evaluation_dir, "prophet_future_predictions.csv"))
    theta_predictions = pd.read_csv(os.path.join(evaluation_dir, "theta_future_predictions.csv"))

    # Step 5: Merge predictions with actual data
    gbr_predictions = merge_with_actual(gbr_predictions, actual_data)
    prophet_predictions = merge_with_actual(prophet_predictions, actual_data)
    theta_predictions = merge_with_actual(theta_predictions, actual_data)

    # Step 6: Evaluate models
    metrics = []
    metrics.append(calculate_metrics(gbr_predictions, "GradientBoostingRegressor"))
    metrics.append(calculate_metrics(prophet_predictions, "Prophet"))
    metrics.append(calculate_metrics(theta_predictions, "Darts Theta"))

    # Step 7: Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_file = os.path.join(reports_dir, "real_time_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Saved metrics to {metrics_file}")

    # Optional: Save merged predictions
    gbr_predictions.to_csv(os.path.join(evaluation_dir, "gbr_predictions_merged.csv"), index=False)
    prophet_predictions.to_csv(os.path.join(evaluation_dir, "prophet_predictions_merged.csv"), index=False)
    theta_predictions.to_csv(os.path.join(evaluation_dir, "theta_predictions_merged.csv"), index=False)
    logger.info("Process completed successfully!")

except Exception as e:
    logger.error(f"Error during processing: {e}")
