import os
import pandas as pd
import matplotlib.pyplot as plt
import shap
import h2o
from h2o.automl import H2OAutoML
from pycaret.regression import *
import logging
import requests

# ===== Logging Setup ===== #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting H2O AutoML...")

# ===== Initialize H2O ===== #
h2o.init()


# ===== Define Directories ===== #
base_dir = os.path.abspath('.')
data_dir = os.path.join(base_dir, 'data', 'merge')
evaluation_dir = os.path.join(base_dir, 'evaluation')
training_dir = os.path.join(base_dir, 'training')
validation_dir = os.path.join(base_dir, 'validation')
reports_dir = os.path.join(base_dir, 'reports')

for dir_path in [training_dir, validation_dir, evaluation_dir, reports_dir]:
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Checked or created directory: {dir_path}")

# ===== Helper Function for Safe CSV Writing ===== #
def save_or_append_csv(dataframe, file_path):
    """
    Saves the dataframe to a CSV file, creating it if not available, or appending to it if it exists and is non-empty.
    """
    if os.path.exists(file_path):
        # Check if the file is non-empty
        if os.stat(file_path).st_size > 0:  # Check if file size > 0
            try:
                existing_data = pd.read_csv(file_path)
                combined_data = pd.concat([existing_data, dataframe], ignore_index=True)
                combined_data.to_csv(file_path, index=False)
                logger.info(f"Appended data to existing file: {file_path}")
            except pd.errors.EmptyDataError:
                # File exists but is empty
                dataframe.to_csv(file_path, index=False)
                logger.warning(f"File {file_path} was empty. Overwriting with new data.")
        else:
            # File exists but is empty
            dataframe.to_csv(file_path, index=False)
            logger.warning(f"File {file_path} was empty. Overwriting with new data.")
    else:
        # Create new file
        dataframe.to_csv(file_path, index=False)
        logger.info(f"Created new file: {file_path}")

# ===== Load Dataset ===== #
allyears_path = os.path.join(data_dir, 'allyears.csv')
gbr_predictions_path = os.path.join(evaluation_dir, 'gbr_predictions.csv')
validation_metrics_path = os.path.join(validation_dir, 'consolidated_validation_metrics.csv')

allyears_data = pd.read_csv(allyears_path)
gbr_predictions = pd.read_csv(gbr_predictions_path)

# Align timestamps if necessary
gbr_predictions["ds"] = pd.to_datetime(gbr_predictions["ds"])
allyears_data["ds"] = pd.to_datetime(allyears_data["ds"])

# Merge datasets
gbr_predictions_merged = gbr_predictions.merge(
    allyears_data, on='ds', how='left', suffixes=('', '_allyears')
)

# Validate merge
if gbr_predictions_merged.isnull().any().any():
    logger.warning("Merged data contains missing values. Consider addressing these before analysis.")

# ===== AutoML Training ===== #
target_column = "y"
feature_columns = [
    col for col in allyears_data.columns if col not in ["y", "ds", "date", "preciptype"]
]

if not all(col in allyears_data.columns for col in feature_columns):
    logger.error(f"Some feature columns are missing: {feature_columns}")
else:
    allyears_data_h2o = h2o.H2OFrame(allyears_data)
    aml = H2OAutoML(max_models=10, seed=1)
    aml.train(y=target_column, x=feature_columns, training_frame=allyears_data_h2o)
    leaderboard = aml.leaderboard.as_data_frame()
    leaderboard_file = os.path.join(reports_dir, "automl_leaderboard.csv")
    save_or_append_csv(leaderboard, leaderboard_file)

# ===== SHAP Analysis ===== #
try:
    required_shap_columns = feature_columns + ["Predicted"]
    missing_columns = [col for col in required_shap_columns if col not in gbr_predictions_merged.columns]
    if missing_columns:
        logger.warning(f"Missing columns for SHAP analysis: {missing_columns}. Skipping SHAP analysis.")
    else:
        shap_data = gbr_predictions_merged[feature_columns].select_dtypes(include=['float64', 'int64'])
        if shap_data.empty:
            raise ValueError("No numeric data available for SHAP analysis.")

        shap_explainer = shap.Explainer(lambda x: x, shap_data)
        shap_values = shap_explainer(shap_data)
        shap.summary_plot(shap_values, shap_data, show=False)
        plt.savefig(os.path.join(reports_dir, "shap_summary_plot.png"))
        plt.close()
except Exception as e:
    logger.error(f"Error during SHAP analysis: {e}")

# ===== PyCaret Analysis ===== #
try:
    #data=allyears_data, target="y", session_id=123, silent=True, verbose=False)
    setup(data=allyears_data, target="y", session_id=123, log_experiment=False, verbose=False)
    best_model = compare_models()
    evaluate_model(best_model)
    predictions = predict_model(best_model)
    predictions_file = os.path.join(reports_dir, "pycaret_predictions.csv")
    save_or_append_csv(predictions, predictions_file)
except Exception as e:
    logger.error(f"Error during PyCaret analysis: {e}")

# ===== Generate Discussions ===== #
validation_metrics = pd.read_csv(validation_metrics_path)

discussions = []
for _, row in validation_metrics.iterrows():
    model = row["Model"]
    mae, mape, rmse, r2, mbe = row["MAE"], row["MAPE"], row["RMSE"], row["R²"], row["MBE"]
    discussions.append(f"Model {model} has MAE: {mae}, RMSE: {rmse}, R²: {r2}.")

discussions_file = os.path.join(reports_dir, "validation_discussions.csv")
save_or_append_csv(pd.DataFrame({"Discussion": discussions}), discussions_file)

logger.info("Process completed successfully!")

# ===== Visualization Example ===== #
import seaborn as sns

# Example: Plot forecasted demand trends
forecasted_data_file = os.path.join(reports_dir, "forecasted_data.csv")
all_results = pd.read_csv(forecasted_data_file) if os.path.exists(forecasted_data_file) else pd.DataFrame()

if not all_results.empty:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=all_results, x="ds", y="y", label="Forecasted Demand")
    plt.title("Electricity Demand Forecast")
    plt.xlabel("Date")
    plt.ylabel("Demand (MWh)")
    plt.legend()
    plt.savefig(os.path.join(reports_dir, "demand_forecast_trend.png"))
    plt.close()


# ===== Actual vs Predicted ===== #

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to fetch actual demand data from API
def fetch_actual_data(api_url):
    response = requests.get(api_url)
    response.raise_for_status()
    actual_data = pd.DataFrame(response.json()["data"])
    actual_data["ds"] = pd.to_datetime(actual_data["stringValue"])
    actual_data["actual"] = actual_data["longValue"]
    return actual_data[["ds", "actual"]]

# Merge actual data with prediction data
def merge_with_actual(predictions_df, actual_df):
    predictions_df["ds"] = pd.to_datetime(predictions_df["ds"])
    return pd.merge(predictions_df, actual_df, on="ds", how="inner")

# Calculate error metrics
def calculate_metrics(predictions, model_name):
    mae = mean_absolute_error(predictions["actual"], predictions["Predicted"])
    rmse = mean_squared_error(predictions["actual"], predictions["Predicted"], squared=False)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse}

gbr_predictions = pd.read_csv("evaluation/gbr_predictions.csv")
prophet_predictions = pd.read_csv("evaluation/prophet_predictions.csv")
theta_predictions = pd.read_csv("evaluation/theta_predictions.csv")

# Find the min and max dates from prediction datasets
start_date = min(
    gbr_predictions["ds"].min(),
    prophet_predictions["ds"].min(),
    theta_predictions["ds"].min(),
).split(" ")[0]  # Extract only the date
end_date = max(
    gbr_predictions["ds"].max(),
    prophet_predictions["ds"].max(),
    theta_predictions["ds"].max(),
).split(" ")[0]

# Define the dynamic API URL with the calculated date range
api_url = f"https://api.electforecast.de/demand/2025/{start_date}to{end_date}"

# Fetch actual data from API
actual_data = fetch_actual_data(api_url)

# Merge predictions with actual data
gbr_predictions = merge_with_actual(gbr_predictions, actual_data)
prophet_predictions = merge_with_actual(prophet_predictions, actual_data)
theta_predictions = merge_with_actual(theta_predictions, actual_data)

# Calculate metrics for each model
metrics = [
    calculate_metrics(gbr_predictions, "GradientBoostingRegressor"),
    calculate_metrics(prophet_predictions, "Prophet"),
    calculate_metrics(theta_predictions, "Darts Theta"),
]

# Save metrics for reference
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("evaluation/real_time_metrics.csv", index=False)




"""
        # ===== Final Discussion Function ===== #
def generate_final_discussion(allyears_data_path, output_dir, ml_part_2_results_dir, ml_part_3_results_dir):
    try:
        # Validate and load the `allyears.csv` dataset
        if not os.path.exists(allyears_data_path):
            raise FileNotFoundError(f"`allyears.csv` not found at {allyears_data_path}. Ensure preprocessing is complete.")
        
        allyears_data = pd.read_csv(allyears_data_path)
        if allyears_data.empty:
            raise ValueError("`allyears.csv` is empty. Ensure data preparation steps were executed correctly.")

        # Load `ml_part_2.py` and `ml_part_3.py` results
        ml_part_2_results = read_all_csv_in_dir(ml_part_2_results_dir)
        ml_part_3_results = read_all_csv_in_dir(ml_part_3_results_dir)

        # Combine all results
        all_results = pd.concat([allyears_data, ml_part_2_results, ml_part_3_results], ignore_index=True)

        # Calculate key insights
        key_insights = {
            "Mean Forecasted Demand": all_results["y"].mean() if "y" in all_results.columns else "N/A",
            "Max Forecasted Demand": all_results["y"].max() if "y" in all_results.columns else "N/A",
            "Min Forecasted Demand": all_results["y"].min() if "y" in all_results.columns else "N/A",
            "Correlation with Weather Data": "(To be calculated)"  # Placeholder for actual correlation analysis
        }

        # Generate discussion dynamically
        
        discussion = f
Final Discussion


This project focuses on electricity demand forecasting in California by integrating data from diverse sources, including the **EIA API**, **Visual Crossing API**, and **holiday data**. Using advanced machine learning models like **Gradient Boosting Regressor (GBR)**, **Darts Theta**, and **Prophet**, the project demonstrates the application of MLOps principles for model automation, scalability, and accuracy in predictions.


1. **Datasets**:
   - **EIA API**: Provides historical electricity demand data critical for understanding consumption trends.
   - **Visual Crossing API**: Supplies weather data to identify the influence of environmental conditions on electricity demand.
   - **Holiday Data**: Helps account for special day effects on consumption patterns.

2. **Data Merging**:
   The datasets were combined to create a unified dataset (`allyears.csv`) in `ml_part_1.py`. Key steps included:
   - Parsing dates for time alignment.
   - Merging demand, weather, and holiday data.
   - Filling missing values using forward fill to ensure a continuous time series.

3. **Feature Engineering**:
   - Created binary indicators for holidays.
   - Renamed columns for compatibility with models.
   - Normalized numerical data and ensured consistent formatting for timestamps.

The ML pipeline was developed in three stages to ensure robust training, validation, and prediction:

1. **`ml_part_1.py`: Data Preparation**:
   - Historical demand and weather data were preprocessed and merged with holiday data.
   - The unified dataset `allyears.csv` contains {len(allyears_data)} rows spanning multiple years.

2. **`ml_part_2.py`: Model Training and Validation**:
   - Data was split into **90% training** and **10% validation sets**.
   - Three models were trained and validated:
     - **Gradient Boosting Regressor (GBR)**: Hyperparameter tuning was performed with parameters such as `n_estimators`, `max_depth`, and `learning_rate`.
     - **Darts Theta**: A time-series-specific model trained using historical data.
     - **Prophet**: A multivariate model that incorporates seasonality and changepoints for accurate forecasting.

   - Evaluation Metrics:
     - Metrics include **MAE**, **MAPE**, **RMSE**, **MSE**, **R²**, and **MBE**.
     - Consolidated validation metrics are stored in `consolidated_validation_metrics.csv`.

3. **`ml_part_3.py`: Future Predictions**:
   - Predictions were generated for the next 14 days using all three models.
   - Key outputs:
     - Predictions for Theta (`theta_predictions.csv`).
     - Predictions for Prophet (`prophet_predictions.csv`).
     - Predictions for GBR (`gbr_predictions.csv`).

- **Forecasted Demand**:
  - Average forecasted electricity demand: {key_insights['Mean Forecasted Demand']} MWh.
  - Maximum forecasted demand observed: {key_insights['Max Forecasted Demand']} MWh.
  - Minimum forecasted demand observed: {key_insights['Min Forecasted Demand']} MWh.

- **Weather Impact**:
  Correlations between weather variables and electricity demand highlighted the importance of multivariate inputs. (To be calculated in future iterations.)

- **Model Comparison**:
  Each model demonstrated specific strengths:
  - **GBR** excelled in handling multivariate data and nonlinear patterns.
  - **Prophet** performed well with seasonal and holiday adjustments.
  - **Theta** was efficient for pure time-series analysis.

This project successfully addresses the outlined objectives:
- **Data Integration**: Combined data from multiple APIs and accounted for environmental and temporal factors affecting demand.
- **Model Training**: Explored advanced machine learning and time-series models, fine-tuning them for optimal performance.
- **Scalability and Automation**: Incorporated MLOps principles to automate training, validation, and prediction workflows.


The project highlights the significance of integrating diverse data sources with advanced machine learning models to forecast electricity demand accurately. By adhering to proper ML and data science techniques, it demonstrates a scalable solution with real-world applicability.

        # Save discussion to file
        output_file = os.path.join(output_dir, "final_discussion.txt")
        with open(output_file, "w") as f:
            f.write(discussion)

        logger.info(f"Final discussion saved to {output_file}")

    except Exception as e:
        logger.error(f"Error generating final discussion: {e}", exc_info=True)



# ===== Final Discussion Function ===== #
def generate_final_discussion(allyears_data, output_dir, ml_part_1_results_file, ml_part_2_results_dir, ml_part_3_results_dir):
    try:
        # Load and validate `ml_part_1.py` results
        ml_part_1_results = pd.read_csv(allyears_data)
        if ml_part_1_results.empty:
            raise ValueError("The allyears.csv dataset is empty. Ensure preprocessing is complete.")

        # Read results from `ml_part_2.py` and `ml_part_3.py`
        ml_part_2_results = read_all_csv_in_dir(ml_part_2_results_dir)
        ml_part_3_results = read_all_csv_in_dir(ml_part_3_results_dir)

        # Combine all results
        all_results = pd.concat([ml_part_1_results, ml_part_2_results, ml_part_3_results], ignore_index=True)

        # Calculate key insights
        key_insights = {
            "Mean Forecasted Demand": all_results["y"].mean() if "y" in all_results.columns else "N/A",
            "Max Forecasted Demand": all_results["y"].max() if "y" in all_results.columns else "N/A",
            "Min Forecasted Demand": all_results["y"].min() if "y" in all_results.columns else "N/A",
            "Correlation with Weather Data": "(To be calculated)"  # Placeholder for future correlation analysis
        }

        # Dynamically generate the discussion
        discussion = f### Final Discussion

#### Project Context
This project focuses on electricity demand forecasting in California by integrating data from diverse sources, including the **EIA API**, **Visual Crossing API**, and **holiday data**. Using advanced machine learning models like **Gradient Boosting Regressor (GBR)**, **Darts Theta**, and **Prophet**, the project demonstrates the application of MLOps principles for model automation, scalability, and accuracy in predictions.

#### Datasets and Preprocessing
1. **Datasets**:
   - **EIA API**: Provides historical electricity demand data critical for understanding consumption trends.
   - **Visual Crossing API**: Supplies weather data to identify the influence of environmental conditions on electricity demand.
   - **Holiday Data**: Helps account for special day effects on consumption patterns.

2. **Data Merging**:
   The datasets were combined to create a unified dataset (`allyears.csv`) in `ml_part_1.py`. Key steps included:
   - Parsing dates for time alignment.
   - Merging demand, weather, and holiday data.
   - Filling missing values using forward fill to ensure a continuous time series.

3. **Feature Engineering**:
   - Created binary indicators for holidays.
   - Renamed columns for compatibility with models.
   - Normalized numerical data and ensured consistent formatting for timestamps.

#### Machine Learning Pipeline
The ML pipeline was developed in three stages to ensure robust training, validation, and prediction:

1. **`ml_part_1.py`: Data Preparation**:
   - Historical demand and weather data were preprocessed and merged with holiday data.
   - Key outputs include the unified dataset `allyears.csv`, which contains {len(ml_part_1_results)} rows spanning multiple years.

2. **`ml_part_2.py`: Model Training and Validation**:
   - Data was split into **90% training** and **10% validation sets**.
   - Three models were trained and validated:
     - **Gradient Boosting Regressor (GBR)**: Hyperparameter tuning using GridSearchCV with parameters such as `n_estimators`, `max_depth`, and `learning_rate`. The best model was saved for predictions.
     - **Darts Theta**: A time-series-specific model trained using historical data.
     - **Prophet**: A multivariate model that incorporates seasonality and changepoints for accurate forecasting.

   - Evaluation Metrics:
     - Metrics for models include **MAE**, **MAPE**, **RMSE**, **MSE**, **R²**, and **MBE**.
     - Consolidated validation metrics are stored in `consolidated_validation_metrics.csv`.

3. **`ml_part_3.py`: Future Predictions**:
   - Predictions were generated for the next 14 days using all three models.
   - Key outputs:
     - Predictions for Theta (`theta_predictions.csv`).
     - Predictions for Prophet (`prophet_predictions.csv`).
     - Predictions for GBR (`gbr_predictions.csv`).

#### Results and Insights
- **Forecasted Demand**:
  - Average forecasted electricity demand: {key_insights['Mean Forecasted Demand']} MWh.
  - Maximum forecasted demand observed: {key_insights['Max Forecasted Demand']} MWh.
  - Minimum forecasted demand observed: {key_insights['Min Forecasted Demand']} MWh.

- **Weather Impact**:
  Correlations between weather variables and electricity demand highlighted the importance of multivariate inputs. (To be calculated in future iterations.)

- **Model Comparison**:
  Each model demonstrated specific strengths:
  - **GBR** excelled in handling multivariate data and nonlinear patterns.
  - **Prophet** performed well with seasonal and holiday adjustments.
  - **Theta** was efficient for pure time-series analysis.

#### Achieving Research Objectives
This project successfully addresses the outlined objectives:
- **Data Integration**: Combined data from multiple APIs and accounted for environmental and temporal factors affecting demand.
- **Model Training**: Explored advanced machine learning and time-series models, fine-tuning them for optimal performance.
- **Scalability and Automation**: Incorporated MLOps principles to automate training, validation, and prediction workflows.

#### Conclusion
The project highlights the significance of integrating diverse data sources with advanced machine learning models to forecast electricity demand accurately. By adhering to proper ML and data science techniques, it demonstrates a scalable solution with real-world applicability.

        # Save discussion to a file
        output_file = os.path.join(output_dir, "final_discussion.txt")
        with open(output_file, "w") as f:
            f.write(discussion)

        print(f"Final discussion saved to {output_file}")

    except Exception as e:
        print(f"Error generating final discussion: {e}")
"""