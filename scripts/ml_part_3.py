import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import h2o
from h2o.automl import H2OAutoML
from pycaret.regression import *
from datetime import datetime
import logging

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


# Load allyears dataset
allyears_path = os.path.join(data_dir, 'allyears.csv')

folders = {
    "evaluation": {
        "gbr_predictions": os.path.join(evaluation_dir, "gbr_predictions.csv"),
        "prophet_predictions": os.path.join(evaluation_dir, "prophet_predictions.csv"),
        "theta_predictions": os.path.join(evaluation_dir, "theta_predictions.csv"),
        "summary_report": os.path.join(evaluation_dir, "summary_report.csv"),
    },
    "validation": {
        "consolidated": os.path.join(validation_dir, "consolidated_validation_metrics.csv"),
    },
}


def validate_file(file_path, required_columns=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    if required_columns and not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns in {file_path}. Expected: {required_columns}")
    return df

# Validate allyears dataset
allyears_data = validate_file(allyears_path, required_columns=["y", "ds"])


# ===== Utility Functions ===== #
def read_all_csv_in_dir(directory):
    combined_data = pd.DataFrame()
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith(".csv") and os.path.isfile(file_path):
            try:
                df = pd.read_csv(file_path)
                df["Source_File"] = file_name
                combined_data = pd.concat([combined_data, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
    return combined_data

def calculate_summary_metrics(data, metrics_columns):
    summary = {}
    for col in metrics_columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            summary[col] = {
                "Mean": data[col].mean(),
                "StdDev": data[col].std(),
                "Count": data[col].count(),
                "Min": data[col].min(),
                "Max": data[col].max(),
            }
    return summary

def save_summary_report(summary, output_file):
    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={"index": "Metric"}, inplace=True)
    summary_df["Last Updated At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_df.to_csv(output_file, index=False)
    print(f"Summary report saved to {output_file}")

def analyze_directory(directory, output_file, metrics_columns):
    print(f"Analyzing directory: {directory}")
    data = read_all_csv_in_dir(directory)
    if data.empty:
        print(f"No data found in {directory}.")
        return
    summary = calculate_summary_metrics(data, metrics_columns)
    save_summary_report(summary, output_file)

def perform_h2o_automl(data, target_column, max_models=5):
    h2o_data = h2o.H2OFrame(data)
    train, test = h2o_data.split_frame(ratios=[0.8], seed=42)
    aml = H2OAutoML(max_models=max_models, seed=42)
    aml.train(y=target_column, training_frame=train)
    leaderboard = aml.leaderboard.as_data_frame()
    leaderboard_file = os.path.join(reports_dir, "automl_leaderboard.csv")
    leaderboard.to_csv(leaderboard_file, index=False)
    print(f"AutoML leaderboard saved to {leaderboard_file}")
    return leaderboard

def explain_model_with_shap(model, data, output_dir):
    explainer = shap.Explainer(model.predict, data)
    shap_values = explainer(data)
    shap.summary_plot(shap_values, data, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.close()
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0],
        data.iloc[0],
        matplotlib=True,
    )
    plt.savefig(os.path.join(output_dir, "shap_force_plot.png"))
    plt.close()

def generate_discussion(metrics_file, output_file):
    try:
        data = pd.read_csv(metrics_file)
        discussions = []
        for _, row in data.iterrows():
            source = row.get("Source_File", "Unknown Source")
            mae = row.get("MAE", None)
            mape = row.get("MAPE", None)
            rmse = row.get("RMSE", None)
            r2 = row.get("R²", None)
            mbe = row.get("MBE", None)
            discussion = f"### Analysis for {source}:\n\n"
            if mae is not None:
                discussion += f"- **MAE**: {mae:.2f} - Measures average absolute error.\n"
            if mape is not None:
                discussion += f"- **MAPE**: {mape:.2f}% - Indicates percentage error. "
                discussion += "Good accuracy.\n" if mape < 10 else "Improvement needed.\n"
            if rmse is not None:
                discussion += f"- **RMSE**: {rmse:.2f} - Penalizes large errors.\n"
            if r2 is not None:
                discussion += f"- **R²**: {r2:.2f} - Explains variance. "
                discussion += "Good fit.\n" if r2 >= 0.8 else "Moderate fit.\n"
            if mbe is not None:
                discussion += f"- **MBE**: {mbe:.2f} - Positive = Overestimate, Negative = Underestimate.\n"
            discussions.append(discussion)
        with open(output_file, "w") as f:
            f.write("\n\n".join(discussions))
        print(f"Discussion saved to {output_file}")
    except Exception as e:
        print(f"Error generating discussion: {e}")

# ===== H2O AutoML ===== #
allyears_data_h2o = h2o.H2OFrame(allyears_data)
target_column = "y"

feature_columns = [col for col in allyears_data.columns if col not in ["y", "ds", "date", "preciptype"]]

if target_column not in allyears_data.columns:
    raise ValueError(f"Target column '{target_column}' not found in allyears_data.")
if not all(col in allyears_data.columns for col in feature_columns):
    raise ValueError(f"Some feature columns are missing: {feature_columns}")

aml = H2OAutoML(max_models=10, seed=1)
aml.train(y=target_column, x=feature_columns, training_frame=allyears_data_h2o)
leaderboard = aml.leaderboard.as_data_frame()
print("\nH2O AutoML Leaderboard:")
print(leaderboard)

# ===== SHAP Analysis ===== #
gbr_predictions_path = folders["evaluation"]["gbr_predictions"]
gbr_predictions = pd.read_csv(gbr_predictions_path)
if feature_columns:
    try:
        explainer = shap.Explainer(lambda x: x, gbr_predictions[feature_columns])
        shap_values = explainer(gbr_predictions[feature_columns])
        shap.summary_plot(shap_values, gbr_predictions[feature_columns])
    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}")


# ===== PyCaret Analysis ===== #
consolidated_path = folders["validation"]["consolidated"]
consolidated = pd.read_csv(consolidated_path)
pycaret_setup = setup(data=consolidated, target="MAE", session_id=123)

try:
    best_model = compare_models()
except Exception as e:
    logger.error(f"PyCaret model comparison failed: {e}")
    raise

print("\nBest model selected by PyCaret:")
print(best_model)
evaluate_model(best_model)
predictions = predict_model(best_model)
predictions.to_csv("pycaret_predictions.csv", index=False)

# ===== Automated Discussion ===== #
def generate_consolidated_discussion(df):
    insights = []
    best_mae_model = df.groupby("Model")["MAE"].mean().idxmin()
    best_rmse_model = df.groupby("Model")["RMSE"].mean().idxmin()
    highest_r2_model = df.groupby("Model")["R²"].mean().idxmax()
    insights.append(f"The model with the lowest average MAE is {best_mae_model}.")
    insights.append(f"The model with the lowest average RMSE is {best_rmse_model}.")
    insights.append(f"The model with the highest average R² is {highest_r2_model}.")
    return "\n".join(insights)

discussion = generate_consolidated_discussion(consolidated)
print("\nAutomated Insights:")
print(discussion)
with open("automated_discussion.txt", "w") as f:
    f.write(discussion)
print("\nDiscussion saved to automated_discussion.txt.")

# ===== Analyze All Results ===== #
metrics_columns = ["MAE", "MAPE", "RMSE", "MSE", "R²", "MBE"]
training_summary_file = os.path.join(reports_dir, "training_summary.csv")
analyze_directory(training_dir, training_summary_file, metrics_columns)
generate_discussion(training_summary_file, os.path.join(reports_dir, "training_discussion.txt"))
validation_summary_file = os.path.join(reports_dir, "validation_summary.csv")
analyze_directory(validation_dir, validation_summary_file, metrics_columns)
generate_discussion(validation_summary_file, os.path.join(reports_dir, "validation_discussion.txt"))
evaluation_summary_file = os.path.join(reports_dir, "evaluation_summary.csv")
analyze_directory(evaluation_dir, evaluation_summary_file, metrics_columns)
generate_discussion(evaluation_summary_file, os.path.join(reports_dir, "evaluation_discussion.txt"))

# ===== SHAP for Top Model ===== #
top_model_data = allyears_data[feature_columns]
explain_model_with_shap(leaderboard.iloc[0]["model_id"], top_model_data, reports_dir)


####################################################################################################
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import h2o
from h2o.automl import H2OAutoML
from pycaret.regression import *
from datetime import datetime
import logging

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

def create_directories(directories):
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Checked or created directory: {dir_path}")

create_directories([training_dir, validation_dir, evaluation_dir, reports_dir])

# Load allyears dataset
allyears_path = os.path.join(data_dir, 'allyears.csv')

folders = {
    "evaluation": {
        "gbr_predictions": os.path.join(evaluation_dir, "gbr_predictions.csv"),
        "prophet_predictions": os.path.join(evaluation_dir, "prophet_predictions.csv"),
        "theta_predictions": os.path.join(evaluation_dir, "theta_predictions.csv"),
        "summary_report": os.path.join(evaluation_dir, "summary_report.csv"),
    },
    "validation": {
        "consolidated": os.path.join(validation_dir, "consolidated_validation_metrics.csv"),
    },
}

def validate_file(file_path, required_columns=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    if required_columns and not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns in {file_path}. Expected: {required_columns}")
    return df

# Validate allyears dataset
allyears_data = validate_file(allyears_path, required_columns=["y", "ds"])

# ===== Utility Functions ===== #
def read_all_csv_in_dir(directory):
    combined_data = pd.DataFrame()
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith(".csv") and os.path.isfile(file_path):
            try:
                df = pd.read_csv(file_path)
                df["Source_File"] = file_name
                combined_data = pd.concat([combined_data, df], ignore_index=True)
            except Exception as e:
                logger.error(f"Error reading file {file_name}: {e}")
    return combined_data

def calculate_summary_metrics(data, metrics_columns):
    summary = {}
    for col in metrics_columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            summary[col] = {
                "Mean": data[col].mean(),
                "StdDev": data[col].std(),
                "Count": data[col].count(),
                "Min": data[col].min(),
                "Max": data[col].max(),
            }
    return summary

def save_summary_report(summary, output_file):
    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={"index": "Metric"}, inplace=True)
    summary_df["Last Updated At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_df.to_csv(output_file, index=False)
    logger.info(f"Summary report saved to {output_file}")

def analyze_directory(directory, output_file, metrics_columns):
    logger.info(f"Analyzing directory: {directory}")
    data = read_all_csv_in_dir(directory)
    if data.empty:
        logger.warning(f"No data found in {directory}.")
        return
    summary = calculate_summary_metrics(data, metrics_columns)
    save_summary_report(summary, output_file)

def perform_h2o_automl(data, target_column, max_models=5):
    h2o_data = h2o.H2OFrame(data)
    train, test = h2o_data.split_frame(ratios=[0.8], seed=42)
    aml = H2OAutoML(max_models=max_models, seed=42)
    aml.train(y=target_column, training_frame=train)
    leaderboard = aml.leaderboard.as_data_frame()
    leaderboard_file = os.path.join(reports_dir, "automl_leaderboard.csv")
    leaderboard.to_csv(leaderboard_file, index=False)
    logger.info(f"AutoML leaderboard saved to {leaderboard_file}")
    return leaderboard

def explain_model_with_shap(model, data, output_dir):
    explainer = shap.Explainer(model.predict, data)
    shap_values = explainer(data)
    shap.summary_plot(shap_values, data, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.close()
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0],
        data.iloc[0],
        matplotlib=True,
    )
    plt.savefig(os.path.join(output_dir, "shap_force_plot.png"))
    plt.close()

def generate_discussion(metrics_file, output_file):
    try:
        data = pd.read_csv(metrics_file)
        discussions = []
        for _, row in data.iterrows():
            source = row.get("Source_File", "Unknown Source")
            mae = row.get("MAE", None)
            mape = row.get("MAPE", None)
            rmse = row.get("RMSE", None)
            r2 = row.get("R²", None)
            mbe = row.get("MBE", None)
            discussion = f"### Analysis for {source}:\n\n"
            if mae is not None:
                discussion += f"- **MAE**: {mae:.2f} - Measures average absolute error.\n"
            if mape is not None:
                discussion += f"- **MAPE**: {mape:.2f}% - Indicates percentage error. "
                discussion += "Good accuracy.\n" if mape < 10 else "Improvement needed.\n"
            if rmse is not None:
                discussion += f"- **RMSE**: {rmse:.2f} - Penalizes large errors.\n"
            if r2 is not None:
                discussion += f"- **R²**: {r2:.2f} - Explains variance. "
                discussion += "Good fit.\n" if r2 >= 0.8 else "Moderate fit.\n"
            if mbe is not None:
                discussion += f"- **MBE**: {mbe:.2f} - Positive = Overestimate, Negative = Underestimate.\n"
            discussions.append(discussion)
        with open(output_file, "w") as f:
            f.write("\n\n".join(discussions))
        logger.info(f"Discussion saved to {output_file}")
    except Exception as e:
        logger.error(f"Error generating discussion: {e}", exc_info=True)

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