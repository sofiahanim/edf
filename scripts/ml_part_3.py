import os
import pandas as pd
import matplotlib.pyplot as plt
import shap
from h2o.automl import H2OAutoML
from datetime import datetime
import h2o

# Initialize H2O
h2o.init()

# Directories for input CSVs
training_dir = "training"
validation_dir = "validation"
evaluation_dir = "evaluation"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

# Function to read all CSV files in a directory
def read_all_csv_in_dir(directory):
    """
    Reads all CSV files in the specified directory and combines them into a single DataFrame.
    Each file's data is appended with its source file name.
    """
    combined_data = pd.DataFrame()
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith(".csv") and os.path.isfile(file_path):
            try:
                df = pd.read_csv(file_path)
                df["Source_File"] = file_name  # Add file name for reference
                combined_data = pd.concat([combined_data, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
    return combined_data

# Function to calculate summary metrics
def calculate_summary_metrics(data, metrics_columns):
    """
    Calculates mean, standard deviation, count, min, and max for specified metrics columns.
    """
    summary = {}
    for col in metrics_columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            summary[col] = {
                "Mean": data[col].mean(),
                "StdDev": data[col].std(),
                "Count": data[col].count(),
                "Min": data[col].min(),
                "Max": data[col].max()
            }
    return summary

# Function to save summary metrics as a CSV
def save_summary_report(summary, output_file):
    """
    Saves the summary dictionary to a CSV file.
    """
    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={"index": "Metric"}, inplace=True)
    summary_df["Last Updated At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_df.to_csv(output_file, index=False)
    print(f"Summary report saved to {output_file}")

# Function to analyze CSVs from a directory
def analyze_directory(directory, output_file, metrics_columns):
    """
    Reads all CSVs from a directory, calculates summary metrics, and saves the report.
    """
    print(f"Analyzing directory: {directory}")
    data = read_all_csv_in_dir(directory)
    if data.empty:
        print(f"No data found in {directory}.")
        return

    summary = calculate_summary_metrics(data, metrics_columns)
    save_summary_report(summary, output_file)

# Function to perform H2O AutoML and save the leaderboard
def perform_h2o_automl(data, target_column, max_models=5):
    """
    Perform AutoML using H2O and return the leaderboard.
    """
    h2o_data = h2o.H2OFrame(data)
    train, test = h2o_data.split_frame(ratios=[0.8], seed=42)
    aml = H2OAutoML(max_models=max_models, seed=42)
    aml.train(y=target_column, training_frame=train)

    leaderboard = aml.leaderboard.as_data_frame()
    leaderboard_file = os.path.join(reports_dir, "automl_leaderboard.csv")
    leaderboard.to_csv(leaderboard_file, index=False)
    print(f"AutoML leaderboard saved to {leaderboard_file}")

    return leaderboard

# Function to generate SHAP visualizations
def explain_model_with_shap(model, data, output_dir):
    """
    Generate SHAP explanations for the model and save visualizations.
    """
    explainer = shap.Explainer(model.predict, data)
    shap_values = explainer(data)

    # Save summary plot
    shap.summary_plot(shap_values, data, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.close()

    # Save individual force plot for the first prediction
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0],
        data.iloc[0],
        matplotlib=True
    )
    plt.savefig(os.path.join(output_dir, "shap_force_plot.png"))
    plt.close()

# Function to generate manual discussions
def generate_discussion(metrics_file, output_file):
    """
    Generates an automated discussion based on metrics in the input CSV file.
    """
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

# Main logic for analyzing all results
metrics_columns = ["MAE", "MAPE", "RMSE", "MSE", "R²", "MBE"]

# Analyze training
training_summary_file = os.path.join(reports_dir, "training_summary.csv")
analyze_directory(training_dir, training_summary_file, metrics_columns)
generate_discussion(training_summary_file, os.path.join(reports_dir, "training_discussion.txt"))

# Analyze validation
validation_summary_file = os.path.join(reports_dir, "validation_summary.csv")
analyze_directory(validation_dir, validation_summary_file, metrics_columns)
generate_discussion(validation_summary_file, os.path.join(reports_dir, "validation_discussion.txt"))

# Analyze evaluation
evaluation_summary_file = os.path.join(reports_dir, "evaluation_summary.csv")
analyze_directory(evaluation_dir, evaluation_summary_file, metrics_columns)
generate_discussion(evaluation_summary_file, os.path.join(reports_dir, "evaluation_discussion.txt"))

# Perform AutoML on training data
training_data = pd.concat([pd.read_csv(os.path.join(training_dir, f)) for f in os.listdir(training_dir) if f.endswith(".csv")])
leaderboard = perform_h2o_automl(training_data, target_column="y")

# SHAP explanation for top model
top_model_data = training_data.drop(columns=["y"])
explain_model_with_shap(leaderboard.iloc[0]["model_id"], top_model_data, reports_dir)
