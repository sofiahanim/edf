import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import h2o
from h2o.automl import H2OAutoML
from pycaret.regression import *
from datetime import datetime

# ===== Initialize H2O ===== #
h2o.init()

# ===== Define Directories ===== #
base_dir = os.path.abspath('.')
data_dir = os.path.join(base_dir, 'data', 'merge')
evaluation_dir = os.path.join(base_dir, 'evaluation')
training_dir = os.path.join(base_dir, 'training')
validation_dir = os.path.join(base_dir, 'validation')
reports_dir = os.path.join(base_dir, 'reports')
os.makedirs(reports_dir, exist_ok=True)

# Load allyears dataset
allyears_path = os.path.join(data_dir, 'allyears.csv')
allyears_data = pd.read_csv(allyears_path)

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
aml = H2OAutoML(max_models=10, seed=1)
aml.train(y=target_column, x=feature_columns, training_frame=allyears_data_h2o)
leaderboard = aml.leaderboard.as_data_frame()
print("\nH2O AutoML Leaderboard:")
print(leaderboard)

# ===== SHAP Analysis ===== #
gbr_predictions_path = folders["evaluation"]["gbr_predictions"]
gbr_predictions = pd.read_csv(gbr_predictions_path)
if "temp" in gbr_predictions.columns:
    explainer = shap.Explainer(lambda x: x, gbr_predictions[feature_columns])
    shap_values = explainer(gbr_predictions[feature_columns])
    shap.summary_plot(shap_values, gbr_predictions[feature_columns])

# ===== PyCaret Analysis ===== #
consolidated_path = folders["validation"]["consolidated"]
consolidated = pd.read_csv(consolidated_path)
pycaret_setup = setup(data=consolidated, target="MAE", session_id=123)
best_model = compare_models()
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
