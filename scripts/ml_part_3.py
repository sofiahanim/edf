import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import h2o
from h2o.automl import H2OAutoML
from tpot import TPOTClassifier
from autosklearn.classification import AutoSklearnClassifier

# Initialize H2O Cluster
h2o.init()

# Utility to append results with timestamp and source
def append_results(df, output_csv, source):
    df['Timestamp'] = datetime.now()
    df['Source'] = source
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_csv(output_csv, index=False)

# Combine all files into a single DataFrame
def combine_files(file_dict):
    combined_data = pd.DataFrame()
    for name, path in file_dict.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Source'] = name
            combined_data = pd.concat([combined_data, df], ignore_index=True)
    return combined_data

# Analyze metrics using Scikit-learn
def analyze_with_sklearn(data, output_csv):
    summary = data.groupby('Source').mean().reset_index()
    insights = []
    for _, row in summary.iterrows():
        source = row['Source']
        mae = row.get('MAE', np.nan)
        mse = row.get('MSE', np.nan)
        rmse = np.sqrt(mse) if not np.isnan(mse) else np.nan
        insights.append({"Source": source, "MAE": mae, "MSE": mse, "RMSE": rmse})
    insights_df = pd.DataFrame(insights)
    append_results(insights_df, output_csv, "scikit-learn")

# Analyze metrics using H2O AutoML
def analyze_with_h2o(data, target_column, output_csv):
    h2o_data = h2o.H2OFrame(data)
    train, test = h2o_data.split_frame(ratios=[0.8], seed=42)
    aml = H2OAutoML(max_models=5, seed=42)
    aml.train(y=target_column, training_frame=train)
    leaderboard = aml.leaderboard.as_data_frame()
    append_results(pd.DataFrame(leaderboard), output_csv, "h2o")

# Analyze metrics using TPOT
def analyze_with_tpot(data, target_column, output_csv):
    train = data.sample(frac=0.8, random_state=42)
    test = data.drop(train.index)
    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
    tpot.fit(train.drop(columns=[target_column]), train[target_column])
    score = tpot.score(test.drop(columns=[target_column]), test[target_column])
    tpot.export("tpot_pipeline.py")
    append_results(pd.DataFrame([{"Accuracy": score}]), output_csv, "tpot")

# Analyze metrics using Auto-sklearn
def analyze_with_autosklearn(data, target_column, output_csv):
    train = data.sample(frac=0.8, random_state=42)
    test = data.drop(train.index)
    automl = AutoSklearnClassifier(time_left_for_this_task=600, per_run_time_limit=60)
    automl.fit(train.drop(columns=[target_column]), train[target_column])
    predictions = automl.predict(test.drop(columns=[target_column]))
    mae = mean_absolute_error(test[target_column], predictions)
    mse = mean_squared_error(test[target_column], predictions)
    rmse = np.sqrt(mse)
    append_results(pd.DataFrame([{"MAE": mae, "MSE": mse, "RMSE": rmse}]), output_csv, "autosklearn")

# Main process
def process_all_tools(evaluation_files, training_files, validation_files, target_column):
    evaluation_data = combine_files(evaluation_files)
    training_data = combine_files(training_files)
    validation_data = combine_files(validation_files)

    os.makedirs("reports", exist_ok=True)

    # Scikit-learn Analysis
    analyze_with_sklearn(evaluation_data, "reports/evaluation_metrics.csv")
    analyze_with_sklearn(training_data, "reports/training_metrics.csv")
    analyze_with_sklearn(validation_data, "reports/validation_metrics.csv")

    # H2O AutoML Analysis
    analyze_with_h2o(evaluation_data, target_column, "reports/evaluation_metrics.csv")
    analyze_with_h2o(training_data, target_column, "reports/training_metrics.csv")
    analyze_with_h2o(validation_data, target_column, "reports/validation_metrics.csv")

    # TPOT Analysis
    analyze_with_tpot(evaluation_data, target_column, "reports/evaluation_metrics.csv")
    analyze_with_tpot(training_data, target_column, "reports/training_metrics.csv")
    analyze_with_tpot(validation_data, target_column, "reports/validation_metrics.csv")

    # Auto-sklearn Analysis
    analyze_with_autosklearn(evaluation_data, target_column, "reports/evaluation_metrics.csv")
    analyze_with_autosklearn(training_data, target_column, "reports/training_metrics.csv")
    analyze_with_autosklearn(validation_data, target_column, "reports/validation_metrics.csv")

# Define files and target column
evaluation_files = {
    "darts_theta_predictions": os.path.join("evaluation", "darts_theta_predictions.csv"),
    "h2o_predictions": os.path.join("evaluation", "h2o_predictions.csv"),
    "prophet_predictions": os.path.join("evaluation", "prophet_predictions.csv"),
    "summary_report": os.path.join("evaluation", "summary_report.csv"),
}
training_files = {
    "darts_theta_metrics": os.path.join("training", "darts theta_metrics.csv"),
    "h2o_metrics": os.path.join("training", "h2o_metrics.csv"),
    "prophet_metrics": os.path.join("training", "prophet_metrics.csv"),
    "training_info": os.path.join("training", "training_info.csv"),
}
validation_files = {
    "h2o_validation_metrics": os.path.join("validation", "h2o_validation_metrics.csv"),
    "prophet_validation_metrics": os.path.join("validation", "prophet_validation_metrics.csv"),
    "theta_validation_metrics": os.path.join("validation", "theta_validation_metrics.csv"),
    "consolidated_validation_metrics": os.path.join("validation", "consolidated_validation_metrics.csv"),
}

# Process all tools
process_all_tools(evaluation_files, training_files, validation_files, "target_column")
