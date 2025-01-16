import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please set it in your environment or in the .env file.")

# Define directories for data and reports
base_dir = os.path.abspath('.')
data_dir = os.path.join(base_dir, 'data', 'merge')
evaluation_dir = os.path.join(base_dir, 'evaluation')
training_dir = os.path.join(base_dir, 'training')
reports_dir = os.path.join(base_dir, 'reports')  # Reports folder in root
os.makedirs(evaluation_dir, exist_ok=True)
os.makedirs(training_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# Error logging
error_log_file = os.path.join(reports_dir, 'error_log.txt')
def log_error(message):
    with open(error_log_file, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")

# Append data to a CSV file
def append_to_csv(file_path, content):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = {'Timestamp': timestamp, 'Insights': content}
    df = pd.DataFrame([new_row])
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)
        updated = pd.concat([existing, df], ignore_index=True)
    else:
        updated = df
    updated.to_csv(file_path, index=False)
    print(f"Appended insights to {file_path}")

# Clean and load a CSV
def load_and_clean_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        # Drop irrelevant columns if present
        if 'Timestamp' in df.columns:
            df = df.drop(columns=['Timestamp'])
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if loading fails

# Define CSV paths
evaluation_files = {
    "darts_theta_predictions": os.path.join(evaluation_dir, "darts_theta_predictions.csv"),
    "h2o_predictions": os.path.join(evaluation_dir, "h2o_predictions.csv"),
    "prophet_predictions": os.path.join(evaluation_dir, "prophet_predictions.csv"),
    "summary_report": os.path.join(evaluation_dir, "summary_report.csv"),
}
training_files = {
    "darts_theta_metrics": os.path.join(training_dir, "darts_theta_metrics.csv"),
    "h2o_metrics": os.path.join(training_dir, "h2o_metrics.csv"),
    "prophet_metrics": os.path.join(training_dir, "prophet_metrics.csv"),
    "training_info": os.path.join(training_dir, "training_info.csv"),
}

# Load and clean data
evaluation_data = {name: load_and_clean_csv(path) for name, path in evaluation_files.items()}
# Load and clean data with 'Model' column assignment
training_data = {}
for name, path in training_files.items():
    df = load_and_clean_csv(path)
    if not df.empty:
        df['Model'] = name  # Assign the file name (key) as the model name
        training_data[name] = df

# Inspect data
print("Inspecting Training Data")
for model, df in training_data.items():
    print(f"Model: {model}, Shape: {df.shape}")
    print(df.head())

print("Inspecting Evaluation Data")
for name, df in evaluation_data.items():
    print(f"{name}: {df.shape}")
    print(df.head())

# Combine data for analysis
try:
    combined_training_data = pd.concat(training_data.values(), ignore_index=True)
    combined_evaluation_data = pd.concat(evaluation_data.values(), ignore_index=True)
    print("Combined training and evaluation data successfully.")
except Exception as e:
    print(f"Error combining data: {e}")
    log_error(f"Error combining data: {e}")

# Debug combined data
print("Columns in combined_training_data:", combined_training_data.columns)
print("Sample data in combined_training_data:\n", combined_training_data.head())
print("Columns in combined_evaluation_data:", combined_evaluation_data.columns)
print("Sample data in combined_evaluation_data:\n", combined_evaluation_data.head())

# Generate visualizations for metrics
def generate_visualizations(data, output_path, metric):
    plt.figure(figsize=(10, 6))
    for model in data['Model'].unique():
        subset = data[data['Model'] == model]
        plt.plot(subset['Iteration'], subset[metric], marker='o', label=model)
    plt.title(f'{metric} Across Iterations')
    plt.xlabel('Iteration')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()

metrics_to_visualize = ['MAE', 'MAPE', 'RMSE', 'MSE', 'R²', 'MBE']
for metric in metrics_to_visualize:
    if metric in combined_training_data.columns:
        output_path = os.path.join(reports_dir, f'{metric}_training_comparison.png')
        generate_visualizations(combined_training_data, output_path, metric)
    if metric in combined_evaluation_data.columns:
        output_path = os.path.join(reports_dir, f'{metric}_evaluation_comparison.png')
        generate_visualizations(combined_evaluation_data, output_path, metric)

# Generate insights
def generate_insights(metrics_df):
    if "Model" not in metrics_df.columns:
        print("Error: 'Model' column is missing from the DataFrame.")
        return "Insights cannot be generated. 'Model' column is missing."

    numeric_columns = metrics_df.select_dtypes(include=['number']).columns
    summary = metrics_df.groupby("Model")[numeric_columns].mean()
    insights = "**Model Performance Insights:**\n\n"
    for model in summary.index:
        insights += f"- **{model}**:\n"
        for col in summary.columns:
            insights += f"  - {col}: {summary.loc[model, col]:.2f}\n"
        insights += "\n"
    return insights

training_insights = generate_insights(combined_training_data)
evaluation_insights = generate_insights(combined_evaluation_data)

training_insights_path = os.path.join(reports_dir, 'training_insights.csv')
evaluation_insights_path = os.path.join(reports_dir, 'evaluation_insights.csv')
append_to_csv(training_insights_path, training_insights)
append_to_csv(evaluation_insights_path, evaluation_insights)

# Use OpenAI to generate AI-driven insights
def interpret_results(metrics_df):
    summary = metrics_df.groupby('Model').agg({
        'MAE': 'mean', 'MAPE': 'mean', 'RMSE': 'mean', 'MSE': 'mean', 'R²': 'mean', 'MBE': 'mean'
    }).reset_index()
    summary_text = summary.to_string(index=False)

    prompt = f"""
    The following evaluation metrics were calculated for multiple models:
    {summary_text}

    Provide a detailed discussion including:
    - Strengths and weaknesses of each model.
    - Identification of the best-performing model and justification.
    - Recommendations for model improvement and parameter adjustments.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        temperature=0.7
    )
    return response['choices'][0]['text']

ai_training_insights = interpret_results(combined_training_data)
ai_evaluation_insights = interpret_results(combined_evaluation_data)

ai_training_path = os.path.join(reports_dir, 'ai_training_insights.csv')
ai_evaluation_path = os.path.join(reports_dir, 'ai_evaluation_insights.csv')
append_to_csv(ai_training_path, ai_training_insights)
append_to_csv(ai_evaluation_path, ai_evaluation_insights)

print("All insights and AI-driven insights have been saved.")
