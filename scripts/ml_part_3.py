import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import openai

# Set OpenAI API Key
openai.api_key = "sk-proj-VYx5HXgXhdyLvRrlSkVtY3mPLmJtwaOOhv0phy6zkj6nLpWnCDl1YEWus_KilZEXWvUzYjqt8cS6u3npgA"

# Define directories
training_dir = os.path.join('.', 'training')
evaluation_dir = os.path.join('.', 'evaluation')
report_dir = os.path.join('.', 'reports')
os.makedirs(report_dir, exist_ok=True)

# Helper: Load CSV
def load_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return None

# Helper: Generate Visualizations
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

# Helper: Generate Dynamic Insights
def generate_insights(metrics_df):
    summary = metrics_df.groupby("Model").mean()
    insights = "**Model Performance Insights:**\n\n"
    for model in summary.index:
        insights += f"- **{model}**:\n"
        insights += f"  - RMSE: {summary.loc[model, 'RMSE']:.2f}\n"
        insights += f"  - MAPE: {summary.loc[model, 'MAPE']:.2f}%\n"
        insights += f"  - R²: {summary.loc[model, 'R²']:.2f}\n"
        insights += f"  - **Summary**: This model is {'good' if summary.loc[model, 'R²'] > 0.8 else 'challenged'} for {('low RMSE' if summary.loc[model, 'RMSE'] < summary['RMSE'].mean() else 'error minimization')}.\n\n"

    overall_recommendation = f"""
    **Overall Recommendations:**
    - The model with the lowest average RMSE is **{summary['RMSE'].idxmin()}** with an RMSE of {summary['RMSE'].min():.2f}.
    - The best average R² score was achieved by **{summary['R²'].idxmax()}** with an R² of {summary['R²'].max():.2f}.
    - Average MAPE across all models is **{summary['MAPE'].mean():.2f}%**.
    """
    insights += overall_recommendation
    return insights

# Helper: AI-Generated Insights
def interpret_results(metrics_df):
    summary = metrics_df.groupby('Model').agg(
        {'MAE': 'mean', 'MAPE': 'mean', 'RMSE': 'mean', 'MSE': 'mean', 'R²': 'mean', 'MBE': 'mean'}
    ).reset_index()
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

# Load all metrics
metrics_files = [
    os.path.join(training_dir, 'h2o_gbm_metrics.csv'),
    os.path.join(training_dir, 'darts_theta_metrics.csv'),
    os.path.join(training_dir, 'prophet_metrics.csv')
]
metrics_data = pd.concat([load_csv(f) for f in metrics_files if load_csv(f) is not None], ignore_index=True)

# Generate Visualizations for Metrics
metrics_to_visualize = ['MAE', 'MAPE', 'RMSE', 'MSE', 'R²', 'MBE']
for metric in metrics_to_visualize:
    output_path = os.path.join(report_dir, f'{metric}_comparison.png')
    generate_visualizations(metrics_data, output_path, metric)

# Generate Automated Insights
automated_insights = generate_insights(metrics_data)

# Generate AI-Driven Interpretation
ai_insights = interpret_results(metrics_data)

# Save Report with Insights
report_file = os.path.join(report_dir, f'report_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.md')
with open(report_file, 'w') as f:
    f.write("# Model Evaluation Report\n\n")
    f.write("## Visualizations\n\n")
    for metric in metrics_to_visualize:
        f.write(f"![{metric} Comparison](./{metric}_comparison.png)\n\n")
    f.write("## Automated Insights\n\n")
    f.write(automated_insights + "\n\n")
    f.write("## AI-Driven Insights\n\n")
    f.write(ai_insights)

print(f"Report saved to: {report_file}")
