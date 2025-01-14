import os
import pandas as pd
from darts import TimeSeries
from darts.models import Theta
from neuralprophet import NeuralProphet
from pycaret.time_series import setup, compare_models, predict_model

# Define directories
print("Current working directory:", os.getcwd())
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
merge_dir = os.path.join(data_dir, 'merge')
evaluation_dir = os.path.join(data_dir, 'evaluation2')

# Ensure necessary directories exist
os.makedirs(evaluation_dir, exist_ok=True)
# Load dataset directly from allyears.csv
input_file = os.path.join(merge_dir, 'allyears.csv')
print(f"Reading input file: {input_file}")
data = pd.read_csv(input_file)

# Ensure 'hour' is zero-padded and create a datetime column
data['hour'] = data['hour'].astype(str).str.zfill(2)  # Ensure hour format is HH
data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['hour'] + ':00:00')

# Select required columns for modeling
data = data.rename(columns={'datetime': 'ds', 'electric': 'y'})
selected_columns = ['ds', 'y', 'temp', 'feelslike', 'humidity', 'windspeed', 
                    'cloudcover', 'solaradiation', 'precip', 'is_holiday']
data = data[selected_columns]

# Remove duplicates and handle missing data
data = data.drop_duplicates(subset='ds').set_index('ds').asfreq('H')  # Ensure hourly frequency
data['y'] = data['y'].interpolate()  # Handle missing values with interpolation
data.reset_index(inplace=True)

# Verify dataset is ready
print("Data preview:")
print(data.head())


# Forecasting with Darts Theta Model
try:
    print("Running Darts with Theta Model...")
    series = TimeSeries.from_dataframe(data[['ds', 'y']], time_col="ds", value_cols="y", freq="H")
    model_darts_theta = Theta()
    model_darts_theta.fit(series)
    forecast_darts_theta = model_darts_theta.predict(14 * 24)  # Predict next 14 days (hourly)

    darts_results_file = os.path.join(evaluation_dir, 'darts_theta_forecast.csv')
    forecast_darts_theta.pd_series().to_csv(darts_results_file, header=True)
    print("Darts Theta forecast completed.")
except Exception as e:
    print(f"Darts Theta error: {e}")

# Forecasting with NeuralProphet
try:
    print("Running NeuralProphet...")
    neuralprophet_data = data[['ds', 'y']]
    model_np = NeuralProphet(batch_size=32, epochs=30)
    model_np.fit(neuralprophet_data, freq="H")
    future = model_np.make_future_dataframe(neuralprophet_data, periods=14 * 24)
    forecast_np = model_np.predict(future)

    neuralprophet_results_file = os.path.join(evaluation_dir, 'neuralprophet_forecast.csv')
    forecast_np.to_csv(neuralprophet_results_file, index=False)
    print("NeuralProphet forecast completed.")
except Exception as e:
    print(f"NeuralProphet error: {e}")

# Forecasting with PyCaret
try:
    print("Running PyCaret...")
    setup(data=data, target='y', session_id=123, fold=2, fh=48)
    best_model = compare_models()
    future_pycaret = predict_model(best_model, fh=14 * 24)

    pycaret_results_file = os.path.join(evaluation_dir, 'pycaret_forecast.csv')
    if isinstance(future_pycaret, pd.DataFrame):
        future_pycaret.to_csv(pycaret_results_file, index=False)
    print("PyCaret forecast completed.")
except Exception as e:
    print(f"PyCaret error: {e}")

# Summary of results
try:
    print("Generating summary of results...")
    summary_file = os.path.join(evaluation_dir, 'summary_forecast.csv')
    predict_for = pd.date_range(start=data['ds'].max(), periods=14 * 24, freq='H')

 # Add forecasts to the summary
    if 'forecast_darts_theta' in locals():
        results['Darts Theta'] = forecast_darts_theta.pd_series().values
    if 'forecast_np' in locals():
        results['NeuralProphet'] = forecast_np['yhat1'].values[:len(predict_for)]
    if 'future_pycaret' in locals():
        results['PyCaret'] = future_pycaret['Label'].values[:len(predict_for)]
        
    results = pd.DataFrame({
        'predict_at': pd.Timestamp.now(),
        'predict_for': predict_for,
        'Darts Theta': forecast_darts_theta.pd_series().values if 'forecast_darts_theta' in locals() else None,
        'NeuralProphet': forecast_np['yhat1'].values[:len(predict_for)] if 'forecast_np' in locals() else None,
        'PyCaret': future_pycaret['Label'].values[:len(predict_for)] if 'future_pycaret' in locals() else None
    })

    results.to_csv(summary_file, index=False)
    print(f"Summary results saved to {summary_file} with {len(results)} rows.")
except Exception as e:
    print(f"Error saving summary results: {e}")