import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from neuralprophet import NeuralProphet
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Define directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
merge_dir = os.path.join(data_dir, 'merge')

# Load the merged dataset
input_file = os.path.join(merge_dir, 'allyears.csv')
data = pd.read_csv(input_file)

# Prepare data
data['hour'] = data['hour'].astype(str).str.zfill(2)
data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['hour'] + ':00:00')
prophet_data = data[['datetime', 'electric']].rename(columns={'datetime': 'ds', 'electric': 'y'})

# Remove duplicate timestamps for NeuralProphet
prophet_data = prophet_data.drop_duplicates(subset='ds')

# Split data for evaluation
train = prophet_data

# Prophet Forecasting
prophet_model = Prophet()
prophet_model.fit(train)
future_prophet = prophet_model.make_future_dataframe(periods=14 * 24, freq='H')
prophet_forecast = prophet_model.predict(future_prophet)

# NeuralProphet Forecasting
neuralprophet_model = NeuralProphet()
neuralprophet_model.fit(train)
future_neural = neuralprophet_model.make_future_dataframe(train, periods=14 * 24)
neural_forecast = neuralprophet_model.predict(future_neural)

# LightGBM Forecasting
features = ['temp', 'feelslike', 'humidity', 'windspeed', 'cloudcover', 'solaradiation', 'precip', 'is_holiday']
target = 'electric'

scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
X = data[features]
y = data[target]

# Prepare training set (past data only)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train LightGBM model
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)

# Create future features for LightGBM
future_features = pd.DataFrame(data.tail(14 * 24)[features], columns=features)  # Use recent trends
lgb_forecast = lgb_model.predict(future_features)

# Prepare output
future_predictions = pd.DataFrame({
    'predict_at': pd.Timestamp.now(),
    'predict_for': prophet_forecast['ds'][-14 * 24:],  # Use datetime from Prophet
    'Prophet': prophet_forecast['yhat'][-14 * 24:],
    'NeuralProphet': neural_forecast['yhat1'][-14 * 24:],
    'LightGBM': lgb_forecast
})

# Save the results
output_file = os.path.join(merge_dir, 'future_14_day_predictions.csv')
future_predictions.to_csv(output_file, index=False)

print(f"Future predictions saved to {output_file}")

# Visualization
plt.figure(figsize=(15, 8))
plt.plot(future_predictions['predict_for'], future_predictions['Prophet'], label='Prophet', linestyle='--')
plt.plot(future_predictions['predict_for'], future_predictions['NeuralProphet'], label='NeuralProphet', linestyle='-.')
plt.plot(future_predictions['predict_for'], future_predictions['LightGBM'], label='LightGBM', linestyle='-')
plt.fill_between(
    prophet_forecast['ds'][-14 * 24:],
    prophet_forecast['yhat_lower'][-14 * 24:],
    prophet_forecast['yhat_upper'][-14 * 24:],
    color='gray',
    alpha=0.3,
    label='Prophet Confidence Interval'
)
plt.xlabel('Datetime')
plt.ylabel('Electric Demand')
plt.title('Electric Demand Forecast for the Next 14 Days (Hourly)')
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance Analysis (LightGBM)
lgb_feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': lgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance (LightGBM):")
print(lgb_feature_importance)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(lgb_feature_importance['Feature'], lgb_feature_importance['Importance'], color='teal')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (LightGBM)')
plt.gca().invert_yaxis()
plt.show()
