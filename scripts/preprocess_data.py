import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define data paths
electricity_data_folder = "dataset/electricity/"  # Folder with electricity data
weather_data_folder = "dataset/weather/"  # Folder with weather data
preprocessed_data_path = "data/preprocessed_electricity_weather_2019_2025.csv"

# Preprocess data function
def preprocess_data():
    try:
        logger.info("Starting data preprocessing...")

        # List of years to load data for
        years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

        # Initialize empty list to store dataframes for all years
        all_electricity_data = []
        all_weather_data = []

        # Load the raw datasets for electricity demand and weather data for all years
        for year in years:
            electricity_file = os.path.join(electricity_data_folder, f"{year}.csv")
            weather_file = os.path.join(weather_data_folder, f"{year}.csv")

            if not os.path.exists(electricity_file):
                raise FileNotFoundError(f"Electricity demand data file not found at {electricity_file}")
            if not os.path.exists(weather_file):
                raise FileNotFoundError(f"Weather data file not found at {weather_file}")

            electricity_data = pd.read_csv(electricity_file)
            weather_data = pd.read_csv(weather_file)

            # Ensure 'time' column exists
            if 'time' not in electricity_data.columns or 'time' not in weather_data.columns:
                raise KeyError(f"Both electricity and weather data for {year} must have a 'time' column.")

            # Convert time columns to datetime
            electricity_data['time'] = pd.to_datetime(electricity_data['time'])
            weather_data['time'] = pd.to_datetime(weather_data['time'])

            # Append the yearly data to the lists
            all_electricity_data.append(electricity_data)
            all_weather_data.append(weather_data)

        # Combine all years' data
        electricity_data_combined = pd.concat(all_electricity_data, ignore_index=True)
        weather_data_combined = pd.concat(all_weather_data, ignore_index=True)

        # Merge electricity and weather data on the 'time' column
        combined_data = pd.merge(electricity_data_combined, weather_data_combined, on='time', how='inner')

        # Drop duplicates based on time column
        combined_data.drop_duplicates(subset=['time'], inplace=True)

        # Handle missing values (forward fill as an example)
        combined_data.fillna(method='ffill', inplace=True)

        # Feature engineering: Create time-based features
        combined_data['day_of_week'] = combined_data['time'].dt.dayofweek
        combined_data['month'] = combined_data['time'].dt.month
        combined_data['season'] = combined_data['month'].apply(
            lambda x: 'Winter' if x in [12, 1, 2] else
                      ('Spring' if x in [3, 4, 5] else
                       ('Summer' if x in [6, 7, 8] else 'Fall'))
        )

        # Scale numerical features like temperature and demand (if needed)
        numerical_features = ['electricity_demand', 'temperature', 'humidity']  # Update with actual column names
        scaler = StandardScaler()
        combined_data[numerical_features] = scaler.fit_transform(combined_data[numerical_features])

        # Train-Test Split: Use 2019-2023 for training and 2024-2025 for testing
        train_data = combined_data[combined_data['time'] < '2024-01-01']
        test_data = combined_data[combined_data['time'] >= '2024-01-01']

        # Save the preprocessed data
        os.makedirs(os.path.dirname(preprocessed_data_path), exist_ok=True)
        combined_data.to_csv(preprocessed_data_path, index=False)
        logger.info(f"Preprocessed combined data saved to {preprocessed_data_path}")

        # Optionally, save the train and test data separately
        train_data_path = "data/train_data.csv"
        test_data_path = "data/test_data.csv"
        train_data.to_csv(train_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)
        logger.info(f"Train data saved to {train_data_path}")
        logger.info(f"Test data saved to {test_data_path}")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise e


# Run the preprocessing script
preprocess_data()
