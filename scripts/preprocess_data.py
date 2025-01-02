import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define data paths
raw_data_path = "dataset/electricity/2025.csv"  # Update as needed
preprocessed_data_path = "data/preprocessed_electricity_2025.csv"

# Preprocess data function
def preprocess_data():
    try:
        logger.info("Starting data preprocessing...")

        # Load the raw dataset
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")
        data = pd.read_csv(raw_data_path)

        # Example preprocessing steps
        # 1. Drop duplicates
        data.drop_duplicates(inplace=True)

        # 2. Handle missing values
        data.fillna(method='ffill', inplace=True)  # Forward fill as an example

        # 3. Convert datetime if applicable
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'])

        # 4. Add more preprocessing steps as needed...

        # Save preprocessed data
        os.makedirs(os.path.dirname(preprocessed_data_path), exist_ok=True)
        data.to_csv(preprocessed_data_path, index=False)
        logger.info(f"Preprocessed data saved to {preprocessed_data_path}")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise e


if __name__ == "__main__":
    preprocess_data()
