import pandas as pd
import boto3
import logging
from datetime import datetime
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATABASE_NAME = 'weatherdb'
TABLE_NAME = 'hourlyweather'

# AWS Timestream client initialization
client = boto3.client('timestream-write', region_name='us-east-1')

def write_records(df):
    records = []
    for _, row in df.iterrows():
        timestamp = str(int(row['datetime'].timestamp() * 1000))  # Timestamp in milliseconds
        for column in df.columns:
            if column != 'datetime':  # Skip datetime since it's used as the timestamp
                measure_type = 'DOUBLE' if pd.api.types.is_numeric_dtype(df[column]) else 'VARCHAR'
                record_value = str(row[column])
                if measure_type == 'VARCHAR':
                    record_value = record_value[:100]  # Ensure VARCHAR fields do not exceed 100 characters
                records.append({
                    'Dimensions': [{'Name': 'period', 'Value': 'general'}],
                    'MeasureName': column,
                    'MeasureValue': record_value,
                    'MeasureValueType': measure_type,
                    'Time': timestamp
                })
        # Send records in batches of 100 to avoid hitting the limit
        if len(records) >= 100:
            try:
                client.write_records(DatabaseName=DATABASE_NAME, TableName=TABLE_NAME, Records=records)
                records = []  # Clear records after writing
                logging.info(f"Successfully wrote 100 records to Timestream.")
            except Exception as e:
                logging.error(f"Failed to write records: {str(e)}")
                break  # Exit the loop on failure to avoid further errors

    if records:  # Write any remaining records
        try:
            client.write_records(DatabaseName=DATABASE_NAME, TableName=TABLE_NAME, Records=records)
            logging.info(f"Successfully wrote remaining {len(records)} records to Timestream.")
        except Exception as e:
            logging.error(f"Failed to write records: {str(e)}")
"""
def write_records(df):
    records = []
    for _, row in df.iterrows():
        timestamp = str(int(row['datetime'].timestamp() * 1000))  # Timestamp in milliseconds
        for column in df.columns:
            if column != 'datetime':  # Skip datetime since it's used as the timestamp
                measure_type = 'DOUBLE' if pd.api.types.is_numeric_dtype(df[column]) else 'VARCHAR'
                record_value = str(row[column])[:100]  # Truncate values to 100 characters to avoid overflow
                records.append({
                    'Dimensions': [{'Name': 'period', 'Value': 'general'}],  # Adjust as necessary
                    'MeasureName': column,
                    'MeasureValue': record_value,
                    'MeasureValueType': measure_type,
                    'Time': timestamp
                })
        # Send records in batches of 100 to avoid hitting the limit
        if len(records) >= 100:
            try:
                client.write_records(DatabaseName=DATABASE_NAME, TableName=TABLE_NAME, Records=records)
                records = []  # Clear records after writing
                logging.info(f"Successfully wrote 100 records to Timestream.")
            except Exception as e:
                logging.error(f"Failed to write records: {str(e)}")
                return

    if records:  # Write any remaining records
        try:
            client.write_records(DatabaseName=DATABASE_NAME, TableName=TABLE_NAME, Records=records)
            logging.info(f"Successfully wrote remaining {len(records)} records to Timestream.")
        except Exception as e:
            logging.error(f"Failed to write records: {str(e)}")
"""
def main():
    df = pd.read_csv('C:/Users/hanim/edf/ingest/cleaned_weather_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()  # Dynamically selecting numeric columns
    df[numeric_columns] = df[numeric_columns].ffill().bfill()  # Forward and backward fill
    categorical_columns = ['preciptype', 'conditions', 'icon']  # Assume these are the only categorical columns
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')
    write_records(df)

if __name__ == "__main__":
    main()
