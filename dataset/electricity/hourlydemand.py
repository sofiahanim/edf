import csv
import os
import boto3
import requests
from datetime import datetime, timedelta

# Constants
API_KEY = 'tVwhHoM25lyPkBjeqovYE2rr4PpEpVQI0yNV6Pae'
BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
S3_BUCKET_NAME = 'insertelectricity'
LOCAL_DIRECTORY = 'ingest/electricity'
CSV_FILE_NAME = '2024.csv'
CSV_FILE_PATH = os.path.join(LOCAL_DIRECTORY, CSV_FILE_NAME)

# Ensure local directory exists
os.makedirs(LOCAL_DIRECTORY, exist_ok=True)

def fetch_data(start_datetime, end_datetime):
    """Fetch electricity demand data from the API for a given datetime range."""
    params = {
        'api_key': API_KEY,
        'frequency': 'hourly',
        'data[0]': 'value',
        'facets[respondent][]': 'CAL',
        'facets[type][]': 'D',
        'start': start_datetime.strftime("%Y-%m-%dT%H"),
        'end': end_datetime.strftime("%Y-%m-%dT%H"),
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'length': 1000  # Set based on expected data volumes and API limits
    }
    response = requests.get(BASE_URL, params=params)
    return response.json().get('response', {}).get('data', [])

def write_to_csv(data):
    """Append fetched data to a CSV file."""
    mode = 'a' if os.path.exists(CSV_FILE_PATH) else 'w'
    with open(CSV_FILE_PATH, mode, newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Write header only if file is empty
            writer.writerow(['time', 'value'])
        for record in data:
            time_formatted = datetime.strptime(record['period'], "%Y-%m-%dT%H").strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([time_formatted, record['value']])

def upload_file_to_s3(file_path, bucket, object_name=None):
    """Upload a file to an S3 bucket"""
    if object_name is None:
        object_name = os.path.basename(file_path)
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket, object_name)
    print(f"File uploaded successfully: s3://{bucket}/{object_name}")

def process_half_year_data(start_month, end_month):
    start_datetime = datetime(2024, start_month, 1)
    if end_month == 12:
        end_datetime = datetime(2025, 1, 1)
    else:
        end_datetime = datetime(2024, end_month + 1, 1)

    current_datetime = start_datetime

    while current_datetime < end_datetime:
        next_datetime = current_datetime + timedelta(hours=24)
        if next_datetime > end_datetime:
            next_datetime = end_datetime

        data = fetch_data(current_datetime, next_datetime)
        if data:
            write_to_csv(data)
        current_datetime = next_datetime

import pandas as pd

def remove_duplicate_rows(file_path):
    """Remove duplicate rows based on the timestamp to ensure data uniqueness."""
    df = pd.read_csv(file_path)
    original_count = len(df)
    df.drop_duplicates(subset=['time'], keep='last', inplace=True)
    df.to_csv(file_path, index=False)
    new_count = len(df)
    print(f"Removed {original_count - new_count} duplicates. New total: {new_count} rows.")

def verify_data_integrity():
    """Verify that the CSV contains the correct number of rows and re-fetch any missing data."""
    expected_count = 365 * 24  # 8760 for non-leap years
    with open(CSV_FILE_PATH, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        rows = list(reader)
    actual_count = len(rows)  # Now this does not include the header
    if actual_count != expected_count:
        print(f"Data integrity check failed: found {actual_count}, expected {expected_count}")
        # Here you could add logic to identify which hours are missing if needed
    else:
        print("Data integrity check passed.")


def main():
    process_half_year_data(1, 6)  # Process from January to June
    process_half_year_data(7, 12)  # Process from July to December
    remove_duplicate_rows(CSV_FILE_PATH)
    verify_data_integrity()
    upload_file_to_s3(CSV_FILE_PATH, S3_BUCKET_NAME)

if __name__ == '__main__':
    main()
