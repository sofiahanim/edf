import json
import boto3
import requests
from datetime import datetime, timedelta
from collections import deque
import logging

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Constants
API_KEY = 'tVwhHoM25lyPkBjeqovYE2rr4PpEpVQI0yNV6Pae'
BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
DATABASE_NAME = 'electricitydb'
TABLE_NAME = 'hourlydemand'
ROW_LIMIT = 5000  # Assuming each request can handle up to 5000 rows

# AWS Timestream client initialization
client = boto3.client('timestream-write', region_name='us-east-1')

def fetch_data(start_date, end_date):
    full_data = []
    temp_start_date = start_date
    while temp_start_date < end_date:
        params = {
            'api_key': API_KEY,
            'frequency': 'hourly',
            'data[0]': 'value',
            'facets[respondent][]': 'CAL',
            'facets[type][]': 'D',
            'start': temp_start_date.strftime("%Y-%m-%dT%H"),
            'end': end_date.strftime("%Y-%m-%dT%H"),
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc',
            'length': ROW_LIMIT
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json().get('response', {}).get('data', [])
        full_data.extend(data)
        if len(data) < ROW_LIMIT:
            break
        last_fetched_period = data[-1]['period']
        temp_start_date = datetime.strptime(last_fetched_period, '%Y-%m-%dT%H') + timedelta(hours=1)

    return full_data

def process_data(data):
    recent_values = deque(maxlen=3)
    processed_records = []
    for record in data:
        period = record['period'][:100]  # Truncate to 100 characters to prevent overflow
        value = float(record.get('value', 0))
        measure_value = str(value)[:100]  # Truncate measure value as well

        processed_record = {
            'Dimensions': [{'Name': 'period', 'Value': period}],
            'MeasureName': 'electricity_demand',
            'MeasureValue': measure_value,
            'MeasureValueType': 'DOUBLE',
            'Time': str(int(datetime.strptime(period, '%Y-%m-%dT%H').timestamp() * 1000))
        }
        processed_records.append(processed_record)
        logger.debug(f"Processed record: {processed_record}")
        recent_values.append(value)
    return processed_records

def lambda_handler(event, context):
    start_date = datetime(2023, 6, 18, 1)  # Start from July 22, 2021, at 1 AM
    end_date = datetime.utcnow()  # End at the current UTC date and time
    total_inserted = 0

    while start_date < end_date:
        next_date = start_date + timedelta(days=1)
        records = fetch_data(start_date, next_date)
        if records:
            processed_records = process_data(records)
            try:
                result = client.write_records(
                    DatabaseName=DATABASE_NAME,
                    TableName=TABLE_NAME,
                    Records=processed_records
                )
                total_inserted += len(processed_records)
                logger.info(f"Inserted {len(processed_records)} records for period starting {start_date}. Total so far: {total_inserted}")
            except Exception as e:
                logger.error(f"Failed to write records: {e}")
                return {
                    'statusCode': 500,
                    'body': json.dumps(f"Failed to write records: {str(e)}")
                }
        else:
            logger.info(f"No data fetched for period starting {start_date}")

        start_date = next_date

    return {
        'statusCode': 200,
        'body': json.dumps(f"Successfully inserted total of {total_inserted} records.")
    }

if __name__ == "__main__":
    lambda_handler(None, None)
