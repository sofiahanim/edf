import json
import boto3
from datetime import datetime, timedelta
from collections import deque
import logging
import http.client
import urllib.parse

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Constants
API_KEY = 'tVwhHoM25lyPkBjeqovYE2rr4PpEpVQI0yNV6Pae' 
BASE_URL = "api.eia.gov"
RESOURCE_PATH = "/v2/electricity/rto/region-data/data/"
DATABASE_NAME = 'electricitydb'
TABLE_NAME = 'hourlydemand'

# AWS Timestream client initialization
client = boto3.client('timestream-write', region_name='us-east-1')

def fetch_data(start_date, end_date):
    conn = http.client.HTTPSConnection(BASE_URL)
    params = {
        'api_key': API_KEY,
        'frequency': 'hourly',
        'data[0]': 'value',
        'facets[respondent][]': 'CAL',
        'facets[type][]': 'D',
        'start': start_date.strftime("%Y-%m-%dT%H"),
        'end': end_date.strftime("%Y-%m-%dT%H"),
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'length': 24  # Only 24 hours in a day
    }
    query_string = urllib.parse.urlencode(params)
    conn.request("GET", RESOURCE_PATH + "?" + query_string)
    response = conn.getresponse()
    if response.status != 200:
        logger.error(f"Failed to fetch data: {response.status}")
        logger.error(f"Response: {response.read().decode()}")
        return []
    data = json.loads(response.read().decode()).get('response', {}).get('data', [])
    conn.close()
    return data

def process_data(data):
    """Process data to handle nulls or zeros."""
    recent_values = deque(maxlen=3)
    processed_records = []
    for record in data:
        value = float(record.get('value', 0))  # Convert value to float and default to 0 if not present
        if value == 0 and recent_values:
            value = sum(recent_values) / len(recent_values)  # Average of last 3 non-zero values
        processed_records.append({
            'Dimensions': [{'Name': 'period', 'Value': record['period']}],
            'MeasureName': 'electricity_demand',
            'MeasureValue': str(value),
            'MeasureValueType': 'DOUBLE',
            'Time': str(int(datetime.strptime(record['period'], '%Y-%m-%dT%H').timestamp() * 1000))
        })
        recent_values.append(value)  # Append non-zero values
    return processed_records

def lambda_handler(event, context):
    yesterday = datetime.utcnow() - timedelta(days=1)
    start_date = datetime(yesterday.year, yesterday.month, yesterday.day)
    end_date = start_date + timedelta(days=1)
    records = fetch_data(start_date, end_date)
    processed_records = process_data(records)
    
    try:
        if processed_records:
            result = client.write_records(
                DatabaseName=DATABASE_NAME,
                TableName=TABLE_NAME,
                Records=processed_records
            )
            logger.info(f"Success: Inserted {len(processed_records)} records into Timestream.")
            for record in processed_records:
                logger.info(f"Inserted: Period {record['Dimensions'][0]['Value']} - Value {record['MeasureValue']}")
        else:
            logger.info("No complete data for the period. Inserting available data.")
    except Exception as e:
        logger.error(f"Failed to write records: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Failed to write records: {str(e)}")
        }
    return {
        'statusCode': 200,
        'body': json.dumps("Success")
    }

# Test the function locally by calling
if __name__ == "__main__":
    lambda_handler(None, None)

