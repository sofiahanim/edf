
import requests
import boto3
from datetime import datetime, timedelta
from collections import deque

# Constants
API_KEY = 'tVwhHoM25lyPkBjeqovYE2rr4PpEpVQI0yNV6Pae'  # Replace with your actual API key
BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
DATABASE_NAME = 'electricitydb'
TABLE_NAME = 'hourlydemand'

# AWS Timestream client initialization
client = boto3.client('timestream-write', region_name='us-east-1')  # Add your AWS region

def fetch_data(start_date, end_date):
    all_data = []
    offset = 0
    while True:
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
            'offset': offset,
            'length': 5000
        }
        response = requests.get(BASE_URL, params=params)
        if response.status_code != 200:
            print("Failed to fetch data:", response.status_code)
            print("Response:", response.text)
            break
        data = response.json()['response']
        all_data.extend(data['data'])
        if len(data['data']) < 5000:
            break
        offset += 5000
    return all_data

def write_to_timestream(records):
   
    if not records:
        return
    try:
        result = client.write_records(
            DatabaseName=DATABASE_NAME,
            TableName=TABLE_NAME,
            Records=records
        )
        print(f'Success: Inserted {len(records)} records into Timestream.')
    except Exception as e:
        print("Failed to write records:", e)

def main():
    start_date = datetime(2019, 1, 1)
    end_date = datetime.now()
    recent_values = deque(maxlen=3)  # Store up to 3 recent non-null values for averaging

    while start_date < end_date:
        next_date = start_date + timedelta(days=1)  # Daily increments
        records = fetch_data(start_date, next_date)
        timestream_records = []
        for record in records:
            # Correct datetime parsing to match 'YYYY-MM-DDTHH'
            timestamp = int(datetime.strptime(record['period'], '%Y-%m-%dT%H').timestamp() * 1000)  # Convert to milliseconds
            value = record.get('value')
            if value is None:
                value = sum(recent_values) / len(recent_values) if recent_values else '0'
            else:
                recent_values.append(float(value))  # Update the deque with new non-null value

            timestream_records.append({
                'Dimensions': [{'Name': 'period', 'Value': record['period']}],
                'MeasureName': 'electricity_demand',
                'MeasureValue': str(value),
                'MeasureValueType': 'DOUBLE',
                'Time': str(timestamp)
            })
        write_to_timestream(timestream_records)
        start_date = next_date  # Move to the next day

if __name__ == '__main__':
    main()