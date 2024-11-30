import boto3
import holidays
from datetime import datetime, date, time, timedelta

# Initialize the Timestream write client
timestream_write = boto3.client('timestream-write')

# Constants for the Timestream database and table
DATABASE_NAME = 'holidaysdb'
TABLE_NAME = 'holidaycali'

# Function to fetch holidays for California
def fetch_california_holidays(year):
    ca_holidays = holidays.US(state='CA', years=year)
    return [{'date': date, 'name': name} for date, name in sorted(ca_holidays.items())]

# Function to prepare and insert holiday data into Timestream
def insert_holidays_into_timestream(holiday_data):
    records = []
    for holiday in holiday_data:
        adjusted_date = adjust_timestamp_to_window(datetime.combine(holiday['date'], time.min))
        record = {
            'Dimensions': [
                {'Name': 'holiday_name', 'Value': holiday['name'], 'DimensionValueType': 'VARCHAR'},
                {'Name': 'date', 'Value': holiday['date'].isoformat(), 'DimensionValueType': 'VARCHAR'}
            ],
            'MeasureName': 'holiday',
            'MeasureValue': '1',  # A simple value to indicate the presence of a holiday
            'MeasureValueType': 'DOUBLE',
            'Time': str(int(adjusted_date.timestamp() * 1000)),  # Adjusted Timestamp of holiday at 12 AM in milliseconds
            'TimeUnit': 'MILLISECONDS'
        }
        records.append(record)
        
        # Write records to Timestream in batches of 100 (or as needed)
        if len(records) >= 100:
            write_to_timestream(records)
            records = []
    
    # Write any remaining records
    if records:
        write_to_timestream(records)

def write_to_timestream(records):
    try:
        result = timestream_write.write_records(
            DatabaseName=DATABASE_NAME,
            TableName=TABLE_NAME,
            Records=records
        )
        print("Write records status:", result['ResponseMetadata']['HTTPStatusCode'])
    except timestream_write.exceptions.RejectedRecordsException as e:
        print("Rejected Records:", e.response['Error']['Message'])
        if 'RejectedRecords' in e.response:
            for record in e.response['RejectedRecords']:
                print("Rejected Record Reason:", record['Reason'])
    except Exception as e:
        print("Failed to write records to Timestream:", str(e))

def adjust_timestamp_to_window(date):
    # Example date range end (you need to adjust this based on your Timestream table configuration)
    range_end = datetime.now()  # Assumes the range end is today's date for example
    if date > range_end:
        # Adjust the year of the date to fit within the range
        return datetime(range_end.year, date.month, date.day, date.hour, date.minute, date.second)
    return date

# Main function to manage the flow
def main():
    year = datetime.now().year  # Fetch current year's holidays or specify any year
    holidays_data = fetch_california_holidays(year)
    insert_holidays_into_timestream(holidays_data)

# Run the main function
if __name__ == "__main__":
    main()
