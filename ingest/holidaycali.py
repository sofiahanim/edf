import boto3
import holidays
from datetime import datetime, date, time

# Constants for the Timestream database and table
DATABASE_NAME = 'holidaysdb'
TABLE_NAME = 'holidaycali'

# Function to fetch holidays for California
def fetch_california_holidays(year):
    ca_holidays = holidays.US(state='CA', years=year)
    return [{'date': date, 'name': name} for date, name in sorted(ca_holidays.items())]

# Function to prepare and display holiday data
def prepare_holiday_data(holiday_data):
    records = []
    for holiday in holiday_data:
        # Ensure the date is a datetime object with time
        holiday_datetime = datetime.combine(holiday['date'], time())  # Adds midnight time
        timestamp = str(int(holiday_datetime.timestamp() * 1000))  # Convert to milliseconds
        
        record = {
            'Dimensions': [
                {'Name': 'holiday_name', 'Value': holiday['name'], 'DimensionValueType': 'VARCHAR'},
            ],
            'MeasureName': 'holiday',
            'MeasureValue': '1',  # A simple value to indicate the presence of a holiday
            'MeasureValueType': 'DOUBLE',
            'Time': timestamp,
            'TimeUnit': 'MILLISECONDS'
        }
        records.append(record)
    return records

# Main function to manage the flow
def main():
    year = datetime.now().year  # Fetch current year's holidays or specify any year
    holidays_data = fetch_california_holidays(year)
    records_to_insert = prepare_holiday_data(holidays_data)
    for record in records_to_insert:
        print(record)

# Run the main function
if __name__ == "__main__":
    main()
