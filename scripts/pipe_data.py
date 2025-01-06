import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import boto3
from botocore.config import Config

# Setup logging
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root directory
DATA_DIR = os.path.join(BASE_DIR, 'data', 'demand')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

csv_file_path = os.path.join(DATA_DIR, '2025.csv')
log_file_name = f"data_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=os.path.join(LOG_DIR, log_file_name),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# AWS Redshift setup
REDSHIFT_REGION = "us-east-1"
DATABASE_NAME = "hourlydemanddb"
TABLE_NAME = "2025"
client = boto3.client("redshift-data", region_name=REDSHIFT_REGION, config=Config(retries={'max_attempts': 10, 'mode': 'adaptive'}))

# Function to fetch the available date range in the database
def get_available_date_range():
    sql = f"SELECT MIN(time), MAX(time) FROM \"{DATABASE_NAME}\".\"public\".\"{TABLE_NAME}\""
    try:
        response = client.execute_statement(Database=DATABASE_NAME, Sql=sql, WorkgroupName='edf-workgroup', WithEvent=True)
        query_id = response['Id']
        while True:
            status_response = client.describe_statement(Id=query_id)
            if status_response['Status'] in ['FINISHED', 'FAILED', 'ABORTED']:
                if status_response['Status'] == 'FINISHED':
                    result = client.get_statement_result(Id=query_id)
                    min_date = result['Records'][0][0]['stringValue'] if result['Records'][0][0] else None
                    max_date = result['Records'][0][1]['stringValue'] if result['Records'][0][1] else None
                    return min_date, max_date
                else:
                    logger.error("Failed to fetch date range from the database.")
                    return None, None
    except Exception as e:
        logger.error(f"Error fetching date range: {e}")
        return None, None

def pipe_value(field):
    if 'stringValue' in field:
        return field['stringValue']
    elif 'longValue' in field:
        return field['longValue']
    elif 'doubleValue' in field:
        return field['doubleValue']
    else:
        return None  # Handle NULL values

# Load and clean CSV
if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
    data_df = pd.read_csv(csv_file_path, parse_dates=['time'])
    data_df.drop_duplicates(subset='time', inplace=True)  # Remove duplicates
    data_df.dropna(subset=['time', 'value'], inplace=True)  # Remove rows with missing data
    data_df.sort_values(by='time', ascending=False, inplace=True)  # Sort by time (latest on top)

    if not data_df.empty:
        latest_timestamp = data_df.iloc[0]['time']
        start_date = latest_timestamp + timedelta(hours=1)
        logger.info(f"Start date set to {start_date} based on the latest timestamp from the CSV.")
    else:
        logger.info("CSV contains no valid data. Skipping further processing.")
        start_date = datetime(datetime.now().year, 1, 1)  # Default to January 1 if CSV has no valid data
else:
    logger.info("CSV file does not exist or is empty. Skipping further processing.")
    start_date = datetime(datetime.now().year, 1, 1)  # Default to January 1 if CSV is missing

# Set initial end_date
initial_end_date = (datetime.now() - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)

# Fetch available dates from the database
min_available_date, max_available_date = get_available_date_range()

# Adjust start_date and end_date based on availability
if min_available_date and max_available_date:
    end_date = min(initial_end_date, datetime.strptime(max_available_date, '%Y-%m-%d %H:%M:%S'))
    if start_date > end_date:
        logger.warning("Start date is later than end date. Skipping query execution.")
    else:
        logger.info(f"Adjusted start_date: {start_date}, end_date: {end_date}")
        # Construct and execute the SQL query
        sql_query = f"""
        SELECT "time", "value"
        FROM "{DATABASE_NAME}"."public"."{TABLE_NAME}"
        WHERE "time" BETWEEN '{start_date.strftime("%Y-%m-%d %H:%M:%S")}' AND '{end_date.strftime("%Y-%m-%d %H:%M:%S")}'
        """
        logger.info(f"SQL Query: {sql_query}")
        try:
            response = client.execute_statement(
                Database=DATABASE_NAME, Sql=sql_query, WorkgroupName='edf-workgroup', WithEvent=True
            )
            query_id = response['Id']
            logger.info(f"Query submitted, ID: {query_id}")

            # Poll for query completion
            status = 'SUBMITTED'
            while status in ['SUBMITTED', 'STARTED', 'RUNNING']:
                description = client.describe_statement(Id=query_id)
                status = description['Status']
                if status in ['FAILED', 'ABORTED']:
                    logger.error(f"Query failed: {description.get('ErrorMessage', 'No error message provided')}")
                    break
                elif status == 'FINISHED':
                    results = client.get_statement_result(Id=query_id)
                    records = results['Records']
                    logger.info(f"Total rows fetched: {len(records)}")
                    if records:
                        new_data = pd.DataFrame([{
                            'time': pipe_value(record[0]),
                            'value': pipe_value(record[1])
                        } for record in records])
                        new_data['time'] = pd.to_datetime(new_data['time'])  # Ensure time is datetime
                        updated_data = pd.concat([data_df, new_data]).drop_duplicates(subset='time').sort_values(by='time')
                        updated_data.to_csv(csv_file_path, index=False)
                        logger.info("Data successfully appended to CSV.")
        except Exception as e:
            logger.error(f"Error executing query: {e}")
else:
    logger.warning("Database has no valid date range. Skipping further processing.")

logger.info("Data completed.")
