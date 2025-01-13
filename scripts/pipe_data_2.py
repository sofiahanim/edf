import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import boto3
from botocore.config import Config

# Setup logging
BASE_DIR = os.getcwd()  # Current working directory
DATA_DIR = os.path.join(BASE_DIR, 'data', 'weather')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

csv_file_path = os.path.join(DATA_DIR, '2025.csv')
log_file_name = f"data_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file_path = os.path.join(LOG_DIR, log_file_name)

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# AWS Redshift setup
REDSHIFT_REGION = "us-east-1"
WORKGROUP_NAME = "edf-workgroup"
DATABASE_NAME = "hourlyweatherdb"
TABLE_NAME = "2025"

client = boto3.client("redshift-data", region_name=REDSHIFT_REGION, config=Config(retries={'max_attempts': 10, 'mode': 'adaptive'}))

def get_available_date_range():
    sql = f"SELECT MIN(datetime), MAX(datetime) FROM \"{DATABASE_NAME}\".\"public\".\"{TABLE_NAME}\""
    try:
        response = client.execute_statement(Database=DATABASE_NAME, Sql=sql, WorkgroupName=WORKGROUP_NAME)
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

if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
    data_df = pd.read_csv(csv_file_path, parse_dates=['datetime'])
    data_df.drop_duplicates(subset='datetime', inplace=True)
    data_df.sort_values(by='datetime', ascending=False, inplace=True)
    latest_timestamp = data_df.iloc[0]['datetime'] if not data_df.empty else None
    start_date = latest_timestamp + timedelta(hours=1) if latest_timestamp else datetime(datetime.now().year, 1, 1)
else:
    data_df = pd.DataFrame()
    start_date = datetime(datetime.now().year, 1, 1)

initial_end_date = datetime.now() - timedelta(days=1)
initial_end_date = initial_end_date.replace(hour=23, minute=59, second=59)

min_available_date, max_available_date = get_available_date_range()

if min_available_date and max_available_date:
    end_date = min(initial_end_date, datetime.strptime(max_available_date, '%Y-%m-%d %H:%M:%S'))
    if start_date > end_date:
        logger.info("No new data to fetch.")
    else:
        query = f"""
        SELECT *
        FROM "{DATABASE_NAME}"."public"."{TABLE_NAME}"
        WHERE "datetime" BETWEEN '{start_date.strftime("%Y-%m-%d %H:%M:%S")}' AND '{end_date.strftime("%Y-%m-%d %H:%M:%S")}'
        """
        try:
            response = client.execute_statement(Database=DATABASE_NAME, Sql=query, WorkgroupName=WORKGROUP_NAME)
            query_id = response['Id']

            while True:
                status_response = client.describe_statement(Id=query_id)
                if status_response['Status'] == 'FINISHED':
                    results = client.get_statement_result(Id=query_id)
                    records = results['Records']
                    # Processing each record into a list of dictionaries
                    data = []
                    for record in records:
                        row = {col['name']: record[idx]['stringValue'] if 'stringValue' in record[idx] else
                                            record[idx]['longValue'] if 'longValue' in record[idx] else
                                            record[idx]['doubleValue'] if 'doubleValue' in record[idx] else None
                               for idx, col in enumerate(results['ColumnMetadata'])}
                        data.append(row)
                    new_data = pd.DataFrame(data)
                    new_data['datetime'] = pd.to_datetime(new_data['datetime'])
                    data_df = pd.concat([data_df, new_data]).drop_duplicates(subset='datetime').sort_values(by='datetime')
                    data_df.to_csv(csv_file_path, index=False)
                    logger.info("Data successfully appended to CSV.")
                    break
                elif status_response['Status'] in ['FAILED', 'ABORTED']:
                    raise Exception(f"Query failed: {status_response.get('ErrorMessage', 'No details available')}")
        except Exception as e:
            logger.error(f"Error executing query: {e}")
else:
    logger.warning("No valid data range available.")

logger.info("Pipeline execution completed.")
