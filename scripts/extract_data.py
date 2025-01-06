import os
import logging
import boto3
import pandas as pd
from datetime import datetime
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor

# Setup logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_name = f"data_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=os.path.join(LOG_DIR, log_file_name),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants for AWS Redshift access
REDSHIFT_REGION = "us-east-1"
REDSHIFT_ROLE_ARN = "arn:aws:iam::022499009488:role/service-role/AmazonRedshift-CommandsAccessRole-20241212T083818"
DB_USER = 'IAM:RootIdentity'

# Initialize Redshift Data API client with retry strategy
retry_config = Config(
    retries={
        'max_attempts': 10,
        'mode': 'adaptive'
    }
)
try:
    client = boto3.client("redshift-data", region_name=REDSHIFT_REGION, config=retry_config)
    logger.info("Successfully initialized Redshift Data API client.")
except Exception as e:
    logger.error(f"Error initializing Redshift Data API client: {e}")
    raise

def execute_redshift_query(client, database, sql, workgroup_name, is_synchronous=True):
    logger.info(f"Executing SQL on database '{database}': {sql}")
    try:
        response = client.execute_statement(
            Database=database,
            WorkgroupName=workgroup_name,
            Sql=sql,
            WithEvent=is_synchronous,
            DbUser=DB_USER
        )
        query_id = response['Id']
        logger.info("SQL execution initiated, awaiting results...")
        return query_id
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return None

def monitor_query(client, query_id):
    while True:
        response = client.describe_statement(Id=query_id)
        status = response['Status']
        if status in ['FINISHED', 'FAILED', 'ABORTED']:
            logger.info(f"Query {query_id} status: {status}")
            break
    if status == 'FINISHED':
        result = client.get_statement_result(Id=query_id)
        return result['Records']
    else:
        logger.error(f"Query {query_id} failed or aborted.")
        return None

def append_to_csv(file_path, data, columns):
    try:
        rows = [dict(zip(columns, [col.get("stringValue", col.get("longValue", col.get("doubleValue"))) for col in record])) for record in data]
        new_data = pd.DataFrame(rows)
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            updated_data = new_data
        updated_data.to_csv(file_path, index=False)
        logger.info(f"Data successfully appended to {file_path}.")
    except Exception as e:
        logger.error(f"Error appending data to {file_path}: {e}")

# Define SQL queries, file paths, and database names
sql_demand = 'SELECT * FROM "hourlydemanddb"."public"."2025";'
sql_weather = 'SELECT * FROM "hourlyweatherdb"."public"."2025";'
DEMAND_DB_NAME = "hourlydemanddb"
WEATHER_DB_NAME = "hourlyweatherdb"
demand_file = "dataset/electricity/2025.csv"
weather_file = "dataset/weather/2025.csv"

# Ensure dataset folders exist
os.makedirs(os.path.dirname(demand_file), exist_ok=True)
os.makedirs(os.path.dirname(weather_file), exist_ok=True)

# Workgroup name
WORKGROUP_NAME = "edf-workgroup"

# Use ThreadPoolExecutor to run queries concurrently
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(execute_redshift_query, client, DEMAND_DB_NAME, sql_demand, WORKGROUP_NAME): (demand_file, ["time", "value"]),
        executor.submit(execute_redshift_query, client, WEATHER_DB_NAME, sql_weather, WORKGROUP_NAME): (weather_file, ["datetime", "temp", "feelslike", "humidity", "windspeed"])
    }
    for future in futures:
        query_id = future.result()
        if query_id:
            records = monitor_query(client, query_id)
            if records:
                append_to_csv(futures[future][0], records, futures[future][1])

logger.info("Data extraction completed.")
