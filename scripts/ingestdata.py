import os
import logging
import boto3
import pandas as pd
from datetime import datetime, timedelta

# Setup logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

# Load environment variables
REDSHIFT_REGION = os.getenv("REDSHIFT_REGION", "us-east-1")
REDSHIFT_WORKGROUP = os.getenv("REDSHIFT_WORKGROUP")

# Initialize Redshift Data API client
try:
    client = boto3.client("redshift-data", region_name=REDSHIFT_REGION)
    logger.info("Successfully initialized Redshift Data API client.")
except Exception as e:
    logger.error(f"Error initializing Redshift Data API client: {e}")
    raise e

# Function to execute Redshift query
def execute_redshift_query(client, sql, workgroup_name):
    try:
        logger.info(f"Executing query on workgroup '{workgroup_name}': {sql}")
        response = client.execute_statement(WorkgroupName=workgroup_name, Sql=sql, WithEvent=True)
        query_id = response["Id"]

        # Wait for the query to complete
        while True:
            status_response = client.describe_statement(Id=query_id)
            status = status_response["Status"]
            logger.info(f"Query status for {query_id}: {status}")
            if status in ["FINISHED", "FAILED", "ABORTED"]:
                break

        if status == "FINISHED":
            result = client.get_statement_result(Id=query_id)
            logger.info("Query completed successfully.")
            return result["Records"]
        else:
            error_message = status_response.get("Error", "No error details provided.")
            logger.error(f"Query failed: {error_message}")
            return None
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return None

# Function to append data to CSV
def append_to_csv(file_path, data, columns):
    try:
        # Convert Redshift results to DataFrame
        rows = []
        for record in data:
            row = [col.get("stringValue") or col.get("longValue") or col.get("doubleValue") for col in record]
            rows.append(row)
        new_data = pd.DataFrame(rows, columns=columns)

        # Append to CSV
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            updated_data = new_data
        updated_data.to_csv(file_path, index=False)
        logger.info(f"Data successfully appended to {file_path}.")
    except Exception as e:
        logger.error(f"Error appending data to {file_path}: {e}")

# SQL Queries
sql_demand = 'SELECT * FROM "hourlydemanddb"."public"."2025";'
sql_weather = 'SELECT * FROM "hourlyweatherdb"."public"."2025";'

# Dataset file paths
demand_file = "dataset/electricity/2025.csv"
weather_file = "dataset/weather/2025.csv"

# Ensure dataset folders exist
os.makedirs(os.path.dirname(demand_file), exist_ok=True)
os.makedirs(os.path.dirname(weather_file), exist_ok=True)

# Fetch and append data
logger.info("Starting data ingestion...")

demand_data = execute_redshift_query(client, sql_demand, REDSHIFT_WORKGROUP)
if demand_data:
    append_to_csv(demand_file, demand_data, ["time", "value"])
else:
    logger.warning("No data retrieved for 'hourlydemanddb'.")

weather_data = execute_redshift_query(client, sql_weather, REDSHIFT_WORKGROUP)
if weather_data:
    append_to_csv(weather_file, weather_data, ["datetime", "temp", "feelslike", "humidity", "windspeed"])
else:
    logger.warning("No data retrieved for 'hourlyweatherdb'.")

logger.info("Data ingestion completed.")
