from flask import Flask, render_template, jsonify
import boto3
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Redshift configuration
REDSHIFT_WORKGROUP = os.getenv('REDSHIFT_WORKGROUP')  # For serverless Redshift
REDSHIFT_DB = os.getenv('REDSHIFT_DB')  # Database name
REDSHIFT_REGION = os.getenv('REDSHIFT_REGION', 'us-east-1')  # Default to us-east-1

# Redshift Data API client
client = boto3.client('redshift-data', region_name=REDSHIFT_REGION)

# Verify Workgroup
try:
    response = client.list_workgroups()
    logger.info(f"Available workgroups: {response.get('Workgroups', [])}")
except Exception as e:
    logger.error(f"Failed to list workgroups: {e}")


def execute_redshift_query(sql):
    """Execute SQL query using Redshift Data API."""
    try:
        logger.info(f"Executing query: {sql}")
        response = client.execute_statement(
            Database=REDSHIFT_DB,
            Sql=sql,
            WorkgroupName=REDSHIFT_WORKGROUP,  # Serverless Redshift
            WithEvent=True
        )
        query_id = response['Id']

        # Wait for the query to complete
        while True:
            status_response = client.describe_statement(Id=query_id)
            logger.info(f"Query Status: {status_response['Status']}")
            if status_response['Status'] in ['FINISHED', 'FAILED', 'ABORTED']:
                break

        if status_response['Status'] == 'FINISHED':
            # Get query results
            result = client.get_statement_result(Id=query_id)
            return result
        else:
            logger.error(f"Query failed: {status_response.get('Error', 'No error details')}")
            return None
    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/hourlydemand')
def hourly_demand():
    """Fetch and display hourly demand data from Redshift."""
    sql = 'SELECT "time", "value" FROM "public"."2019" LIMIT 10;'
    result = execute_redshift_query(sql)

    if result:
        try:
            # Process the query results into a list of dictionaries
            data = []
            for row in result['Records']:
                row_data = {}
                for column, record in zip(result['ColumnMetadata'], row):
                    # Handle different data types
                    if 'stringValue' in record:
                        row_data[column['label']] = record['stringValue']
                    elif 'longValue' in record:
                        row_data[column['label']] = record['longValue']
                    elif 'doubleValue' in record:
                        row_data[column['label']] = record['doubleValue']
                    else:
                        row_data[column['label']] = None  # Fallback for unexpected data types
                data.append(row_data)

            return jsonify({"data": data})  # Wrap the data in a 'data' key
        except Exception as e:
            logger.error(f"Failed to process query result: {e}")
            return jsonify({"error": "Failed to process query result."}), 500
    else:
        return jsonify({"error": "Failed to fetch data from Redshift."}), 500

if __name__ == '__main__':
    # Log configuration details
    logger.info(f"Using region: {REDSHIFT_REGION}")
    logger.info(f"Using workgroup: {REDSHIFT_WORKGROUP}")
    logger.info(f"Using database: {REDSHIFT_DB}")

    app.run(debug=True, host='0.0.0.0', port=5000)
