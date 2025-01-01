from flask import Flask, render_template, jsonify, request
import boto3
import os
import logging
from dotenv import load_dotenv
from flask_caching import Cache

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Caching configuration
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Redshift configuration
REDSHIFT_WORKGROUP = os.getenv('REDSHIFT_WORKGROUP')
REDSHIFT_DB = os.getenv('REDSHIFT_DB')
REDSHIFT_REGION = os.getenv('REDSHIFT_REGION', 'us-east-1')

# Initialize Redshift Data API client
client = boto3.client('redshift-data', region_name=REDSHIFT_REGION)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def execute_redshift_query(sql):
    """Executes a SQL query using Redshift Data API."""
    try:
        response = client.execute_statement(
            Database=REDSHIFT_DB,
            Sql=sql,
            WorkgroupName=REDSHIFT_WORKGROUP
        )
        query_id = response['Id']

        # Wait for the query to complete
        while True:
            status = client.describe_statement(Id=query_id)['Status']
            if status in ['FINISHED', 'FAILED', 'ABORTED']:
                break

        if status == 'FINISHED':
            result = client.get_statement_result(Id=query_id)
            return result
        else:
            logger.error(f"Query failed: {status}")
            return None
    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        return None

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')
@app.route('/hourlydemand', methods=['GET'])
@cache.cached(timeout=300, query_string=True)  # Cache responses for 5 minutes based on query parameters
def hourly_demand():
    """
    Fetch Hourly Demand data with caching and dynamic total count.
    Supports server-side pagination for DataTables.
    """
    # Pagination parameters from DataTables
    start = int(request.args.get('start', 0))  # Start row
    length = int(request.args.get('length', 10))  # Number of rows per page

    # Query for total row count (cached separately)
    total_count_query = 'SELECT COUNT(*) FROM "public"."2019"'
    total_count = cache.get('total_count')
    if total_count is None:
        total_count_result = execute_redshift_query(total_count_query)
        total_count = (
            int(total_count_result['Records'][0][0]['longValue']) if total_count_result else 0
        )
        cache.set('total_count', total_count, timeout=300)  # Cache total count for 5 minutes

    # Query for paginated data
    sql = f'''
        SELECT "time", "value"
        FROM "public"."2019"
        ORDER BY "time" DESC
        LIMIT {length} OFFSET {start}
    '''
    result = execute_redshift_query(sql)

    if result:
        try:
            # Parse result into JSON-compatible format
            data = [
                {
                    col['label']: (
                        row.get('stringValue') or
                        row.get('longValue') or
                        row.get('doubleValue') or None
                    )
                    for col, row in zip(result['ColumnMetadata'], record)
                }
                for record in result['Records']
            ]
            return jsonify({
                "recordsTotal": total_count,
                "recordsFiltered": total_count,
                "data": data
            })
        except Exception as e:
            logger.error(f"Error processing query result: {e}")
            return jsonify({"error": "Failed to process query result."}), 500
    else:
        return jsonify({"error": "Failed to fetch data from Redshift."}), 500


from holidays import UnitedStates

@app.route('/holiday', methods=['GET'])
@cache.cached(timeout=300, query_string=True)  # Cache data for 5 minutes
def holiday():
    """
    Fetch holiday data for California, USA (years 2019–2024) using the holidays library.
    """
    try:
        # Fetch holidays for the United States
        us_holidays = UnitedStates(years=range(2019, 2025), state='CA')

        # Process the holiday data
        data = [
            {"date": str(holiday_date), "name": holiday_name}
            for holiday_date, holiday_name in us_holidays.items()
        ]

        # Sort data by date for consistency
        data.sort(key=lambda x: x['date'])

        return jsonify({'data': data})
    except Exception as e:
        logger.error(f"Error fetching holiday data: {e}")
        return jsonify({'error': 'Failed to fetch holiday data'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
