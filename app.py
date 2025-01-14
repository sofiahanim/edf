
from flask import Flask, Response, redirect, jsonify, request, render_template, send_from_directory
import pandas as pd
import holidays
from flask_caching import Cache
from flask_cors import CORS
import logging
import os
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from pathlib import Path
from mimetypes import guess_type
import json
import re
import plotly.express as px
from serverless_wsgi import handle_request
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Initialize app and API
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['CACHE_TYPE'] = 'simple' 
cache = Cache(app)
cache.init_app(app)

CORS(app, resources={r"/*": {"origins": "*"}})

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory paths
BASE_DIR = "/var/task"
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

logger.info(f"Base Directory: {BASE_DIR}")
logger.info(f"Data Directory: {DATA_DIR}")
logger.info(f"Templates Directory: {TEMPLATES_DIR}")
logger.info(f"Static Directory: {STATIC_DIR}")
logger.info(f"Cache Directory: {CACHE_DIR}")

# Ensure required directories exist
REQUIRED_DIRS = [DATA_DIR, TEMPLATES_DIR, STATIC_DIR, CACHE_DIR]
for directory in REQUIRED_DIRS:
    if not os.path.exists(directory):
        logger.warning(f"Directory missing: {directory}")
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created missing directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}", exc_info=True)

def lambda_handler(event, context):
    """
    AWS Lambda handler to process API Gateway events with the Flask app.
    """
    try:
        # Log the incoming event and context
        logger.info("Lambda handler invoked.")
        logger.debug(f"Event: {json.dumps(event)}")
        logger.debug(f"Context: {context}")

        # Validate app.wsgi_app
        if app.wsgi_app is None:
            logger.info("Initializing DispatcherMiddleware for Flask app.")
            app.wsgi_app = DispatcherMiddleware(None, {"/": app})

        # Process the request using serverless_wsgi
        response = handle_request(app, event, context)
        logger.debug(f"Response from handle_request: {response}")

        # Handle missing keys gracefully
        response_body = response.get("body", "No body returned")
        return {
            "statusCode": response.get("statusCode", 500),
            "headers": response.get("headers", {"Access-Control-Allow-Origin": "*"}),
            "body": response_body,
            "isBase64Encoded": response.get("isBase64Encoded", False),
        }
    except Exception as e:
        logger.error("Error in Lambda handler:", exc_info=True)
        return {
            "statusCode": 500,
            "body": "Internal Server Error",
            "headers": {"Access-Control-Allow-Origin": "*"},
        }


"""0. START SECTION 0 <HELPER FUNCTIONS>"""
@app.route("/data/<file_name>")
def get_data(file_name):
    """Fetch CSV data."""
    try:
        df = load_csv(file_name)
        if df.empty:
            logger.warning(f"No data found for {file_name}")
            return jsonify({"error": f"No data found for {file_name}"}), 404
        logger.info(f"Successfully fetched data for {file_name}")
        return df.to_json(orient="records"), 200, {"Content-Type": "application/json"}
    except Exception as e:
        logger.error(f"Error fetching data for {file_name}: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch data"}), 500

@app.route("/static/<path:filename>")
def serve_static_files(filename):
    """Serve static files."""
    try:
        full_path = os.path.join(STATIC_DIR, filename)
        if os.path.exists(full_path):
            logger.info(f"Serving static file: {filename}")
            return send_from_directory(STATIC_DIR, filename)
        else:
            logger.warning(f"Static file not found: {filename}")
            return jsonify({"error": "Static file not found"}), 404
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}", exc_info=True)
        return jsonify({"error": "Static file error"}), 500
"""0. END SECTION 0 <HELPER FUNCTIONS>"""

"""1. START SECTION 1 DATA HANDLING AND LOADING"""

def save_to_cache(data, filename, folder=CACHE_DIR):
    """Save DataFrame to cache as CSV."""
    try:
        os.makedirs(folder, exist_ok=True)  # Ensure the cache folder exists
        filepath = os.path.join(folder, filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Data saved to cache: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to cache: {e}", exc_info=True)

def load_from_cache(filename, folder=CACHE_DIR):
    """Load DataFrame from cached CSV."""
    filepath = os.path.join(folder, filename)
    try:
        if os.path.exists(filepath):
            logger.info(f"Loading data from cache: {filepath}")
            return pd.read_csv(filepath)
        else:
            logger.warning(f"Cache file not found: {filepath}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load data from cache: {e}", exc_info=True)
        return pd.DataFrame()

def load_csv(file_path):
    """Load and normalize CSV file into a DataFrame."""
    full_path = os.path.join(DATA_DIR, file_path)
    try:
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            if df.empty:
                logger.warning(f"CSV file is empty: {full_path}")
            # Normalize column names
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            # Parse date columns if they exist
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            logger.info(f"Successfully loaded data from {full_path}")
            return df
        else:
            logger.warning(f"File not found: {full_path}")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}", exc_info=True)
    return pd.DataFrame()

def load_data():
    """Load and combine data for demand and weather."""
    try:
        years = range(2019, 2026)

        # Load demand data
        demand_dfs = []
        for year in years:
            file_path = f"demand/{year}.csv"
            df = load_csv(file_path)
            if not df.empty:
                demand_dfs.append(df)
        hourly_demand_data = pd.concat(demand_dfs, ignore_index=True) if demand_dfs else pd.DataFrame()

        # Load weather data
        weather_dfs = []
        for year in years:
            file_path = f"weather/{year}.csv"
            df = load_csv(file_path)
            if not df.empty:
                weather_dfs.append(df)
        hourly_weather_data = pd.concat(weather_dfs, ignore_index=True) if weather_dfs else pd.DataFrame()

        logger.info(f"Loaded {len(hourly_demand_data)} rows of demand data.")
        logger.info(f"Loaded {len(hourly_weather_data)} rows of weather data.")
        return hourly_demand_data, hourly_weather_data
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()

# Load data into global variables
hourly_demand_data, hourly_weather_data = load_data()
logger.debug(f"Hourly demand data: {hourly_demand_data.shape}")
logger.debug(f"Hourly weather data: {hourly_weather_data.shape}")

"""1. END SECTION 1 DATA HANDLING AND LOADING"""

"""2. START SECTION 2 DASHBOARD AND API ENDPOINTS"""

@app.route("/")
def dashboard():
    """Render the dashboard page."""
    try:
        logger.info("Rendering the main dashboard page.")
        return render_template("dashboard.html")
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}", exc_info=True)
        return jsonify({"error": "Failed to load dashboard"}), 500

@app.route("/api/dashboard", methods=["GET"])
def fetch_dashboard_data():
    try:
        demand_summary = hourly_demand_data.groupby(hourly_demand_data['time'].dt.date).agg({'value': 'sum'}).rename(columns={'value': 'total_demand'}).reset_index()
        weather_summary = hourly_weather_data.groupby(hourly_weather_data['datetime'].dt.date).agg({'temp': 'mean'}).reset_index()

        data = []
        for date in demand_summary['time']:
            demand = demand_summary[demand_summary['time'] == date]['total_demand'].values[0]
            temp = weather_summary[weather_summary['datetime'] == date]['temp'].values[0]
            data.append({'date': str(date), 'demand': demand, 'temperature': temp})

        return jsonify(data=data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

"""2. END SECTION 2 DASHBOARD AND API ENDPOINTS"""

"""3. START SECTION 3 HOURLY DEMAND"""

@app.route("/hourlydemand")
def hourly_demand_page():
    """Render the hourly demand page."""
    try:
        logger.info("Rendering the hourly demand page.")
        accept_header = request.headers.get("Accept", "")
        if "application/json" in accept_header:
            logger.info("Redirecting to hourly demand API for JSON response.")
            return redirect("/api/hourlydemand", code=302)
        return render_template("hourly_demand.html")
    except Exception as e:
        logger.error(f"Error rendering hourly demand page: {e}", exc_info=True)
        return jsonify({"error": "Failed to load hourly demand page"}), 500

@app.route("/api/hourlydemand", methods=["GET"])
def fetch_hourly_demand():
    """Fetch paginated hourly demand data."""
    try:
        if hourly_demand_data.empty:
            logger.warning("No hourly demand data available.")
            return jsonify({"error": "No hourly demand data available"}), 404

        # Pagination and filtering
        start = int(request.args.get("start", 0))
        length = int(request.args.get("length", 10))
        search_value = request.args.get("search[value]", "").lower()
        search_value = re.escape(search_value)

        # Filter and sort data
        df_sorted = hourly_demand_data.sort_values(by="time", ascending=False)
        if search_value:
            mask = (
                df_sorted["time"].astype(str).str.contains(search_value) |
                df_sorted["value"].astype(str).str.contains(search_value)
            )
            filtered_df = df_sorted[mask]
        else:
            filtered_df = df_sorted

        paginated_data = filtered_df.iloc[start:start + length].copy()
        paginated_data["time"] = paginated_data["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Prepare response
        response_data = {
            "draw": int(request.args.get("draw", 1)),
            "recordsTotal": len(hourly_demand_data),
            "recordsFiltered": len(filtered_df),
            "data": paginated_data.to_dict(orient="records")
        }

        logger.info("Hourly Demand API Response fetched successfully.")
        return jsonify(response_data), 200
    except Exception as e:
        logger.error(f"Error fetching hourly demand data: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch hourly demand data"}), 500

"""3. END SECTION 3 HOURLY DEMAND"""

"""4. START SECTION 4 HOURLY WEATHER"""

@app.route('/hourlyweather')
def hourly_weather_page():
    """Render the hourly weather page."""
    try:
        accept_header = request.headers.get('Accept', '')
        if 'application/json' in accept_header:
            return redirect('/api/hourlyweather', code=302)
        return render_template('hourly_weather.html')
    except Exception as e:
        logger.error(f"Error rendering hourly weather page: {e}", exc_info=True)
        return jsonify({"error": "Failed to load hourly weather page"}), 500

@app.route('/api/hourlyweather', methods=['GET'])
def fetch_hourly_weather():
    """Fetch hourly weather data."""
    try:
        if hourly_weather_data.empty:
            logger.warning("No hourly weather data available")
            return jsonify({"error": "No weather data available"}), 404

        # Pagination and filtering
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        search_value = request.args.get('search[value]', '').lower()
        search_value = re.escape(search_value)  # Prevent regex injection

        df_sorted = hourly_weather_data.sort_values(by='datetime', ascending=False)

        if search_value:
            mask = (
                df_sorted['datetime'].astype(str).str.contains(search_value) |
                df_sorted['temp'].astype(str).str.contains(search_value) |
                df_sorted['humidity'].astype(str).str.contains(search_value) |
                df_sorted['cloudcover'].astype(str).str.contains(search_value) |
                df_sorted['windspeed'].astype(str).str.contains(search_value) |
                df_sorted['precip'].astype(str).str.contains(search_value)
            )
            filtered_df = df_sorted[mask]
        else:
            filtered_df = df_sorted

        paginated_data = filtered_df.iloc[start:start + length].copy()
        paginated_data['datetime'] = paginated_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        response_data = {
            'draw': int(request.args.get('draw', 1)),
            'recordsTotal': len(hourly_weather_data),
            'recordsFiltered': len(filtered_df),
            'data': paginated_data.to_dict(orient='records')
        }

        logger.info("Hourly Weather API Response: %s", response_data)
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error fetching hourly weather data: {e}", exc_info=True)
        return jsonify({'error': 'Failed to fetch hourly weather data'}), 500

"""4. END SECTION 4 HOURLYWEATHER"""

"""5. START SECTION 5 HOLIDAYS"""

@app.route('/holidays')
def holidays_page():
    """Render the holidays page."""
    try:
        accept_header = request.headers.get('Accept', '')
        if 'application/json' in accept_header:
            return redirect('/api/holidays', code=302)
        return render_template('holidays.html')
    except Exception as e:
        logger.error(f"Error rendering holidays page: {e}", exc_info=True)
        return jsonify({"error": "Failed to load holidays page"}), 500

@cache.cached(timeout=86400, key_prefix='holidays')
@app.route('/api/holidays', methods=['GET'])
def fetch_holidays():
    """Fetch a list of holidays."""
    try:
        years = range(2019, 2026)
        cal_holidays = holidays.US(state='CA', years=years)
        holidays_list = [{'date': str(date), 'name': name} for date, name in cal_holidays.items()]

        if not holidays_list:
            logger.warning("No holidays data available")
            return jsonify({"error": "No holidays data available"}), 404

        # Pagination and filtering
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        search_value = request.args.get('search[value]', '').lower()
        search_value = re.escape(search_value)

        filtered_holidays = holidays_list
        if search_value:
            filtered_holidays = [
                holiday for holiday in holidays_list
                if search_value in holiday['name'].lower() or search_value in holiday['date']
            ]

        paginated_holidays = filtered_holidays[start:start + length]

        response_data = {
            'draw': int(request.args.get('draw', 1)),
            'recordsTotal': len(holidays_list),
            'recordsFiltered': len(filtered_holidays),
            'data': paginated_holidays
        }

        logger.info("Holidays API Response: %s", response_data)
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error fetching holidays data: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch holidays data"}), 500

"""5. END SECTION 5 HOLIDAYS"""

"""6. START SECTION 6 LAST UPDATED TIMESTAMP"""

@app.route('/api/lastUpdated', methods=['GET'])
def get_last_updated():
    """Fetch the last updated timestamps for demand, weather, and holidays."""
    try:
        # Get the latest timestamps from hourly demand and weather data
        latest_demand_timestamp = hourly_demand_data['time'].max() if not hourly_demand_data.empty else None
        latest_weather_timestamp = hourly_weather_data['datetime'].max() if not hourly_weather_data.empty else None

        # Fetch the latest holiday date from the holidays dataset
        years = range(2019, 2026)
        cal_holidays = holidays.US(state='CA', years=years)
        latest_holiday_date = max(cal_holidays.keys()) if cal_holidays else None

        # Prepare the response
        response_data = {
            'lastUpdatedDemand': latest_demand_timestamp.strftime('%d %b %Y, %H:%M') if latest_demand_timestamp else 'N/A',
            'lastUpdatedWeather': latest_weather_timestamp.strftime('%d %b %Y, %H:%M') if latest_weather_timestamp else 'N/A',
            'lastUpdatedHoliday': latest_holiday_date.strftime('%d %b %Y') if latest_holiday_date else 'N/A',
        }

        logger.info("Last Updated API Response: %s", response_data)
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error fetching last updated timestamps: {e}", exc_info=True)
        return jsonify({'error': 'Failed to fetch last updated timestamps'}), 500

"""6. END SECTION 6 LAST UPDATED TIMESTAMP"""

"""7. START SECTION 7 HEALTHCHECK ENDPOINT"""

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to validate data, static files, and service readiness."""
    try:
        # Check for existing files in the required directories
        data_files = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
        static_files = os.listdir(STATIC_DIR) if os.path.exists(STATIC_DIR) else []

        # Construct the response
        response_data = {
            "status": "healthy",
            "data_files_count": len(data_files),
            "static_files_count": len(static_files),
            "data_files": data_files[:5],  # Show up to 5 data files for quick diagnostics
            "static_files": static_files[:5],  # Show up to 5 static files for quick diagnostics
        }

        logger.info(f"Healthcheck successful: {response_data}")
        return jsonify(response_data), 200

    except Exception as e:
        # Log error and return unhealthy status
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

"""7. END SECTION 7 HEALTHCHECK ENDPOINT"""

"""8. START SECTION 8 MACHINE LEARNING"""










"""8. END SECTION 8 MACHINE LEARNING"""




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)