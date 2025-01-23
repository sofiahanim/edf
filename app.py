
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




import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
import xgboost as xgb
print(xgb.__version__)


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
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/")
def dashboard():
    """Render the dashboard page."""
    try:
        logger.info("Rendering the main dashboard page.")
        return render_template("dashboard.html")
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}", exc_info=True)
        return jsonify({"error": "Failed to load dashboard"}), 500


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

@app.route("/eda/demand", methods=["GET"])
def demand_eda():
    try:
        # Step 1: Define the range of years and initialize demand data list
        years = range(2019, 2026)
        demand_dfs = []

        # Step 2: Load and preprocess CSV files for each year
        for year in years:
            file_path = os.path.join(DATA_DIR, "demand", f"{year}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                df.dropna(subset=['time'], inplace=True)
                df['year'] = df['time'].dt.year
                df['month'] = df['time'].dt.month
                df['day_of_week'] = df['time'].dt.day_name()
                df['hour'] = df['time'].dt.hour
                demand_dfs.append(df)

        # Step 3: Combine all yearly DataFrames
        demand_data = pd.concat(demand_dfs, ignore_index=True) if demand_dfs else pd.DataFrame()

        # Step 4: Handle empty data
        if demand_data.empty:
            return jsonify({"error": "No demand data available"}), 404

        # Step 5: Compute summary metrics
        total_demand_per_year = demand_data.groupby('year')['value'].sum().reset_index(name='total_demand')
        avg_daily_demand = demand_data.groupby(demand_data['time'].dt.date)['value'].mean().mean()
        max_demand = demand_data['value'].max()
        min_demand = demand_data['value'].min()

        # Step 6: Aggregations for visualizations
        monthly_demand = demand_data.groupby(['year', 'month'])['value'].mean().reset_index()
        daily_demand = (
            demand_data.groupby('day_of_week')['value']
            .mean()
            .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            .reset_index()
        )
        hourly_demand = demand_data.groupby('hour')['value'].mean().reset_index(name='avg_demand')

        # Step 7: Compute Peak Demand Hours Analysis
        peak_demand_hours = (
            demand_data.groupby(['hour'])['value']
            .sum()
            .reset_index(name='total_demand')
            .sort_values(by='total_demand', ascending=False)
        )

        # Step 8: Compute Demand Heatmap (Hourly vs. Day-of-Week)
        heatmap_hourly_day = (
            demand_data.groupby(['day_of_week', 'hour'])['value']
            .mean()
            .unstack(fill_value=0)
            .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        )

        # Prepare heatmap data
        heatmap_data = [
            demand_data[demand_data['year'] == year].groupby('month')['value']
            .mean()
            .reindex(range(1, 13), fill_value=0)
            .tolist()
            for year in demand_data['year'].unique()
        ]

        # Demand Distribution Data
        demand_distribution = demand_data['value'].tolist()

        # Step 9: Convert to Python-native types for JSON serialization
        response_data = {
            "total_demand_per_year": total_demand_per_year.astype(object).to_dict(orient='records'),
            "avg_daily_demand": float(avg_daily_demand),
            "max_demand": float(max_demand),
            "min_demand": float(min_demand),
            "monthly_demand": monthly_demand.astype(object).to_dict(orient='records'),
            "daily_demand": daily_demand.astype(object).to_dict(orient='records'),
            "hourly_demand": hourly_demand.astype(object).to_dict(orient='records'),
            "peak_demand_hours": peak_demand_hours.astype(object).to_dict(orient='records'),
            "heatmap_hourly_day": heatmap_hourly_day.values.tolist(),
            "heatmap_data": heatmap_data,
            "demand_distribution": list(map(float, demand_distribution)),
        }


        # Check 'Accept' header to decide JSON or HTML response
        accept_header = request.headers.get('Accept', '')
        if 'application/json' in accept_header:
            return jsonify(response_data)

        # Render HTML template
        return render_template(
            "demand_eda.html",
            total_demand_per_year=total_demand_per_year.to_dict(orient='records'),
            avg_daily_demand=avg_daily_demand,
            max_demand=max_demand,
            min_demand=min_demand,
            monthly_demand=monthly_demand.to_dict(orient='records'),
            daily_demand=daily_demand.to_dict(orient='records'),
            hourly_demand=hourly_demand.to_dict(orient='records'),
            peak_demand_hours=peak_demand_hours.to_dict(orient='records'),
            heatmap_hourly_day=heatmap_hourly_day.values.tolist(),
        )

    except Exception as e:
        app.logger.error(f"Error in Demand EDA: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

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


@app.route('/eda/weather', methods=['GET'])
def weather_eda():
    try:
        years = range(2019, 2026)
        weather_dfs = []

        for year in years:
            file_path = os.path.join(DATA_DIR, "weather", f"{year}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

                required_columns = {'datetime', 'temp', 'humidity', 'windspeed', 'solaradiation', 'preciptype'}
                if required_columns.issubset(df.columns):
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    df.dropna(subset=['datetime'], inplace=True)
                    df['year'] = df['datetime'].dt.year
                    df['month'] = df['datetime'].dt.month
                    df['hour'] = df['datetime'].dt.hour
                    df['day'] = df['datetime'].dt.date
                    weather_dfs.append(df)
                else:
                    logger.warning(f"Missing columns in {file_path}. Skipping.")
            else:
                logger.warning(f"File not found: {file_path}")

        weather_data = pd.concat(weather_dfs, ignore_index=True) if weather_dfs else pd.DataFrame()

        if weather_data.empty:
            return jsonify({"error": "No weather data available"}), 404

        # Existing Visualizations
        avg_temp_by_year = weather_data.groupby('year')['temp'].mean().reset_index(name='avg_temp')
        avg_humidity_by_month = weather_data.groupby('month')['humidity'].mean().reset_index(name='avg_humidity')
        wind_speed_distribution = weather_data['windspeed'].value_counts().reset_index(name='count')
        wind_speed_distribution.rename(columns={'index': 'windspeed'}, inplace=True)
        solar_radiation_by_hour = weather_data.groupby('hour')['solaradiation'].mean().reset_index(name='avg_radiation')
        precip_type_distribution = weather_data['preciptype'].value_counts(normalize=True).reset_index(name='percentage')
        precip_type_distribution.rename(columns={'index': 'type'}, inplace=True)

        # New Visualizations
        # 1. Monthly Average Temperature (Seasonal Trends)
        monthly_avg_temp = weather_data.groupby(['year', 'month'])['temp'].mean().reset_index()

        # 2. Daily Average Humidity
        daily_avg_humidity = weather_data.groupby('day')['humidity'].mean().reset_index(name='avg_humidity')

        # 3. Hourly Average Wind Speed
        hourly_avg_windspeed = weather_data.groupby('hour')['windspeed'].mean().reset_index(name='avg_windspeed')

        highest_temp = weather_data['temp'].max()
        lowest_temp = weather_data['temp'].min()
        highest_wind_speed = weather_data['windspeed'].max()
        total_precipitation = weather_data['precip'].sum()
        avg_solar_radiation = weather_data['solaradiation'].mean()
        most_frequent_precip_type = weather_data['preciptype'].mode()[0]

        # Preparing Response Data
        response_data = {
            'highest_temp': highest_temp,
            'lowest_temp': lowest_temp,
            'highest_wind_speed': highest_wind_speed,
            'total_precipitation': total_precipitation,
            'avg_solar_radiation': avg_solar_radiation,
            'most_frequent_precip_type': most_frequent_precip_type,
            'avg_temp_by_year': avg_temp_by_year.to_dict(orient='records'),
            'avg_humidity_by_month': avg_humidity_by_month.to_dict(orient='records'),
            'wind_speed_distribution': wind_speed_distribution.to_dict(orient='records'),
            'solar_radiation_by_hour': solar_radiation_by_hour.to_dict(orient='records'),
            'precip_type_distribution': precip_type_distribution.to_dict(orient='records'),
            'monthly_avg_temp': monthly_avg_temp.to_dict(orient='records'),
            'daily_avg_humidity': daily_avg_humidity.to_dict(orient='records'),
            'hourly_avg_windspeed': hourly_avg_windspeed.to_dict(orient='records'),
        }

        # Respond as JSON or render template
        if 'application/json' in request.headers.get('Accept', ''):
            return jsonify(response_data)

        return render_template("weather_eda.html", **response_data)

    except Exception as e:
        logger.error(f"Error processing weather EDA: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


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


@app.route('/eda/holiday', methods=['GET'])
def holiday_eda():
    try:
        # Define the range of years to process
        years = range(2019, 2026)
        holiday_dfs = []

        # Load CSV files for each year and preprocess
        for year in years:
            file_path = os.path.join(DATA_DIR, "holiday", f"{year}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df.dropna(subset=['date'], inplace=True)
                    df['year'] = df['date'].dt.year
                    df['month'] = df['date'].dt.month
                    holiday_dfs.append(df)

        # Combine data from all years
        holiday_data = pd.concat(holiday_dfs, ignore_index=True) if holiday_dfs else pd.DataFrame()

        if holiday_data.empty:
            return jsonify({"error": "No holiday data available"}), 404

        # Compute metrics for the response
        total_holidays = len(holiday_data)
        common_month = holiday_data['month'].mode()[0]
        common_month_name = pd.to_datetime(str(common_month), format='%m').strftime('%B')
        holiday_trends = holiday_data.groupby('year')['date'].count().reset_index(name='total_holidays')
        monthly_distribution = (
            holiday_data['month']
            .value_counts(normalize=True)
            .reset_index()
            .rename(columns={'index': 'month', 'month': 'percentage'})
            .sort_values('month')
        )
        top_holidays_per_year = holiday_data.groupby(['year', 'name']).size().reset_index(name='count')
        heatmap_data = []
        for year in holiday_data['year'].unique():
            year_data = holiday_data[holiday_data['year'] == year]
            monthly_counts = year_data.groupby('month').size().reindex(range(1, 13), fill_value=0).tolist()
            heatmap_data.append(monthly_counts)

        holiday_data['day_of_week'] = holiday_data['date'].dt.day_name()
        holidays_by_day = (
            holiday_data['day_of_week']
            .value_counts()
            .reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], fill_value=0)
            .reset_index()
            .rename(columns={'index': 'day', 'day_of_week': 'count'})
            .to_dict(orient='records')
        )

        # Prepare the response data
        response_data = {
            "total_holidays": total_holidays,
            "common_month": common_month_name,
            "holiday_trends": holiday_trends.to_dict(orient='records'),
            "holidays_by_day": holidays_by_day,
            "monthly_distribution": monthly_distribution.to_dict(orient='records'),
            "top_holidays_per_year": top_holidays_per_year.to_dict(orient='records'),
            "heatmap_data": heatmap_data,
        }

        # Determine response format based on 'Accept' header
        accept_header = request.headers.get('Accept', '')
        if 'application/json' in accept_header:
            return jsonify(response_data)  # Return JSON response

        # Render HTML response for browser requests
        return render_template("holiday_eda.html", **response_data)

    except Exception as e:
        logger.error(f"Error in Holiday EDA: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/invalidate_cache', methods=['POST'])
def invalidate_cache():
    try:
        scope = request.json.get('scope', 'all')  # Default to clearing all caches
        if scope == 'holidays':
            cache.pop('holiday_data', None)
        elif scope == 'demand':
            cache.pop('demand_data', None)
        elif scope == 'weather':
            cache.pop('weather_data', None)
        else:
            cache.clear()  # Clear all caches

        return jsonify({"message": f"Cache invalidated successfully for scope: {scope}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

##############################################################################################################

"""8. START SECTION 8 MLOPS"""

@app.route("/mlops_preprocessing", methods=["GET"])
def mlops_preprocessing():
    """
    Render the Data Preprocessing page with dynamic content for SB Admin 2.
    """
    try:
        # Path to dataset
        allyears_path = os.path.join(DATA_DIR, "merge", "allyears.csv")

        # Define preprocessing steps
        processes = [
            "Step 1: Fetch demand data from AWS Redshift (`pipe_data.py`).",
            "Step 2: Fetch weather data from AWS Redshift (`pipe_data_2.py`).",
            "Step 3: Merge demand, weather, and holiday data (`ml_part_1.py`).",
            "Step 4: Handle missing values with forward fill (`ml_part_1.py`).",
            "Step 5: Add binary holiday indicators (`ml_part_1.py`)."
        ]

        # Load dataset for summary metrics
        if os.path.exists(allyears_path):
            df = pd.read_csv(allyears_path)

            # Compute summary metrics
            missing_values = int(df.isnull().sum().sum())
            total_rows = len(df)
            total_columns = len(df.columns)
            column_names = df.columns.tolist()
        else:
            # Handle the case where the dataset is missing
            df = pd.DataFrame()
            missing_values = total_rows = total_columns = 0
            column_names = []

        # Log summary metrics for debugging
        app.logger.info(f"Summary Metrics: Rows={total_rows}, Columns={total_columns}, Missing Values={missing_values}")

        # Pass summary metrics to the frontend
        summary = {
            "missing_values": missing_values,
            "total_rows": total_rows,
            "total_columns": total_columns,
            "columns": column_names,
        }

        return render_template(
            "mlops_preprocessing.html",
            title="Data Preprocessing",
            summary=summary,
            processes=processes,
        )
    except Exception as e:
        app.logger.error(f"Error rendering Data Preprocessing page: {e}", exc_info=True)
        return jsonify({"error": "Failed to load Data Preprocessing page"}), 500

@app.route("/mlops_preprocessing/data", methods=["GET"])
def mlops_preprocessing_data():
    """
    Provide all data from allyears.csv for DataTables in JSON format.
    """
    try:
        # Path to the dataset
        allyears_path = os.path.join(DATA_DIR, "merge", "allyears.csv")
        if not os.path.exists(allyears_path):
            app.logger.warning("Dataset file not found.")
            return jsonify({"error": "Dataset file not found."}), 404

        # Load the dataset
        df = pd.read_csv(allyears_path)

        # Validate required columns
        required_columns = {
            "ds", "y", "temp", "feelslike", "humidity", "windspeed", "cloudcover",
            "solaradiation", "precip", "preciptype", "date", "is_holiday"
        }
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            app.logger.error(f"Dataset is missing required columns: {missing_cols}")
            return jsonify({"error": f"Dataset missing required columns: {list(missing_cols)}"}), 400

        # Convert `ds` to datetime and ensure proper datetime64[ns] type
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        if df["ds"].isnull().any():
            app.logger.error("Invalid datetime values found in 'ds' column.")
            return jsonify({"error": "Invalid datetime values in 'ds' column."}), 400

        # Sort the dataset by `ds` in descending order
        df = df.sort_values(by="ds", ascending=False).reset_index(drop=True)

        # Debug log to verify sorted data
        app.logger.info(f"First 5 rows after sorting: {df.head(5).to_dict(orient='records')}")

        # Convert to JSON format
        response_data = {"data": df.to_dict(orient="records")}
        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"Error in /mlops_preprocessing/data: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error."}), 500


##############################################################################################################

def clean_parameters(dataframe, column_name):
    """
    Ensure Parameters column is treated as a valid JSON string and handles invalid values gracefully.
    """
    def stringify(param):
        try:
            if pd.isna(param) or param == "" or param == "NaN":  # Handle empty, NaN, or None
                return json.dumps({"Default Parameters": ""})
            return json.dumps(param) if isinstance(param, (dict, list)) else str(param)  # Convert to JSON string
        except Exception as e:
            app.logger.error(f"Error converting parameter to string: {param}, Error: {e}")
            return json.dumps({"Invalid": "Unable to parse"})

    dataframe[column_name] = dataframe[column_name].apply(stringify)

@app.route("/mlops_trainingvalidation", methods=["GET"])
def mlops_trainingvalidation():
    try:
        # Define the training logs path
        training_logs_path = os.path.join(BASE_DIR, "training", "training_info.csv")
        
        # Check if the file exists and process it
        if os.path.exists(training_logs_path):
            training_logs_df = pd.read_csv(training_logs_path)
            clean_parameters(training_logs_df, "Parameters")  # Clean parameters column
            training_summary = {
                "total_rows": len(training_logs_df),
                "total_columns": len(training_logs_df.columns),
                "columns": training_logs_df.columns.tolist(),
            }
        else:
            training_logs_df = pd.DataFrame()
            training_summary = {"total_rows": 0, "total_columns": 0, "columns": []}

        # Validation Logs
        validation_logs_path = os.path.join(BASE_DIR, "validation", "consolidated_validation_metrics.csv")
        if os.path.exists(validation_logs_path):
            validation_logs_df = pd.read_csv(validation_logs_path)
            clean_parameters(validation_logs_df, "Parameters")
            validation_summary = {
                "total_rows": len(validation_logs_df),
                "total_columns": len(validation_logs_df.columns),
                "columns": validation_logs_df.columns.tolist(),
            }
        else:
            validation_logs_df = pd.DataFrame()
            validation_summary = {"total_rows": 0, "total_columns": 0, "columns": []}

        # Render the training logs page
        return render_template(
            "mlops_trainingvalidation.html",
            title="Model Training & Validation",
            training_summary=training_summary,
            validation_summary=validation_summary,
        )
    except Exception as e:
        app.logger.error(f"Error rendering training logs page: {e}", exc_info=True)
        return jsonify({"error": "Failed to load Model Training Logs page"}), 500
    
@app.route("/api/mlops/validation/logs", methods=["GET"])
def fetch_validation_logs():
    """
    Provide validation logs in JSON format.
    """
    try:
        validation_logs_path = os.path.join(BASE_DIR, "validation", "consolidated_validation_metrics.csv")
        if not os.path.exists(validation_logs_path):
            return jsonify({"error": "Validation logs file not found."}), 404

        validation_logs_df = pd.read_csv(validation_logs_path)
        clean_parameters(validation_logs_df, "Parameters")  # Clean Parameters column
        response_data = {"data": validation_logs_df.to_dict(orient="records")}
        return jsonify(response_data), 200
    except Exception as e:
        app.logger.error(f"Error fetching validation logs: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch validation logs."}), 500
    
@app.route("/api/mlops/training/logs", methods=["GET"])
def fetch_training_logs():
    try:
        training_logs_path = os.path.join(BASE_DIR, "training", "training_info.csv")
        if not os.path.exists(training_logs_path):
            return jsonify({"error": "Training logs file not found."}), 404

        training_logs_df = pd.read_csv(training_logs_path)

        # Replace NaN or invalid values with "N/A"
        training_logs_df.fillna("N/A", inplace=True)
        training_logs_df.replace("NaN", "N/A", inplace=True)

        response_data = {"data": training_logs_df.to_dict(orient="records")}
        return jsonify(response_data), 200
    except Exception as e:
        app.logger.error(f"Error fetching training logs: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch training logs."}), 500


##############################################################################################################



##############################################################################################################

@app.route('/mlops_predictionevaluation', methods=['GET'])
def prediction_evaluation_page():
    """
    Render the prediction evaluation page.
    """
    try:
        # Fetch data directly from the API endpoint
        evaluation_data = get_prediction_evaluation_data()

        # Handle errors from the API response
        if "error" in evaluation_data:
            raise ValueError(evaluation_data["error"])

        # Extract metrics summary and chart data
        metrics_summary = evaluation_data.get("metrics_summary", [])
        chart_data = evaluation_data.get("chart_data", [])
        prediction_comparison = evaluation_data.get("prediction_comparison", [])

        # Process metrics summary to rank models
        metrics_summary_df = pd.DataFrame(metrics_summary)
        metrics_summary_df["r_squared"] = pd.to_numeric(metrics_summary_df["r_squared"], errors="coerce")
        metrics_summary_df["rank_mae"] = metrics_summary_df["mae"].rank()
        metrics_summary_df["rank_r2"] = metrics_summary_df["r_squared"].rank(ascending=False)
        metrics_summary_df["average_rank"] = metrics_summary_df[["rank_mae", "rank_r2"]].mean(axis=1)
        best_model = metrics_summary_df.sort_values("average_rank").iloc[0].to_dict()

        # Render the HTML page with the best model and data for visualization
        return render_template(
            "mlops_predictionevaluation.html",
            title="Prediction & Evaluation",
            best_model=best_model,
            chart_data=chart_data,
            metrics_summary=metrics_summary,
            prediction_comparison=prediction_comparison
        )
    except Exception as e:
        logger.error(f"Error rendering prediction evaluation page: {e}", exc_info=True)
        return jsonify({"error": "Failed to load prediction evaluation page."}), 500


@app.route('/api/mlops_predictionevaluation', methods=['GET'])
def get_prediction_evaluation_data():
    """
    API endpoint to fetch prediction evaluation data for DataTables.
    """
    try:
        # Define paths for CSV files
        prophet_csv_path = os.path.join(BASE_DIR, "evaluation", "prophet_future_predictions.csv")
        theta_csv_path = os.path.join(BASE_DIR, "evaluation", "theta_future_predictions.csv")
        gbr_csv_path = os.path.join(BASE_DIR, "evaluation", "gbr_future_predictions.csv")
        summary_csv_path = os.path.join(BASE_DIR, "evaluation", "summary_report.csv")

        # Load CSV files
        prophet_df = load_csv(prophet_csv_path)
        theta_df = load_csv(theta_csv_path)
        gbr_df = load_csv(gbr_csv_path)
        summary_df = load_csv(summary_csv_path)

        # Rename prediction columns for consistency
        prophet_df.rename(columns={"Prophet_Predicted": "y"}, inplace=True)
        theta_df.rename(columns={"Theta_Predicted": "y"}, inplace=True)
        gbr_df.rename(columns={"Predicted": "y"}, inplace=True)

        # Add model identifiers
        prophet_df["model"] = "Prophet"
        theta_df["model"] = "Theta"
        gbr_df["model"] = "GBR"

        # Combine prediction dataframes
        combined_predictions = pd.concat([prophet_df, theta_df, gbr_df], ignore_index=True).fillna(0)

        # Standardize column names
        summary_df.columns = summary_df.columns.str.strip().str.lower()
        summary_df.rename(columns={"r²": "r_squared"}, inplace=True)

        # Fill missing values with 0 for rendering purposes
        summary_df.fillna(0, inplace=True)

        # Prepare metrics summary
        metrics_summary = summary_df[[
            "model", "mae", "mape", "rmse", "r_squared", "mbe", "parameters"
        ]].to_dict(orient="records")

        # Prepare chart data
        chart_ready_df = summary_df.dropna(subset=["mae", "mape", "rmse", "r_squared", "mbe"])
        chart_ready_df.fillna(0, inplace=True)
        chart_data = chart_ready_df[[
            "model", "mae", "mape", "rmse", "r_squared", "mbe"
        ]].to_dict(orient="records")

        # Return the combined results
        return {
            "metrics_summary": metrics_summary,
            "prediction_comparison": combined_predictions.to_dict(orient="records"),
            "chart_data": chart_data
        }
    except Exception as e:
        logger.error(f"Error in get_prediction_evaluation_data: {e}", exc_info=True)
        return {"error": "Failed to load prediction evaluation data."}

##############################################################################################################

@app.route("/mlops_discussion")
def mlops_discussion():
    """
    Route for the Discussion page.
    """
    try:
        # Example: Render discussion details
        return render_template("mlops_discussion.html", title="Discussion")
    except Exception as e:
        app.logger.error(f"Error rendering Discussion page: {e}", exc_info=True)
        return jsonify({"error": "Failed to load Discussion page"}), 500

"""8. END SECTION 8 MLOPS"""


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)