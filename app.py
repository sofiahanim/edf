from flask import Flask, Response, redirect, jsonify, request, render_template, send_from_directory
import pandas as pd
import holidays
from flask_caching import Cache
from flask_cors import CORS
import logging
import os
import subprocess
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.wrappers import Request, Response
from pathlib import Path
from mimetypes import guess_type
import json
import re
import plotly.express as px
from serverless_wsgi import handle_request
from app import app

# Initialize app and API
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['CACHE_TYPE'] = 'simple' 
cache = Cache(app)
cache.init_app(app)

CORS(app, resources={r"/*": {"origins": "*"}})

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories
base_dir = os.getenv("DATA_BASE_DIR", os.path.join(os.getcwd(), "data"))
static_dir = os.path.join(os.getcwd(), 'static')
template_dir = os.path.join(os.getcwd(), 'templates')

# Use /tmp for cache to ensure it is writable in serverless environments
cache_dir = os.getenv("CACHE_DIR", '/tmp/cache')

logger.info(f"Base directory: {base_dir}")
logger.info(f"Static directory: {static_dir}")
logger.info(f"Template directory: {template_dir}")
logger.info(f"Cache directory: {cache_dir}")

# Ensure cache directory exists
try:
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Cache directory created: {cache_dir}")
except OSError as e:
    logger.error(f"Failed to create cache directory: {e}")
    raise


def lambda_handler(event, context):
    """
    AWS Lambda handler to serve the Flask application via API Gateway.
    """
    try:
        # Log event and context for debugging
        logger.info("Received event: %s", json.dumps(event, indent=2))
        
        # Ensure the Flask app is wrapped in a DispatcherMiddleware for compatibility
        app.wsgi_app = DispatcherMiddleware(None, {"/": app.wsgi_app})
        
        # Handle the request using serverless_wsgi
        response = handle_request(app, event, context)

        # Ensure a valid API Gateway-compatible response
        return {
            "statusCode": response["statusCode"],
            "headers": {
                **response.get("headers", {}),
                "Access-Control-Allow-Origin": "*",  # Enable CORS if required
            },
            "body": response["body"],  # HTML, JSON, or other content
            "isBase64Encoded": response.get("isBase64Encoded", False),  # Handle binary responses
        }
    except Exception as e:
        # Log the error and return a generic error response
        logger.error(f"Error in Lambda handler: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Internal Server Error"}),
        }
    
"""README"""


"""end README"""

"""1. START SECTION 1 DATA"""

def save_to_cache(data, filename, folder='cache'):
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Data saved to cache: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to cache: {e}", exc_info=True)

def load_from_cache(filename, folder='cache'):
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

@app.route('/static/<path:filename>')
def serve_static_files(filename):
    """Serve static files."""
    mimetype, _ = guess_type(filename)
    if mimetype is None:
        mimetype = "application/octet-stream"
    
    return send_from_directory('static', filename, mimetype=mimetype)

# Helper function to load and normalize CSV files
def load_and_normalize_csv(path):
    try:
        df = pd.read_csv(path)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {path}, Error: {e}", exc_info=True)
        return pd.DataFrame()

def load_data():
    years = range(2019, 2026)
    hourly_demand_data = pd.DataFrame()
    hourly_weather_data = pd.DataFrame()

    for year in years:
        path_demand = os.path.join(base_dir, 'demand', f'{year}.csv')
        if os.path.exists(path_demand):
            try:
                logger.info(f"Loading demand data for year {year}: {path_demand}")
                demand_data = load_and_normalize_csv(path_demand)
                hourly_demand_data = pd.concat([hourly_demand_data, demand_data], ignore_index=True)
            except Exception as e:
                logger.error(f"Failed to load demand data for year {year}: {e}", exc_info=True)

        path_weather = os.path.join(base_dir, 'weather', f'{year}.csv')
        if os.path.exists(path_weather):
            try:
                logger.info(f"Loading weather data for year {year}: {path_weather}")
                weather_data = load_and_normalize_csv(path_weather)
                hourly_weather_data = pd.concat([hourly_weather_data, weather_data], ignore_index=True)
            except Exception as e:
                logger.error(f"Failed to load weather data for year {year}: {e}", exc_info=True)

    logger.info(f"Total demand data rows loaded: {len(hourly_demand_data)}")
    logger.info(f"Total weather data rows loaded: {len(hourly_weather_data)}")

    return hourly_demand_data, hourly_weather_data

hourly_demand_data, hourly_weather_data = load_data()

"""1. END SECTION 1 DATA"""

"""2. START SECTION 2 DASHBOARD"""
@app.route('/')
def dashboard():
    """Render the dashboard page.""" 
    try:
        accept_header = request.headers.get('Accept', '')
        if 'application/json' in accept_header:
            return redirect('/api/dashboard', code=302)
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}", exc_info=True)
        return jsonify({"error": "Failed to load dashboard"}), 500


@app.route('/api/dashboard', methods=['GET'])
def fetch_dashboard_data():
    """Fetch summarized dashboard data."""
    try:
        demand_summary = (
            hourly_demand_data
            .groupby(hourly_demand_data['time'].dt.date)
            .agg({'value': 'sum'})
            .rename(columns={'value': 'total_demand'})
            .reset_index()
        )

        weather_summary = (
            hourly_weather_data
            .groupby(hourly_weather_data['datetime'].dt.date)
            .agg({'temp': 'mean'})
            .rename(columns={'temp': 'average_temperature'})
            .reset_index()
        )

        combined_data = pd.merge(
            demand_summary,
            weather_summary,
            left_on='time',
            right_on='datetime',
            how='outer'
        ).fillna({'total_demand': 0, 'average_temperature': 0})

        response_data = combined_data.rename(columns={'time': 'date'}).to_dict(orient='records')

        logger.info("Dashboard API Response: %s", response_data)
        return jsonify({'data': response_data})
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {e}", exc_info=True)
        return jsonify({'error': 'Failed to fetch dashboard data'}), 500

"""2. END SECTION 2 DASHBOARD"""

"""3. START SECTION 3 HOURLYDEMAND"""

@app.route('/hourlydemand')
def hourly_demand_page():
    """Render the hourly demand page."""
    try:
        accept_header = request.headers.get('Accept', '')
        if 'application/json' in accept_header:
            return redirect('/api/hourlydemand', code=302)
        return render_template('hourly_demand.html')
    except Exception as e:
        logger.error(f"Error rendering hourly demand page: {e}", exc_info=True)
        return jsonify({"error": "Failed to load hourly demand page"}), 500


@app.route('/eda/demand', methods=['GET'])
def eda_demand_page():
    """Render the EDA page for demand data."""
    try:
        # Paths to demand data files
        demand_file_paths = [
            "data/demand/2019.csv",
            "data/demand/2020.csv",
            "data/demand/2021.csv",
            "data/demand/2022.csv",
            "data/demand/2023.csv",
            "data/demand/2024.csv",
            "data/demand/2025.csv"
        ]

        # Load and process demand data
        demand_data_frames = [pd.read_csv(file_path) for file_path in demand_file_paths]
        demand_data = pd.concat(demand_data_frames, ignore_index=True)
        demand_data['time'] = pd.to_datetime(demand_data['time'], errors='coerce')
        demand_data = demand_data.dropna(subset=['time', 'value'])

        # Add time components
        demand_data['year'] = demand_data['time'].dt.year
        demand_data['month'] = demand_data['time'].dt.month
        demand_data['hour'] = demand_data['time'].dt.hour

        # Compute statistics
        summary_stats = demand_data.groupby('year')['value'].agg(['mean', 'median', 'min', 'max', 'sum']).reset_index()
        daily_trends = demand_data.groupby(demand_data['time'].dt.date)['value'].sum().reset_index()
        daily_trends.columns = ['date', 'total_demand']
        top_5_days = demand_data.groupby(demand_data['time'].dt.date)['value'].sum().nlargest(5).reset_index()
        top_5_days.columns = ['date', 'total_demand']

        # Generate dynamic visualizations with Plotly
        # Monthly Average Line Plot
        monthly_avg = demand_data.groupby(['year', 'month'])['value'].mean().reset_index()
        fig_monthly_avg = px.line(
            monthly_avg, x='month', y='value', color='year',
            title='Monthly Average Demand by Year',
            labels={'value': 'Average Demand (kWh)', 'month': 'Month'}
        )
        fig_monthly_avg.update_layout(legend_title_text='Year')

        # Hourly Average Bar Plot
        hourly_avg = demand_data.groupby('hour')['value'].mean().reset_index()
        fig_hourly_avg = px.bar(
            hourly_avg, x='hour', y='value',
            title='Hourly Average Demand',
            labels={'value': 'Average Demand (kWh)', 'hour': 'Hour of Day'}
        )

        # Heatmap
        heatmap_data = demand_data.pivot_table(index=demand_data['time'].dt.date, columns='hour', values='value', aggfunc='mean').reset_index()
        fig_heatmap = px.imshow(
            heatmap_data.set_index('time').T,
            labels=dict(color='Demand (kWh)', x='Date', y='Hour of Day'),
            title='Daily Demand Heatmap'
        )

        # Prepare data for rendering
        summary_stats_json = summary_stats.to_dict(orient='records')
        daily_trends_json = daily_trends.to_dict(orient='records')
        top_5_days_json = top_5_days.to_dict(orient='records')

        return render_template(
            'eda_demand.html',
            summary_stats=summary_stats_json,
            daily_trends=daily_trends_json,
            top_5_days=top_5_days_json,
            monthly_avg_plot=fig_monthly_avg.to_json(),
            hourly_avg_plot=fig_hourly_avg.to_json(),
            heatmap_plot=fig_heatmap.to_json()
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
@app.route('/api/hourlydemand', methods=['GET'])
def fetch_hourly_demand():
    """Fetch hourly demand data."""
    try:
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        search_value = request.args.get('search[value]', '').lower()
        search_value = re.escape(search_value)

        df_sorted = hourly_demand_data.sort_values(by='time', ascending=False)
        if search_value:
            mask = (
                df_sorted['time'].astype(str).str.contains(search_value) |
                df_sorted['value'].astype(str).str.contains(search_value)
            )
            filtered_df = df_sorted[mask]
        else:
            filtered_df = df_sorted

        paginated_data = filtered_df.iloc[start:start + length].copy()
        paginated_data['time'] = paginated_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        response_data = {
            'draw': int(request.args.get('draw', 1)),
            'recordsTotal': len(hourly_demand_data),
            'recordsFiltered': len(filtered_df),
            'data': paginated_data.to_dict(orient='records')
        }

        logger.info("Hourly Demand API Response: %s", response_data)
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error fetching hourly demand data: {e}", exc_info=True)
        return jsonify({'error': 'Failed to fetch hourly demand data'}), 500

"""3. END SECTION 3 HOURLYDEMAND"""

"""4. START SECTION 4 HOURLYWEATHER"""

@app.route('/hourlyweather')
def hourly_weather_page():
    """Fetch hourly weather data."""
    try:
        accept_header = request.headers.get('Accept', '')
        if 'application/json' in accept_header:
            return redirect('/api/hourlyweather', code=302)
        return render_template('hourly_weather.html')
    except Exception as e:
        logger.error(f"Error rendering hourly weather page: {e}", exc_info=True)
        return jsonify({"error": "Failed to load hourly weather page"}), 500

@app.route('/eda/weather')
def eda_weather_page():
    years = range(2019, 2026)
    weather_data = pd.concat([load_and_normalize_csv(f'data/weather/{year}.csv') for year in years], ignore_index=True)
    return render_template(
        'eda_weather.html',
        data=weather_data.to_html(classes='table table-striped', index=False)
    )

@app.route('/api/hourlyweather', methods=['GET'])
def fetch_hourly_weather():
    """Fetch hourly weather data."""
    try:
        if hourly_weather_data.empty:
            return jsonify({"message": "No weather data available"}), 204
        
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        search_value = request.args.get('search[value]', '').lower()
        search_value = re.escape(search_value)  # Escape special characters to avoid regex injection

        df_sorted = hourly_weather_data.sort_values(by='datetime', ascending=False)
        if search_value:
            mask = (
                df_sorted['datetime'].astype(str).str.contains(search_value) |
                df_sorted['temp'].astype(str).str.contains(search_value) |
                df_sorted['feelslike'].astype(str).str.contains(search_value) |
                df_sorted['humidity'].astype(str).str.contains(search_value) |
                df_sorted['windspeed'].astype(str).str.contains(search_value) |
                df_sorted['cloudcover'].astype(str).str.contains(search_value) |
                df_sorted['solaradiation'].astype(str).str.contains(search_value) |
                df_sorted['precip'].astype(str).str.contains(search_value) |
                df_sorted['preciptype'].astype(str).str.contains(search_value)
            )
            filtered_df = df_sorted[mask]
        else:
            filtered_df = df_sorted

        paginated_data = filtered_df.iloc[start:start + length].copy()
        paginated_data['datetime'] = paginated_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        paginated_data.fillna({'preciptype': 'N/A'}, inplace=True)

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
        logger.error(f"Error rendering holiday page: {e}", exc_info=True)
        return jsonify({"error": "Failed to load holiday page"}), 500

@app.route('/eda/holidays')
def eda_holidays_page():
    years = range(2019, 2026)
    cal_holidays = holidays.US(state='CA', years=years)
    holiday_data = pd.DataFrame([{'date': str(date), 'name': name} for date, name in cal_holidays.items()])
    return render_template(
        'eda_holidays.html',
        data=holiday_data.to_html(classes='table table-striped', index=False)
    )

@cache.cached(timeout=86400, key_prefix='holidays')
@app.route('/api/holidays', methods=['GET'])
def fetch_holidays():
    """Fetch a list of holidays."""
    try:
        start = int(request.args.get('start', 0))  # Pagination start
        length = int(request.args.get('length', 10))  # Pagination length
        search_value = request.args.get('search[value]', '').lower()  # Search query
        search_value = re.escape(search_value) 
        order_column = int(request.args.get('order[0][column]', 0))  # Sorting column index
        order_dir = request.args.get('order[0][dir]', 'asc')  # Sorting direction

        cal_holidays = holidays.US(state='CA', years=range(2019, 2026))
        holidays_list = [{'date': str(date), 'name': name} for date, name in cal_holidays.items()]

        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        if start_time and end_time:
            holidays_list = [h for h in holidays_list if start_time <= h['date'] <= end_time]

        if search_value:
            holidays_list = [
                h for h in holidays_list
                if search_value in h['name'].lower() or search_value in h['date']
            ]

        order_key = 'date' if order_column == 0 else 'name'
        holidays_list = sorted(
            holidays_list,
            key=lambda x: x[order_key],
            reverse=(order_dir == 'desc')
        )

        records_total = len(holidays_list)
        holidays_list = holidays_list[start:start + length]

        response_data = {
            "draw": int(request.args.get('draw', 1)),
            "recordsTotal": records_total,
            "recordsFiltered": records_total,
            "data": holidays_list
        }

        logger.info("Holidays API Response: %s", response_data)
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error fetching holidays: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch holidays"}), 400


"""5. END SECTION 5 HOLIDAYS"""

"""6. START SECTION 6 LAST UPDATED"""

@app.route('/api/lastUpdated', methods=['GET'])
def get_last_updated():
    try:
        latest_demand_timestamp = hourly_demand_data['time'].max()
        latest_weather_timestamp = hourly_weather_data['datetime'].max()

        cal_holidays = holidays.US(state='CA', years=range(2019, 2026))
        latest_holiday = max(cal_holidays.keys()) if cal_holidays else None

        response_data = {
            'lastUpdatedDemand': latest_demand_timestamp.strftime('%d %b %Y, %H:%M') if latest_demand_timestamp else 'N/A',
            'lastUpdatedWeather': latest_weather_timestamp.strftime('%d %b %Y, %H:%M') if latest_weather_timestamp else 'N/A',
            'lastUpdatedHoliday': latest_holiday.strftime('%d %b %Y') if latest_holiday else 'N/A',
        }

        logger.info("Last Updated API Response: %s", response_data)
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error fetching last updated timestamps: {e}", exc_info=True)
        return jsonify({'error': 'Failed to fetch last updated timestamps'}), 500

"""6. END SECTION 6 LAST UPDATED"""

"""7. START SECTION 7 HEALTHCHECK"""

@app.route('/health', methods=['GET'])
def health_check():
    try:
        demand_status = "Available" if not hourly_demand_data.empty else "Not Available"
        weather_status = "Available" if not hourly_weather_data.empty else "Not Available"
        data_dir_status = "Exists" if os.path.exists(base_dir) else "Missing"

        # Verify additional routes
        routes_status = {}
        routes_to_check = ['/', '/api/dashboard', '/api/hourlydemand', '/api/hourlyweather', '/api/holidays']
        for route in routes_to_check:
            try:
                with app.test_request_context(route):
                    response = app.full_dispatch_request()
                    routes_status[route] = "OK" if response.status_code == 200 else "FAILED"
            except Exception as e:
                logger.error(f"Error checking route {route}: {e}", exc_info=True)
                routes_status[route] = "FAILED"

        response_data = {
            "status": "OK",
            "demand_data_status": demand_status,
            "weather_data_status": weather_status,
            "data_dir_status": data_dir_status,
            "routes_status": routes_status,
        }

        logger.info("Health Check Response: %s", response_data)
        return jsonify(response_data), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({"status": "FAILED", "error": str(e)}), 500

"""7. END SECTION 7 HEALTHCHECK"""


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)