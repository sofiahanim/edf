from flask import Flask, render_template, jsonify, request
import pandas as pd
import holidays
from flask_caching import Cache
import logging
import pytz
import os

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'  # Caching for demonstration purposes
cache = Cache(app)
cache.init_app(app)


def save_to_cache(data, filename, folder='cache'):
    """Save data to a CSV file in the cache folder."""
    os.makedirs(folder, exist_ok=True)  # Create cache folder if it doesn't exist
    filepath = os.path.join(folder, filename)
    data.to_csv(filepath, index=False)  # Save as CSV
    return filepath

def load_from_cache(filename, folder='cache'):
    """Load data from a CSV file in the cache folder."""
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)  # Load CSV
    return None


# Setup logging for better error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_normalize_csv(path):
    """Load and normalize CSV files."""
    try:
        df = pd.read_csv(path)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        return df
    except Exception as e:
        logging.error(f"Failed to load or normalize {path}: {e}")
        return pd.DataFrame()


def load_data():
    years = range(2019, 2025)  # Update range as needed
    hourly_demand_data = pd.DataFrame()
    hourly_weather_data = pd.DataFrame()

    for year in years:
        try:
            # Dynamically load and normalize CSVs
            path_demand = f'dataset/electricity/{year}.csv'
            path_weather = f'dataset/weather/{year}.csv'
            data_demand = load_and_normalize_csv(path_demand)
            data_weather = load_and_normalize_csv(path_weather)

            hourly_demand_data = pd.concat([hourly_demand_data, data_demand], ignore_index=True)
            hourly_weather_data = pd.concat([hourly_weather_data, data_weather], ignore_index=True)
        except Exception as e:
            logging.error(f"Error loading data for year {year}: {e}")

    return hourly_demand_data, hourly_weather_data


# Load and preprocess data
hourly_demand_data, hourly_weather_data = load_data()

@app.route('/api/hourlydemand', methods=['GET'])
def fetch_hourly_demand():
    start = int(request.args.get('start', 0))  # Start index for pagination
    length = int(request.args.get('length', 10))  # Number of records per page
    search_value = request.args.get('search[value]', '').lower()  # Search query

    # Sort and filter logic
    df_sorted = hourly_demand_data.sort_values(by='time', ascending=False)  # Sort by latest timestamps
    if search_value:
        mask = (
            df_sorted['time'].astype(str).str.contains(search_value) |
            df_sorted['value'].astype(str).str.contains(search_value)
        )
        filtered_df = df_sorted[mask]
    else:
        filtered_df = df_sorted

    # Pagination logic
    paginated_data = filtered_df.iloc[start:start + length].copy()  # Use `.copy()` to avoid SettingWithCopyWarning

    # Format time as a simple datetime string
    paginated_data['time'] = paginated_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    response = {
            'draw': int(request.args.get('draw', 1)),
            'recordsTotal': len(hourly_demand_data),
            'recordsFiltered': len(filtered_df),
            'data': paginated_data.to_dict(orient='records')
    }
    
    logging.info("API Response: %s", response)
    return jsonify(response)


@app.route('/api/hourlyweather', methods=['GET'])
def fetch_hourly_weather():
    start = int(request.args.get('start', 0))  # Start index for pagination
    length = int(request.args.get('length', 10))  # Number of records per page
    search_value = request.args.get('search[value]', '').lower()  # Search query

    # Sort and filter logic
    df_sorted = hourly_weather_data.sort_values(by='datetime', ascending=False)  # Sort by latest timestamps
    if search_value:
        mask = (
            df_sorted['datetime'].astype(str).str.contains(search_value) |
            df_sorted['temp'].astype(str).str.contains(search_value) |
            df_sorted['feelslike'].astype(str).str.contains(search_value) |
            df_sorted['humidity'].astype(str).str.contains(search_value) |
            df_sorted['windspeed'].astype(str).str.contains(search_value) |
            df_sorted['cloudcover'].astype(str).str.contains(search_value) |
            df_sorted['solarradiation'].astype(str).str.contains(search_value) |
            df_sorted['precip'].astype(str).str.contains(search_value) |
            df_sorted['preciptype'].astype(str).str.contains(search_value)
        )
        filtered_df = df_sorted[mask]
    else:
        filtered_df = df_sorted

    # Pagination logic
    paginated_data = filtered_df.iloc[start:start + length]

    # Format datetime to exclude GMT
    paginated_data['datetime'] = paginated_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Return response in DataTables format
    response = {
        'draw': int(request.args.get('draw', 1)),
        'recordsTotal': len(hourly_weather_data),
        'recordsFiltered': len(filtered_df),
        'data': paginated_data.to_dict(orient='records')
    }
    return jsonify(response)

@app.route('/api/holidays', methods=['GET'])
def fetch_holidays():
    try:
        # Get DataTable parameters
        start = int(request.args.get('start', 0))  # Pagination start
        length = int(request.args.get('length', 10))  # Pagination length
        search_value = request.args.get('search[value]', '')  # Search value
        order_column = int(request.args.get('order[0][column]', 0))  # Sorting column index
        order_dir = request.args.get('order[0][dir]', 'asc')  # Sorting direction

        # Get time range from query params
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')

        # Create a calendar for US holidays
        cal_holidays = holidays.US(state='CA', years=range(2019, 2025))

        # Convert holidays to a list
        holidays_list = [{'date': str(date), 'name': name} for date, name in cal_holidays.items()]

        # Filter by date range
        if start_time and end_time:
            holidays_list = [h for h in holidays_list if start_time <= h['date'] <= end_time]

        # Filter by search value
        if search_value:
            holidays_list = [h for h in holidays_list if search_value.lower() in h['name'].lower() or search_value in h['date']]

        # Sort holidays
        order_key = 'date' if order_column == 0 else 'name'
        reverse = order_dir == 'desc'
        holidays_list = sorted(holidays_list, key=lambda x: x[order_key], reverse=reverse)

        # Paginate the results
        records_total = len(holidays_list)
        holidays_list = holidays_list[start: start + length]

        # Prepare response
        response = {
            "draw": int(request.args.get('draw', 1)),
            "recordsTotal": records_total,
            "recordsFiltered": records_total,
            "data": holidays_list
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    
@app.route('/api/dashboard', methods=['GET'])
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


@app.route('/')
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/hourlydemand')
def hourly_demand_page():
    return render_template('hourly_demand.html')

@app.route('/hourlyweather')
def hourly_weather_page():
    return render_template('hourly_weather.html')

@app.route('/holidays')
def holidays_page():
    return render_template('holidays.html')

@app.route('/eda/demand')
def eda_demand_page():
    cache_file = 'demand_summary.csv'  # Cached file name
    folder = 'cache'  # Cache folder

    # Try loading from cache
    summary = load_from_cache(cache_file, folder)
    if summary is None:
        # Perform calculations if no cache exists
        years = range(2019, 2025)
        demand_data = pd.concat(
            [load_and_normalize_csv(f'dataset/electricity/{year}.csv') for year in years],
            ignore_index=True
        )
        summary = demand_data.groupby(demand_data['time'].dt.date).agg({'value': ['mean', 'max', 'min']}).reset_index()
        summary.columns = ['date', 'mean_demand', 'max_demand', 'min_demand']

        # Save to cache
        save_to_cache(summary, cache_file, folder)

    # Render template
    return render_template('eda_demand.html', summary=summary.to_dict(orient='records'))


@app.route('/eda/weather')
def eda_weather_page():
    # Load and combine weather data for all years
    years = range(2019, 2025)
    weather_data = pd.concat([load_and_normalize_csv(f'dataset/weather/{year}.csv') for year in years], ignore_index=True)
    return render_template(
        'eda_weather.html',
        data=weather_data.to_html(classes='table table-striped', index=False)
    )

@app.route('/eda/holidays')
def eda_holidays_page():
    # Generate a list of holidays for all years
    years = range(2019, 2025)
    cal_holidays = holidays.US(state='CA', years=years)
    holiday_data = pd.DataFrame([{'date': str(date), 'name': name} for date, name in cal_holidays.items()])
    return render_template(
        'eda_holidays.html',
        data=holiday_data.to_html(classes='table table-striped', index=False)
    )


if __name__ == '__main__':
    app.run(debug=True)
