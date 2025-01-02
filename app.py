from flask import Flask, render_template, request, jsonify
import pandas as pd
from flask_caching import Cache
import holidays

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'  # Simple caching for demonstration; use Redis for production
cache = Cache(app)
cache.init_app(app)

def load_and_normalize_csv(path):
    try:
        df = pd.read_csv(path)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        # Check for missing values and print the summary
        print(f"Missing values in {path}:")
        print(df.isnull().sum())
        return df
    except Exception as e:
        print(f"Failed to load or normalize {path}: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def load_data():
    years = range(2019, 2025)
    try:
        hourly_demand_data = pd.concat([load_and_normalize_csv(f'dataset/electricity/{year}.csv') for year in years])
        hourly_weather_data = pd.concat([load_and_normalize_csv(f'dataset/weather/{year}.csv') for year in years])
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames on failure

    return hourly_demand_data, hourly_weather_data


hourly_demand_data, hourly_weather_data = load_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hourlydemand', methods=['GET'])
@cache.cached(timeout=60*5)  # Cache this endpoint for 5 minutes
def fetch_hourly_demand():
    try:
        start = max(int(request.args.get('start', 0)), 0)
        length = max(int(request.args.get('length', 10)), 1)
        search_value = request.args.get('search[value]', '')

        if search_value:
            filtered_data = hourly_demand_data[hourly_demand_data['time'].str.contains(search_value, na=False, regex=False)]
        else:
            filtered_data = hourly_demand_data

        data_slice = filtered_data.iloc[start:start + length].to_dict(orient='records')
        response = {
            "draw": int(request.args.get('draw', 1)),
            "recordsTotal": len(hourly_demand_data),
            "recordsFiltered": len(filtered_data),
            "data": data_slice
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/hourlyweather', methods=['GET'])
def fetch_hourly_weather():
    try:
        start = max(int(request.args.get('start', 0)), 0)
        length = max(int(request.args.get('length', 10)), 1)
        search_value = request.args.get('search[value]', '')

        # Filter data based on the search value
        filtered_data = hourly_weather_data[hourly_weather_data['datetime'].str.contains(search_value, case=False, na=False)]

        # Slice the data for pagination
        data_slice = filtered_data.iloc[start:start + length].to_dict(orient='records')

        # Dynamically extract column names
        column_names = [{'data': col, 'title': col.replace('_', ' ').title()} for col in filtered_data.columns]

        response = {
            "draw": int(request.args.get('draw', 1)),
            "recordsTotal": len(hourly_weather_data),
            "recordsFiltered": len(filtered_data),
            "columns": column_names,  # Pass column information dynamically
            "data": data_slice
        }

        return jsonify(response)
    except Exception as e:
        print("Error in /hourlyweather:", e)
        return jsonify({"error": "Internal Server Error: " + str(e)}), 500


cal_holidays = holidays.US(state='CA', years=range(2019, 2025))

@app.route('/holidays', methods=['GET'])
def fetch_holidays():
    try:
        start = max(int(request.args.get('start', 0)), 0)
        length = max(int(request.args.get('length', 10)), 1)
        search_value = request.args.get('search[value]', '')

        # Filter holidays based on search value
        filtered_holidays = [
            {'date': str(date), 'name': name} 
            for date, name in cal_holidays.items() 
            if search_value.lower() in name.lower()
        ]

        # Paginate the data
        data_slice = filtered_holidays[start:start + length]

        response = {
            "draw": int(request.args.get('draw', 1)),
            "recordsTotal": len(cal_holidays),
            "recordsFiltered": len(filtered_holidays),
            "data": data_slice
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
