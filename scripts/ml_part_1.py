
import pandas as pd
import os
from glob import glob

# Print the current working directory
print("Current working directory:", os.getcwd())

# Define directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
merge_dir = os.path.join(data_dir, 'merge')
os.makedirs(merge_dir, exist_ok=True)

# Define file patterns for all years
weather_files = sorted(glob(os.path.join(data_dir, 'weather', '*.csv')))
demand_files = sorted(glob(os.path.join(data_dir, 'demand', '*.csv')))
holiday_files = sorted(glob(os.path.join(data_dir, 'holiday', '*.csv')))

# Unified output file path for all merged years
output_file = os.path.join(merge_dir, 'allyears.csv')

# Load and combine data from all files
def load_and_combine_csv(file_list, parse_date_col):
    combined_df = pd.DataFrame()
    for file in file_list:
        print(f"Processing file: {file}")
        df = pd.read_csv(file, parse_dates=[parse_date_col])
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

# Load individual datasets
weather_data = load_and_combine_csv(weather_files, 'datetime')
demand_data = load_and_combine_csv(demand_files, 'time')
holiday_data = load_and_combine_csv(holiday_files, 'date')

# Ensure consistent datetime formatting
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
demand_data['time'] = pd.to_datetime(demand_data['time'])
holiday_data['date'] = pd.to_datetime(holiday_data['date'])

# Check the last date and time in the existing CSV
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    existing_data = pd.read_csv(output_file)
    existing_data['datetime'] = pd.to_datetime(existing_data['datetime'])
    latest_timestamp = existing_data['datetime'].max()
    print(f"Last merged timestamp: {latest_timestamp}")

    # Filter weather and demand data to start from the next hour
    weather_data = weather_data[weather_data['datetime'] > latest_timestamp]
    demand_data = demand_data[demand_data['time'] > latest_timestamp]

# Proceed only if there's new data to process
if weather_data.empty or demand_data.empty:
    print("No new data to process. Exiting.")
    exit(0)

# Merge weather and demand data on datetime
combined_data = pd.merge(
    demand_data.rename(columns={'time': 'datetime'}),
    weather_data,
    on='datetime',
    how='left'
)

# Merge holiday data
combined_data['date'] = combined_data['datetime'].dt.date
holiday_data['date'] = holiday_data['date'].dt.date
combined_data = pd.merge(combined_data, holiday_data, on='date', how='left')

# Create binary column for holidays
combined_data['is_holiday'] = combined_data['name'].notna().astype(int)
combined_data.drop(columns=['name'], inplace=True)

# Fill missing values
combined_data.ffill(inplace=True)

# Ensure numeric conversion for applicable columns
numeric_cols = combined_data.select_dtypes(include=['float64', 'int64']).columns
combined_data[numeric_cols] = combined_data[numeric_cols].round(3)

# Rename and reorder columns for compatibility with models
if 'value' in combined_data.columns:
    combined_data.rename(columns={'value': 'electric'}, inplace=True)
combined_data.rename(columns={'electric': 'y'}, inplace=True)

# Prepare the dataset with 'ds' column for models
combined_data['ds'] = combined_data['datetime']
final_columns = ['ds', 'y'] + [col for col in combined_data.columns if col not in ['ds', 'y', 'datetime']]
combined_data = combined_data[final_columns]

# Check if the output file exists and is not empty
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    existing_data = pd.read_csv(output_file)
    existing_data['datetime'] = pd.to_datetime(existing_data['datetime'])
    combined_data['datetime'] = pd.to_datetime(combined_data['ds'])

    # Filter new data to append
    new_data = combined_data[combined_data['datetime'] > existing_data['datetime'].max()]
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)

# Save the combined dataset
combined_data.to_csv(output_file, index=False)
print(f"Data has been processed and saved to {output_file}")

# Display sample rows
print(combined_data.head())
print(combined_data.tail())



"""
Data Preprocessing

1. Data Integration
Task: Combines data from three sources:
a. Weather data (weather/{all years}.csv): Contains hourly weather observations.
b. Electricity demand data (demand/{allyears}.csv): Contains hourly electricity consumption values.
c. Holiday data (holiday/{allyears}.csv): Contains information about public holidays.


2. Feature Engineering

Task: Generates new features:
a. Splits datetime into date and hour.
b. Adds a binary column is_holiday to indicate whether a given date is a holiday.

Purpose in ML:
a. Splitting datetime allows models to capture trends or patterns specific to time (e.g., daily or hourly trends).
b. is_holiday helps models learn the potential impact of holidays on electricity demand.

3. Handling Missing Values
Task:
a. Forward-fills (ffill) missing data.
b. Interpolates remaining missing values using linear methods.

4. Data Cleaning and Transformation
Task:
a. Rounds weather-related values to 3 decimal places for consistency.
b. Converts object-type columns to numeric where possible.
c. Renames the value column to electric to make it more descriptive.
d. Reorders columns (date, hour, electric, followed by others).

in MLOps
Primary Use Case: The resulting dataset is likely intended for time series forecasting or regression to predict hourly electricity demand.
Target Variable: electric (hourly electricity demand).
Features: Weather data (temp, feelslike, humidity, etc.), time data (date, hour), and holiday indicator (is_holiday).

"""