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
output_file = os.path.join(merge_dir, 'allyears.csv')  # Unified output for all years

# Load and combine data from all files
def load_and_combine_csv(file_list, parse_date_col):
    combined_df = pd.DataFrame()
    for file in file_list:
        print(f"Processing file: {file}")
        df = pd.read_csv(file, parse_dates=[parse_date_col])
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

weather_data = load_and_combine_csv(weather_files, 'datetime')
demand_data = load_and_combine_csv(demand_files, 'time')
holiday_data = load_and_combine_csv(holiday_files, 'date')

# **Step 1: Check for Matching Dates**
# Extract unique dates from each dataset
weather_dates = set(weather_data['datetime'].dt.date.unique())
demand_dates = set(demand_data['time'].dt.date.unique())
holiday_dates = set(holiday_data['date'].dt.date.unique())


# Check the last date and time in the existing CSV
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    existing_data = pd.read_csv(output_file, parse_dates=[['date', 'hour']])
    existing_data['datetime'] = pd.to_datetime(existing_data['date_hour'], format='%Y-%m-%d %H')
    latest_timestamp = existing_data['datetime'].max()
    print(f"Last merged timestamp: {latest_timestamp}")

    # Filter weather and demand data to start from the next hour
    weather_data = weather_data[weather_data['datetime'] > latest_timestamp]
    demand_data = demand_data[demand_data['time'] > latest_timestamp]

# Proceed only if there's new data to process
if weather_data.empty or demand_data.empty:
    print("No new data to process. Exiting.")
    exit(0)

# **Step 1: Check for Matching Dates and Hours**
# Align both datasets to the last matching date and hour
weather_data['hour'] = weather_data['datetime'].dt.hour
demand_data['hour'] = demand_data['time'].dt.hour

last_common_timestamp = min(weather_data['datetime'].max(), demand_data['time'].max())

# Filter to include only matching dates and hours
weather_data = weather_data[weather_data['datetime'] <= last_common_timestamp]
demand_data = demand_data[demand_data['time'] <= last_common_timestamp]

print(f"Merging data up to: {last_common_timestamp}")

# Adjust precision for weather data to 3 decimal places
cols_to_adjust = ['temp', 'feelslike', 'humidity', 'windspeed', 'cloudcover', 'solaradiation', 'precip']
weather_data[cols_to_adjust] = weather_data[cols_to_adjust].round(3)

# Extract date and hour from the datetime columns in weather and demand data
weather_data['date'] = weather_data['datetime'].dt.date
weather_data['hour'] = weather_data['datetime'].dt.strftime('%H').str.zfill(2)  # Ensure two-digit hour
demand_data['date'] = demand_data['time'].dt.date
demand_data['hour'] = demand_data['time'].dt.strftime('%H').str.zfill(2)  # Ensure two-digit hour

# Drop the original datetime columns as they are no longer needed
weather_data.drop('datetime', axis=1, inplace=True)
demand_data.drop('time', axis=1, inplace=True)

# Merge weather and demand data on date and hour
combined_data = pd.merge(demand_data, weather_data, on=['date', 'hour'], how='left')

# Prepare the holiday data (adjust date format for consistency)
holiday_data['date'] = holiday_data['date'].dt.date
combined_data = pd.merge(combined_data, holiday_data, on='date', how='left')

# Create a binary column indicating whether the day is a holiday
combined_data['is_holiday'] = combined_data['name'].notna().astype(int)

# Drop the 'name' column as it's no longer needed
combined_data.drop('name', axis=1, inplace=True)

# Fill forward any missing values
combined_data.ffill(inplace=True)

# Ensure numeric conversion for applicable columns
# Explicitly exclude non-numeric columns such as 'preciptype'
non_numeric_columns = ['date', 'hour', 'preciptype']
numeric_cols = combined_data.select_dtypes(include=['object']).columns.difference(non_numeric_columns)

# Apply numeric conversion only to the identified numeric columns
combined_data[numeric_cols] = combined_data[numeric_cols].apply(pd.to_numeric, errors='coerce', axis=1)

# Interpolate only numeric columns
numeric_only_cols = combined_data.select_dtypes(include=['number']).columns
combined_data[numeric_only_cols] = combined_data[numeric_only_cols].interpolate(method='linear', inplace=False)

# Ensure 'date', 'hour', and 'preciptype' remain unchanged
combined_data['date'] = combined_data['date'].astype(str)
combined_data['hour'] = combined_data['hour'].astype(str)
# 'preciptype' remains non-numeric and untouched

# Rename 'value' column to 'electric' if present
if 'value' in combined_data.columns:
    combined_data.rename(columns={'value': 'electric'}, inplace=True)

# Reorder columns: date, hour, followed by others
column_order = ['date', 'hour', 'electric'] + [col for col in combined_data.columns if col not in ['date', 'hour', 'electric']]
combined_data = combined_data[column_order]

print(combined_data.head())
print(combined_data.tail())

# Check if the output file exists and is not empty
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    # Load existing data
    existing_data = pd.read_csv(output_file)

    # Handle datetime conversion
    existing_data['date'] = existing_data['date'].fillna('1970-01-01')  # Handle NaN values
    existing_data['hour'] = pd.to_numeric(existing_data['hour'], errors='coerce').fillna(0).astype(int).astype(str).str.zfill(2)
    existing_data['datetime'] = pd.to_datetime(
        existing_data['date'].astype(str) + ' ' + existing_data['hour']
    )

    # Sort by the combined datetime column
    existing_data.sort_values(by='datetime', inplace=True)

    # Identify the latest timestamp in the existing data
    latest_timestamp = existing_data['datetime'].max()

    # Ensure combined_data has a datetime column for comparison
    combined_data['datetime'] = pd.to_datetime(
        combined_data['date'].astype(str) + ' ' + combined_data['hour']
    )

    # Filter new data to append (rows with datetime later than latest_timestamp)
    new_data = combined_data[combined_data['datetime'] > latest_timestamp]

    # Append new data to the existing data
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Drop the temporary datetime column before saving
    combined_data.drop(columns=['datetime'], inplace=True)
else:
    print(f"No existing data found. Creating a new file.")

# Save the combined and appended data
combined_data.to_csv(output_file, index=False)

# Display a success message
print(f"Data has been processed and saved to {output_file}")





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