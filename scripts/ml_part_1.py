import pandas as pd
import os

# Print the current working directory
print("Current working directory:", os.getcwd())

# Define directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
merge_dir = os.path.join(data_dir, 'merge')
os.makedirs(merge_dir, exist_ok=True)

# Define file paths
weather_file = os.path.join(data_dir, 'weather/2025.csv')
demand_file = os.path.join(data_dir, 'demand/2025.csv')
holiday_file = os.path.join(data_dir, 'holiday/2025.csv')
output_file = os.path.join(merge_dir, '2025.csv')

# Load the data
weather_data = pd.read_csv(weather_file, parse_dates=['datetime'])
demand_data = pd.read_csv(demand_file, parse_dates=['time'])
holiday_data = pd.read_csv(holiday_file, parse_dates=['date'])

# Adjust precision for weather data to 3 decimal places
cols_to_adjust = ['temp', 'feelslike', 'humidity', 'windspeed', 'cloudcover', 'solaradiation', 'precip']
weather_data[cols_to_adjust] = weather_data[cols_to_adjust].round(3)

# Extract date and hour from the datetime columns in weather and demand data
weather_data['date'] = weather_data['datetime'].dt.date
weather_data['hour'] = weather_data['datetime'].dt.strftime('%H')
demand_data['date'] = demand_data['time'].dt.date
demand_data['hour'] = demand_data['time'].dt.strftime('%H')

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

# Exclude 'date' and 'hour' from numeric conversion
numeric_cols = combined_data.select_dtypes(include=['object']).columns
numeric_cols = numeric_cols.drop(['date', 'hour'], errors='ignore')  # Exclude 'date' and 'hour'
combined_data[numeric_cols] = combined_data[numeric_cols].apply(pd.to_numeric, errors='coerce', axis=1)

# Interpolate remaining missing values for numeric columns
combined_data.interpolate(method='linear', inplace=True)

# Ensure 'date' and 'hour' columns are treated correctly
combined_data['date'] = combined_data['date'].astype(str)
combined_data['hour'] = combined_data['hour'].astype(str)

# Rename 'value' column to 'electric'
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
    
    # Combine date and hour into a datetime column for sorting and comparison
    existing_data['datetime'] = pd.to_datetime(existing_data['date'] + ' ' + existing_data['hour'])
    
    # Sort by the combined datetime column
    existing_data.sort_values(by='datetime', inplace=True)
    
    # Identify the latest timestamp in the existing data
    latest_timestamp = existing_data['datetime'].max()
    
    # Ensure combined_data has a datetime column for comparison
    combined_data['datetime'] = pd.to_datetime(combined_data['date'] + ' ' + combined_data['hour'])
    
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



"""
Data Preprocessing

1. Data Integration
Task: Combines data from three sources:
a. Weather data (weather/2025.csv): Contains hourly weather observations.
b. Electricity demand data (demand/2025.csv): Contains hourly electricity consumption values.
c. Holiday data (holiday/2025.csv): Contains information about public holidays.


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