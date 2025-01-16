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

# Function to load and combine CSV files
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

# Add 'ds' column for models (datetime with time) and reorder columns
combined_data['ds'] = combined_data['datetime']
final_columns = ['ds', 'y'] + [col for col in combined_data.columns if col not in ['ds', 'y', 'datetime']]
combined_data = combined_data[final_columns]

# Check if the output file exists and is not empty
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    existing_data = pd.read_csv(output_file)
    existing_data['ds'] = pd.to_datetime(existing_data['ds'])

    # Filter new data to append
    new_data = combined_data[combined_data['ds'] > existing_data['ds'].max()]
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)

# Save the combined dataset
combined_data.to_csv(output_file, index=False)
print(f"Data has been processed and saved to {output_file}")

# Display sample rows
print(combined_data.head())
print(combined_data.tail())
