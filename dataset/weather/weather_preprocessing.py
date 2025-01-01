import pandas as pd

# Load the dataset
file_path = 'C:/Users/hanim/edf/dataset/weather/cleaned_weather.csv'
data = pd.read_csv(file_path)

# Ensure the 'datetime' column is a datetime type
data['datetime'] = pd.to_datetime(data['datetime'])

# Set 'datetime' as the index
data.set_index('datetime', inplace=True)

# Function to replace NaNs in numeric columns with the average of the previous three non-null values
def fill_numeric_with_avg(series):
    return series.fillna(series.rolling(window=3, min_periods=1).mean())

# Apply the function to numeric columns
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].apply(fill_numeric_with_avg)

# Replace NaNs in character columns with default strings
character_cols = data.select_dtypes(include=['object']).columns
data[character_cols] = data[character_cols].fillna("none")

# Correct the typo and replace NaN values in the 'severerisk' column with 0
data['severerisk'] = data['severerisk'].fillna(0)

# Validate and adjust the 'cloudcover' column to ensure values are within the 0-100 range
data['cloudcover'] = data['cloudcover'].clip(lower=0, upper=100)

# Function to check each year for completeness
def check_yearly_completeness(data, start_year, end_year):
    summary = {}
    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        year_data = data.loc[start_date:end_date]
        daily_counts = year_data.resample('D').count()
        incomplete_days = daily_counts[daily_counts.iloc[:, 0] != 24]
        summary[year] = f"{year} has {365 - len(incomplete_days)} complete days with 24 hours each."
        if len(incomplete_days) > 0:
            summary[year] += f" {len(incomplete_days)} days have missing or extra records."
    return summary

# Check completeness from 2019 to 2024
completeness_summary = check_yearly_completeness(data, 2019, 2024)
for year, summary in completeness_summary.items():
    print(summary)

# Save the cleaned dataset to a CSV file
cleaned_file_path = 'C:/Users/hanim/edf/dataset/weather/cleaned_weather_2.csv'
data.to_csv(cleaned_file_path, index=False)

# Output the path to the cleaned dataset
print("Cleaned dataset saved to:", cleaned_file_path)
