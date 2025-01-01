import pandas as pd

# Load the dataset
file_path = 'C:/Users/hanim/edf/dataset/weather/cleaned_weather_2.csv'
data = pd.read_csv(file_path)

# Convert 'datetime' to datetime type to ensure proper handling of dates
data['datetime'] = pd.to_datetime(data['datetime'])

# Filter columns relevant for electricity demand forecasting
columns_to_keep = [
    'datetime', 'temp', 'feelslike', 'humidity', 'windspeed', 
    'cloudcover', 'solarradiation', 'precip', 'preciptype'
]
data = data[columns_to_keep]

# Export data by year from 2019 to 2024
for year in range(2019, 2025):
    yearly_data = data[data['datetime'].dt.year == year]
    # Generate a file path dynamically based on the year
    file_output_path = f'C:/Users/hanim/edf/dataset/weather/electricity_demand_{year}.csv'
    yearly_data.to_csv(file_output_path, index=False)

print("Files have been saved for each year from 2019 to 2024.")
