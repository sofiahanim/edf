import pandas as pd

# Load the dataset
file_path = 'C:/Users/hanim/edf/ingest/weather.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset and the data types of each column
data.head()
data.dtypes

# Function to replace NaNs in numeric columns with the average of the previous three non-null values
def fill_numeric_with_avg(series):
    return series.fillna(series.rolling(window=3, min_periods=1).mean())

# Apply the function to numeric columns
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].apply(fill_numeric_with_avg)

# Replace NaNs in character columns with default strings
character_cols = data.select_dtypes(include=['object']).columns
data[character_cols] = data[character_cols].fillna("none")

# Check if there are any remaining NaNs or nulls in the dataset
remaining_nulls = data.isnull().sum().sum()

data.head(), remaining_nulls

# Correct the typo and replace NaN values in the 'severerisk' column with 0
data['severerisk'] = data['severerisk'].fillna(0)

# Verify if there are any remaining null values in the dataset
final_nulls = data.isnull().sum().sum()

data.head(), final_nulls

# CHECK THAT IT IS ALIGNED WITH DOCUMENTATION

# Validate and adjust the 'cloudcover' column to ensure values are within the 0-100 range
data['cloudcover'] = data['cloudcover'].clip(lower=0, upper=100)

# Define the path for the cleaned dataset
cleaned_file_path = 'C:/Users/hanim/edf/ingest/cleaned_weather.csv'

# Save the cleaned dataset to a CSV file
data.to_csv(cleaned_file_path, index=False)

cleaned_file_path
