import csv
import holidays

def fetch_california_holidays(year):
    ca_holidays = holidays.US(state='CA', years=year)
    return [{'date': str(date), 'name': name} for date, name in sorted(ca_holidays.items())]

def export_holidays_to_csv(year, filename):
    holidays_data = fetch_california_holidays(year)
    # Ensure the filename includes the path to the ingest folder
    filepath = f'ingest/holidays/{filename}'
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['date', 'name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for holiday in holidays_data:
            writer.writerow(holiday)

# Example usage:
year = 2019  # specify the year you want
filename = f'ca_holidays_{year}.csv'  # specify the output file name
export_holidays_to_csv(year, filename)
