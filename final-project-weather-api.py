# Test Visual Crossing Weather API
import json
import requests
import pandas as pd
from io import StringIO


with open('./final-project-config.json') as f:
    config = json.load(f)
location = 'Omaha,NE'
start_date = '2023-04-01'
end_date = '2023-09-30'
API_KEY = config['visual_crossing_api_key']
BASE_URL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline'
url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date}/{end_date}?unitGroup=metric&include=days&key={API_KEY}&contentType=csv&elements=datetime,temp,humidity,precip'

params = {
    'key': API_KEY,
    'locations': location,
    'startDateTime': start_date,
    'endDateTime': end_date,
    'aggregateHours': '24',  # Aggregate daily data
    'unitGroup': 'metric',  
    'contentType': 'json'  # Response format
}

# Make API request
response = requests.get(url)
print(response.status_code)
if response.status_code == 200:
    output_csv_file = StringIO(response.text)
    output_df = pd.read_csv(output_csv_file)
    print(f'Average Temperature: {output_df["temp"].mean()}, Average Precipitation: {output_df["precip"].sum()}, Average Humidity: {output_df["humidity"].mean()}')
else:
    print(f"Error: Request failed with status code {response.status_code}")