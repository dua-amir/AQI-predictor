import os
import requests
import pandas as pd
import hopsworks
from datetime import datetime

# CONFIGURATION
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") 
CITY = "Islamabad" 
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# 1. Fetch Data from OpenWeather
url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OPENWEATHER_API_KEY}&units=metric"
response = requests.get(url).json()

# AQI Predictor ke liye features nikalna
weather_data = {
    "city": CITY,
    "temperature": response["main"]["temp"],
    "humidity": response["main"]["humidity"],
    "wind_speed": response["wind"]["speed"],
    "visibility": response.get("visibility", 10000),
    "timestamp": int(datetime.now().timestamp())
}

# Dummy Target Variable
weather_data["aqi_target"] = int((weather_data["humidity"] * 1.5) - (weather_data["visibility"] / 200) + 50)

df = pd.DataFrame([weather_data])

# 2. Connect and Upload to Hopsworks Feature Store
print("Connecting to Hopsworks Cloud...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

print("Creating/Fetching Feature Group on Cloud...")
# Create or Get Feature Group
weather_fg = fs.get_or_create_feature_group(
    name="weather_aqi_fg",
    version=1,
    primary_key=['timestamp'],
    description="Weather and simulated AQI dataset"
)

print("Inserting data row...")
weather_fg.insert(df)
print("Data successfully pushed to Hopsworks Feature Store!")