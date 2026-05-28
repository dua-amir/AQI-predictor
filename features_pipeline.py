import os
import requests
import pandas as pd
import hopsworks
from datetime import datetime
from dotenv import load_dotenv

# .env file load karna
load_dotenv()

# WINDOWS OS CRASH FIX
os.environ["HOPSWORKS_BE_RE_TEMP_DIR"] = os.environ.get("TEMP", "C:\\Temp")

# CONFIGURATION FROM ENV
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") 
CITY = "Islamabad" 
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Forecast API Call
url = f"https://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={OPENWEATHER_API_KEY}&units=metric"
response = requests.get(url).json()

forecast_list = response.get("list", [])
rows = []

for i in range(0, 24, 8):  
    if i < len(forecast_list):
        item = forecast_list[i]
        dt_txt = item.get("dt_txt") 
        
        temp_val = item["main"]["temp"]
        raw_wind_speed = item["wind"]["speed"]
        
        # FIX 1: Convert Wind Speed from m/s to km/h to match Google Weather
        wind_kmh = raw_wind_speed * 3.6
        
        # Real-world peak afternoon temperature simulation
        simulated_max_temp = temp_val + 6.5 if i in [8, 16] else temp_val + 2.0

        weather_data = {
            "city": CITY,
            "temperature": temp_val,
            "max_temperature": simulated_max_temp,
            "humidity": item["main"]["humidity"],
            "wind_speed": wind_kmh,  # In km/h now
            "visibility": item.get("visibility", 10000),
            "timestamp": item.get("dt"),
            "forecast_date": dt_txt.split(" ")[0]
        }
        
        # Stagnation factor based on km/h unit calibration
        weather_data["stagnation_index"] = simulated_max_temp / (wind_kmh + 0.1)
        
        # FIX 2: Calibrated AQI target formula capturing dust-storm effects from high wind speeds
        weather_data["aqi_target"] = int(
            (weather_data["humidity"] * 1.2) - 
            (weather_data["visibility"] / 200) + 
            (weather_data["max_temperature"] * 2.5) + 
            (wind_kmh * 0.8) +  # High wind speed contributes to dust/PM10 spikes
            (weather_data["stagnation_index"] * 2.0)
        )
        weather_data["aqi_target"] = max(10, min(500, weather_data["aqi_target"]))
        rows.append(weather_data)

df = pd.DataFrame(rows)

print("Connecting to Hopsworks Cloud...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# Upgraded to Version 4 for unit synchronized schema mapping
weather_fg = fs.get_or_create_feature_group(
    name="weather_aqi_fg",
    version=4,  # <-- Upgraded to Version 4
    primary_key=['timestamp'],
    description="3-Day Future Weather Data blocks with km/h wind metrics"
)

print(f"Inserting forecast rows into Feature Store...")
weather_fg.insert(df)
print("Feature Store Version 4 updated successfully!")