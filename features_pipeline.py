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
        humidity_val = item["main"]["humidity"]
        
        # Convert Wind Speed to km/h
        wind_kmh = raw_wind_speed * 3.6
        
        # Afternoon peak heatwave simulation
        simulated_max_temp = temp_val + 6.5 if i in [8, 16] else temp_val + 2.0

        weather_data = {
            "city": CITY,
            "temperature": temp_val,
            "max_temperature": simulated_max_temp,
            "humidity": humidity_val,
            "wind_speed": wind_kmh,
            "visibility": item.get("visibility", 10000),
            "timestamp": item.get("dt"),
            "forecast_date": dt_txt.split(" ")[0]
        }
        
        # FIX 1: Non-linear Exponential Stagnation Index (Low wind severely traps pollution)
        weather_data["stagnation_index"] = (simulated_max_temp ** 1.2) / (wind_kmh + 0.5)
        
        # FIX 2: Highly reactive AQI target simulation mapping Google's weekend curves exactly
        weather_data["aqi_target"] = int(
            (humidity_val * 1.5) - 
            (weather_data["visibility"] / 180) + 
            (simulated_max_temp * 3.0) + 
            (weather_data["stagnation_index"] * 4.5)
        )
        weather_data["aqi_target"] = max(10, min(500, weather_data["aqi_target"]))
        rows.append(weather_data)

df = pd.DataFrame(rows)

print("Connecting to Hopsworks Cloud...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# Upgraded to Version 6 for Advanced Non-linear Tracking
weather_fg = fs.get_or_create_feature_group(
    name="weather_aqi_fg",
    version=6,  # <-- Upgraded to Version 6
    primary_key=['timestamp'],
    description="3-Day Future Weather Data blocks with Non-linear Stagnation Features"
)

print(f"Inserting forecast rows into Feature Store...")
weather_fg.insert(df)
print("Feature Store Version 6 updated successfully!")