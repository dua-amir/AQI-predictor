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
        
        # Unit conversion to km/h
        wind_kmh = raw_wind_speed * 3.6
        
        # Real-world peak afternoon heatwave simulation
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
        
        # Stagnation Index and Interaction Features
        weather_data["stagnation_index"] = simulated_max_temp / (wind_kmh + 0.1)
        
        # Target Formula calibration to catch heavy 130+ spikes
        weather_data["aqi_target"] = int(
            (humidity_val * 1.4) - 
            (weather_data["visibility"] / 200) + 
            (simulated_max_temp * 2.8) + 
            (wind_kmh * 0.9) + 
            (weather_data["stagnation_index"] * 2.5)
        )
        weather_data["aqi_target"] = max(10, min(500, weather_data["aqi_target"]))
        rows.append(weather_data)

df = pd.DataFrame(rows)

print("Connecting to Hopsworks Cloud...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# Upgraded to Version 5 for Advanced Robust Calibration
weather_fg = fs.get_or_create_feature_group(
    name="weather_aqi_fg",
    version=5,  # <-- Version 5
    primary_key=['timestamp'],
    description="3-Day Future Weather Data blocks with Huber Loss alignment parameters"
)

print(f"Inserting forecast rows into Feature Store...")
weather_fg.insert(df)
print("Feature Store Version 5 updated successfully!")