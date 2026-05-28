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

# Forecast API Call (Future offsets le rha hai)
url = f"https://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={OPENWEATHER_API_KEY}&units=metric"
response = requests.get(url).json()

forecast_list = response.get("list", [])
rows = []

# Agle 3 din ka forecast separate blocks mein map karna (Every 24 hours interval)
for i in range(0, 24, 8):  
    if i < len(forecast_list):
        item = forecast_list[i]
        dt_txt = item.get("dt_txt") 
        
        temp_val = item["main"]["temp"]
        wind_val = item["wind"]["speed"]
        
        # FIX 1: Afternoon peak heatwave ko model ke liye simulate karna (Asal temperature peak target)
        simulated_max_temp = temp_val + 6.0 if i in [8, 16] else temp_val + 1.5

        weather_data = {
            "city": CITY,
            "temperature": temp_val,
            "max_temperature": simulated_max_temp,  # <-- Naya Feature 1
            "humidity": item["main"]["humidity"],
            "wind_speed": wind_val,
            "visibility": item.get("visibility", 10000),
            "timestamp": item.get("dt"),
            "forecast_date": dt_txt.split(" ")[0]
        }
        
        # FIX 2: Stagnation Index (Heat waves traps pollution when wind drops)
        weather_data["stagnation_index"] = simulated_max_temp / (wind_val + 0.1)  # <-- Naya Feature 2
        
        # FIX 3: Target formula scaling based on Heatwave and Stagnation Factor
        weather_data["aqi_target"] = int(
            (weather_data["humidity"] * 1.1) - 
            (weather_data["visibility"] / 250) + 
            (weather_data["max_temperature"] * 2.2) + 
            (weather_data["stagnation_index"] * 1.6)
        )
        weather_data["aqi_target"] = max(10, min(500, weather_data["aqi_target"]))
        rows.append(weather_data)

df = pd.DataFrame(rows)

# Connect to Hopsworks Store
print("Connecting to Hopsworks Cloud...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# Version updated to 3 for new schema mapping
weather_fg = fs.get_or_create_feature_group(
    name="weather_aqi_fg",
    version=3,
    primary_key=['timestamp'],
    description="3-Day Future Weather Data blocks with Stagnation and Max Temp Features"
)

print(f"Inserting forecast rows into Feature Store...")
weather_fg.insert(df)
print("Feature Store Version 3 updated successfully!")