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
        
        weather_data = {
            "city": CITY,
            "temperature": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "wind_speed": item["wind"]["speed"],
            "visibility": item.get("visibility", 10000),
            "timestamp": item.get("dt"),
            "forecast_date": dt_txt.split(" ")[0]
        }
        
        # Simulated target index configuration
        weather_data["aqi_target"] = int((weather_data["humidity"] * 1.5) - (weather_data["visibility"] / 200) + 50)
        rows.append(weather_data)

df = pd.DataFrame(rows)

# Connect to Hopsworks Store
print("Connecting to Hopsworks Cloud...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

weather_fg = fs.get_or_create_feature_group(
    name="weather_aqi_fg",
    version=2,
    primary_key=['timestamp'],
    description="3-Day Future Weather Data blocks with Date indexing"
)

print(f"Inserting forecast rows into Feature Store...")
weather_fg.insert(df)
print("Feature Store updated successfully!")