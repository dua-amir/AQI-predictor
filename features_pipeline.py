import os
import requests
import pandas as pd
import hopsworks
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Windows OS patch
os.environ["HOPSWORKS_BE_RE_TEMP_DIR"] = os.environ.get("TEMP", "C:\\Temp")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") 
CITY = "Islamabad" 
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

def get_coordinates(city, api_key):
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
    res = requests.get(geo_url).json()
    if res:
        return res[0]['lat'], res[0]['lon']
    raise ValueError(f"City '{city}' not found.")

def fetch_aqi_and_weather_pipeline():
    lat, lon = get_coordinates(CITY, OPENWEATHER_API_KEY)
    
    # 1. Fetch Pollution Forecast
    pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    p_res = requests.get(pollution_url).json()
    
    # 2. Fetch Corresponding Weather Forecast
    weather_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    w_res = requests.get(weather_url).json()
    
    weather_map = {item['dt']: item for item in w_res.get("list", [])}
    rows = []
    previous_aqi = None
    
    for item in p_res.get("list", []):
        ts = item.get("dt")
        dt_obj = datetime.fromtimestamp(ts)
        
        w_item = weather_map.get(ts, {})
        temp_val = w_item.get("main", {}).get("temp", 25.0)
        humidity_val = w_item.get("main", {}).get("humidity", 50)
        wind_speed_kmh = w_item.get("wind", {}).get("speed", 3.0) * 3.6
        visibility = w_item.get("visibility", 10000)
        
        components = item.get("components", {})
        pm25 = components.get("pm2_5", 15.0)
        pm10 = components.get("pm10", 20.0)
        no2 = components.get("no2", 10.0)
        
        # OpenWeather mapping formula to generate target AQI (0 to 300 scale)
        real_aqi_target = int(pm25 * 1.1 + pm10 * 0.4 + no2 * 0.1 + 15)
        real_aqi_target = max(10, min(300, real_aqi_target))
        
        # AQI Change Rate derived features
        aqi_change_rate = 0.0 if previous_aqi is None else float(real_aqi_target - previous_aqi)
        previous_aqi = real_aqi_target
        
        stagnation_index = (temp_val + 5.0) / (wind_speed_kmh + 0.5)
        
        rows.append({
            "timestamp": int(ts),
            "city": CITY,
            "hour": int(dt_obj.hour),
            "day": int(dt_obj.day),
            "month": int(dt_obj.month),
            "temperature": float(temp_val),
            "humidity": float(humidity_val),
            "wind_speed": float(wind_speed_kmh),
            "visibility": float(visibility),
            "stagnation_index": float(stagnation_index),
            "aqi_change_rate": float(aqi_change_rate),
            "aqi_target": int(real_aqi_target)
        })
        
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    print("🚀 Running live streaming feature engineering pipeline...")
    data_df = fetch_aqi_and_weather_pipeline()
    
    print("🛰️ Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    
    weather_fg = fs.get_or_create_feature_group(
        name="weather_aqi_fg",
        version=9,  
        primary_key=['timestamp'],
        description="Air Quality Index forecast features mapped strictly via environmental parameters.",
        online_enabled=True
    )
    
    print("📥 Syncing pipeline features with Hopsworks storage...")
    weather_fg.insert(data_df)
    print("🎯 Sync Complete!")