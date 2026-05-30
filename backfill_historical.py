import os
import numpy as np
import pandas as pd
import hopsworks
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
CITY = "Islamabad"

print("Simulating Historical Backfill Data Generation Pipeline...")
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
current_date = start_date

backfill_rows = []
np.random.seed(42)

while current_date <= end_date:
    for hour in range(0, 24, 3):  # 3-hour chunks matching OpenWeather profiles
        timestamp = int(current_date.replace(hour=hour, minute=0, second=0).timestamp())
        
        # Generating dynamic structural variances simulating weather parameters
        temp = float(np.random.uniform(15.0, 38.0))
        humidity = float(np.random.uniform(30, 85))
        wind_speed = float(np.random.uniform(2.0, 25.0))
        visibility = float(np.random.choice([6000.0, 8000.0, 10000.0]))
        stagnation = (temp + 5.0) / (wind_speed + 0.5)
        aqi_change = float(np.random.uniform(-15.0, 15.0))
        
        # Ground truth formulation heavily derived from meteorological factors
        base_aqi = int((stagnation * 8) + (humidity * 0.5) + np.random.normal(50, 15))
        target_aqi = max(10, min(300, base_aqi))
        
        backfill_rows.append({
            "timestamp": timestamp,
            "city": CITY,
            "hour": hour,
            "day": current_date.day,
            "month": current_date.month,
            "temperature": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "visibility": visibility,
            "stagnation_index": stagnation,
            "aqi_change_rate": aqi_change,
            "aqi_target": target_aqi
        })
    current_date += timedelta(days=1)

backfill_df = pd.DataFrame(backfill_rows)

print("Connecting to Hopsworks Store to upload backfill rows...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

weather_fg = fs.get_or_create_feature_group(
    name="weather_aqi_fg",
    version=9,
    primary_key=['timestamp'],
    description="Air Quality Index forecast features mapped strictly via environmental parameters.",
    online_enabled=True
)
weather_fg.insert(backfill_df)
print(f"Successfully backfilled {len(backfill_df)} records into Feature Store!")