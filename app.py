import streamlit as st
import hopsworks
import joblib
import os
import pandas as pd
import requests
from dotenv import load_dotenv

# .env loading variables
load_dotenv()

os.environ["HOPSWORKS_BE_RE_TEMP_DIR"] = os.environ.get("TEMP", "C:\\Temp")

st.set_page_config(page_title="3-Day Automated AQI Engine", layout="centered", page_icon="🌤️")

st.title("Automated 3-Day Future AQI Predictor")
st.write("This intelligence dashboard runs predictions for the next 3 days using a machine learning model optimized on the last 3 months of historical data trends.")

# ENV CONFIGURATIONS
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") 
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
CITY = "Islamabad"

@st.cache_resource
def load_best_model():
    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        
        # FIX 1: Hum model registry se latest version automatically pull kar rahay hain dynamically
        model_list = mr.get_models("aqi_prediction_model")
        if model_list:
            # Sort by version to get highest/latest model artifact
            latest_model_meta = sorted(model_list, key=lambda m: m.version)[-1]
            st.sidebar.info(f"Cloud Model Version Active: {latest_model_meta.version}")
            model_dir = latest_model_meta.download()
            return joblib.load(os.path.join(model_dir, "model.pkl"))
        
        raise Exception("Registry query sequence matching failed.")
    except Exception as e:
        if os.path.exists("saved_model/model.pkl"):
            st.sidebar.warning("Using local cache model buffer")
            return joblib.load("saved_model/model.pkl")
        return None

model = load_best_model()

if model is not None:
    st.sidebar.success("Heatwave Model Activated")
    
    st.markdown("### Next 3-Days Automated Air Quality Insights")
    
    with st.spinner("Streaming future weather telemetry from OpenWeather API..."):
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(forecast_url).json()
        
        if "list" in res:
            tab1, tab2, tab3 = st.tabs(["Tomorrow", "Day 2", "Day 3"])
            tabs = [tab1, tab2, tab3]
            indices = [8, 16, 24] # ~24h, ~48h, ~72h offsets
            
            for idx, tab in zip(indices, tabs):
                item = res["list"][idx]
                t = item["main"]["temp"]
                h = item["main"]["humidity"]
                w = item["wind"]["speed"]
                v = item.get("visibility", 10000)
                date_str = item["dt_txt"].split(" ")[0]
                
                # Dynamic calculations matching training setup
                simulated_max_temp = t + 6.0 if idx in [8, 16] else t + 1.5
                stagnation_idx = simulated_max_temp / (w + 0.1)
                
                # FIX 2: Features order exactly train_pipeline matching column set
                # columns placement sequential integrity check
                input_df = pd.DataFrame(
                    [[t, simulated_max_temp, h, w, v, stagnation_idx]], 
                    columns=['temperature', 'max_temperature', 'humidity', 'wind_speed', 'visibility', 'stagnation_index']
                )
                
                pred_aqi = int(model.predict(input_df)[0])
                
                with tab:
                    st.markdown(f"#### Forecast Target Date: **{date_str}**")
                    st.write(f"🔹 **Expected Metrics:** Temp: `{t}°C` | Max Temp: `{simulated_max_temp:.1f}°C` | Humidity: `{h}%` | Wind: `{w} m/s`")
                    st.metric(label="Calculated Future AQI", value=f"{pred_aqi}")
                    
                    if pred_aqi <= 100:
                        st.success("**Good / Moderate:** Ambient air quality is clear and optimal.")
                    elif pred_aqi <= 200:
                        st.warning("**Unhealthy Warning:** Heatwave parameters triggering smog buildup. Precaution advised.")
                    else:
                        st.error("**Hazardous Level:** Elevated pollutant vectors trapped due to atmospheric stagnation.")
        else:
            st.error("Failed to parse forecast data timeline segments from OpenWeather API.")

    # Sandbox Playground Engine
    st.markdown("---")
    st.subheader("🛠️ Experimental Sandbox Console")
    s_temp = st.slider("Testing Base Temperature (°C)", 0.0, 50.0, 25.0)
    s_max_temp = st.slider("Testing Peak Max Temperature (°C)", 0.0, 55.0, 32.0)
    s_humidity = st.slider("Testing Humidity (%)", 5, 100, 45)
    s_wind = st.slider("Testing Wind Speed (m/s)", 0.0, 20.0, 4.0)
    s_visibility = st.slider("Testing Visibility (m)", 1000, 10000, 8000)
    
    # Custom interaction scaling for sandbox inputs
    s_stagnation = s_max_temp / (s_wind + 0.1)

    if st.button("Compute Sandbox Inputs"):
        s_input = pd.DataFrame(
            [[s_temp, s_max_temp, s_humidity, s_wind, s_visibility, s_stagnation]], 
            columns=['temperature', 'max_temperature', 'humidity', 'wind_speed', 'visibility', 'stagnation_index']
        )
        s_pred = int(model.predict(s_input)[0])
        st.metric(label="Sandbox Realtime Predicted AQI", value=f"{s_pred}")
else:
    st.error("Critical MLOps framework failure: Model artifacts missing from registers.")