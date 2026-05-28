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
        model_metadata = mr.get_model("aqi_prediction_model", version=1)
        model_dir = model_metadata.download()
        return joblib.load(os.path.join(model_dir, "model.pkl"))
    except Exception as e:
        if os.path.exists("saved_model/model.pkl"):
            return joblib.load("saved_model/model.pkl")
        return None

model = load_best_model()

if model is not None:
    st.sidebar.success("3-Month Champion Model Loaded")
    
    st.markdown("### Next 3-Days Automated Air Quality Insights")
    
    with st.spinner("Streaming future weather telemetry from OpenWeather API..."):
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(forecast_url).json()
        
        if "list" in res:
            # 3 Dedicated separate presentation blocks/tabs
            tab1, tab2, tab3 = st.tabs(["Tomorrow", "Day 2", "Day 3"])
            tabs = [tab1, tab2, tab3]
            indices = [8, 16, 24] # Gaps corresponding to ~24h, ~48h, ~72h offsets
            
            for idx, tab in zip(indices, tabs):
                item = res["list"][idx]
                t = item["main"]["temp"]
                h = item["main"]["humidity"]
                w = item["wind"]["speed"]
                v = item.get("visibility", 10000)
                date_str = item["dt_txt"].split(" ")[0]
                
                # Dynamic prediction input formatting dataframe
                input_df = pd.DataFrame([[t, h, w, v]], columns=['temperature', 'humidity', 'wind_speed', 'visibility'])
                pred_aqi = int(model.predict(input_df)[0])
                
                with tab:
                    st.markdown(f"#### Forecast Target Date: **{date_str}**")
                    st.write(f"🔹 **Expected Metrics:** Temp: `{t}°C` | Humidity: `{h}%` | Wind Vectors: `{w} m/s` | Visibility: `{v}m`")
                    st.metric(label="Calculated Future AQI", value=f"{pred_aqi}")
                    
                    # AQI Safety Category thresholds
                    if pred_aqi <= 100:
                        st.success("🟢 **Good / Moderate:** Ambient air quality is clear and optimal.")
                    elif pred_aqi <= 200:
                        st.warning("🟡 **Unhealthy Warning:** Ambient parameters slightly compromised. Precaution for sensitive profiles.")
                    else:
                        st.error("🔴 **Hazardous Level:** Elevated pollutant mapping indicators. Critical mitigation active.")
        else:
            st.error("Failed to parse forecast data timeline segments from OpenWeather API.")

    # Sandbox playground metrics component
    st.markdown("---")
    st.subheader("🛠️ Experimental Sandbox Console")
    s_temp = st.slider("Testing Temperature (°C)", 0.0, 50.0, 25.0)
    s_humidity = st.slider("Testing Humidity (%)", 10, 100, 50)
    s_wind = st.slider("Testing Wind Speed (m/s)", 0.0, 20.0, 5.0)
    s_visibility = st.slider("Testing Visibility (m)", 1000, 10000, 8000)

    if st.button("Compute Sandbox Inputs"):
        s_input = pd.DataFrame([[s_temp, s_humidity, s_wind, s_visibility]], columns=['temperature', 'humidity', 'wind_speed', 'visibility'])
        s_pred = int(model.predict(s_input)[0])
        st.metric(label="Sandbox Realtime Predicted AQI", value=f"{s_pred}")
else:
    st.error("Critical MLOps framework failure: Model artifacts missing from registers.")