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
def load_best_model_with_meta():
    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        
        model_list = mr.get_models("aqi_prediction_model")
        if model_list:
            latest_model_meta = sorted(model_list, key=lambda m: m.version)[-1]
            model_dir = latest_model_meta.download()
            loaded_model = joblib.load(os.path.join(model_dir, "model.pkl"))
            
            algo_description = getattr(latest_model_meta, 'description', "Active ML Engine")
            return loaded_model, algo_description, latest_model_meta.version
        
        raise Exception("Registry fallback loop trigger.")
    except Exception as e:
        if os.path.exists("saved_model/model.pkl"):
            return joblib.load("saved_model/model.pkl"), "Robust Ensemble Engine Active", "Local-Cache"
        return None, None, None

model, model_desc, model_version = load_best_model_with_meta()

if model is not None:
    # Safe title rendering matching the active metadata description 
    if "|" in str(model_desc):
        cleaned_title = model_desc.split("|")[-1].strip()
    else:
        cleaned_title = f"{model_desc} (v{model_version})"
        
    st.sidebar.success(cleaned_title)
    st.sidebar.info(f"Model Registry Version: {model_version}")
    
    st.markdown("### Next 3-Days Automated Air Quality Insights")
    
    with st.spinner("Streaming future weather telemetry from OpenWeather API..."):
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(forecast_url).json()
        
        if "list" in res:
            tab1, tab2, tab3 = st.tabs(["Tomorrow", "Day 2", "Day 3"])
            tabs = [tab1, tab2, tab3]
            indices = [8, 16, 24] 
            
            for idx, tab in zip(indices, tabs):
                item = res["list"][idx]
                t = item["main"]["temp"]
                h = item["main"]["humidity"]
                raw_w = item["wind"]["speed"]
                v = item.get("visibility", 10000)
                date_str = item["dt_txt"].split(" ")[0]
                
                # Convert wind speed to km/h
                w_kmh = raw_w * 3.6
                
                simulated_max_temp = t + 6.5 if idx in [8, 16] else t + 2.0
                
                # Feature Engineering Layer Alignment matching Training setup
                stagnation_idx = (simulated_max_temp ** 1.2) / (w_kmh + 0.5)
                
                # Exact sequential features payload construction
                input_df = pd.DataFrame(
                    [[t, simulated_max_temp, h, w_kmh, v, stagnation_idx]], 
                    columns=['temperature', 'max_temperature', 'humidity', 'wind_speed', 'visibility', 'stagnation_index']
                )
                
                pred_aqi = int(model.predict(input_df)[0])
                
                with tab:
                    st.markdown(f"#### Forecast Target Date: **{date_str}**")
                    st.write(f"🔹 **Expected Metrics:** Temp: `{t}°C` | Max Temp: `{simulated_max_temp:.1f}°C` | Humidity: `{h}%` | Wind: `{w_kmh:.1f} km/h`")
                    st.metric(label="Calculated Future AQI", value=f"{pred_aqi}")
                    
                    if pred_aqi <= 50:
                        st.success("**Good:** Air quality is satisfactory, and air pollution poses little or no risk.")
                    elif pred_aqi <= 100:
                        st.info("**Moderate:** Air quality is acceptable; however, some periods might cause health concerns for sensitive groups.")
                    elif pred_aqi <= 150:
                        st.warning("**Unhealthy for Sensitive Groups:** Atmospheric thresholds crossed. Sensitive groups should monitor active exposures.")
                    else:
                        st.error("**Hazardous Level:** Heavy inversion layers trapping pollutants. High particulate mitigation protocols active.")
        else:
            st.error("Failed to parse forecast data timeline segments from OpenWeather API.")

    # Sandbox Playground Engine
    st.markdown("---")
    st.subheader("Experimental Sandbox Console")
    s_temp = st.slider("Testing Base Temperature (°C)", 0.0, 50.0, 25.0)
    s_max_temp = st.slider("Testing Peak Max Temperature (°C)", 0.0, 55.0, 32.0)
    s_humidity = st.slider("Testing Humidity (%)", 5, 100, 45)
    s_wind = st.slider("Testing Wind Speed (km/h)", 0.0, 70.0, 15.0)
    s_visibility = st.slider("Testing Visibility (m)", 1000, 10000, 8000)
    
    s_stagnation = (s_max_temp ** 1.2) / (s_wind + 0.5)

    if st.button("Compute Sandbox Inputs"):
        s_input = pd.DataFrame(
            [[s_temp, s_max_temp, s_humidity, s_wind, s_visibility, s_stagnation]], 
            columns=['temperature', 'max_temperature', 'humidity', 'wind_speed', 'visibility', 'stagnation_index']
        )
        s_pred = int(model.predict(s_input)[0])
        st.metric(label="Sandbox Realtime Predicted AQI", value=f"{s_pred}")
else:
    st.error("Critical MLOps framework failure: Model artifacts missing from registers.")