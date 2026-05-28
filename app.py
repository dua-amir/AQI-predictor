import streamlit as st
import hopsworks
import joblib
import os
import pandas as pd

# Windows Crash Fix
os.environ["HOPSWORKS_BE_RE_TEMP_DIR"] = os.environ.get("TEMP", "C:\\Temp")

st.set_page_config(page_title="AQI Predictor Dashboard", layout="centered", page_icon="🌤️")

st.title("Real-time AQI Predictor Dashboard (MLOps)")
st.write("This interactive dashboard fetches the best trained model architecture from the Hopsworks Cloud Model Registry to predict Air Quality Index (AQI).")

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

@st.cache_resource
def load_best_model():
    """Fetches model from Hopsworks Model Registry or falls back to local save."""
    try:
        print("Connecting to Hopsworks for model retrieval...")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        
        # Get the latest version of registered model
        model_metadata = mr.get_model("aqi_prediction_model", version=1)
        model_dir = model_metadata.download()
        model_path = os.path.join(model_dir, "model.pkl")
        
        st.sidebar.success("Model loaded fresh from Hopsworks Cloud Registry!")
        return joblib.load(model_path)
    except Exception as e:
        st.sidebar.warning("Cloud registry busy. Loading locally cached champion model.")
        # Fallback to locally trained model if cloud fetch fails due to gRPC indexing
        if os.path.exists("saved_model/model.pkl"):
            return joblib.load("saved_model/model.pkl")
        else:
            st.error("No local model found! Please run train_pipeline.py first.")
            return None

# Load the champion model
model = load_best_model()

if model is not None:
    st.markdown("---")
    st.subheader("Configure Weather Parameters")
    
    # Grid columns for sliders to look clean
    col1, col2 = st.columns(2)
    
    with col1:
        temp = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=28.0, step=0.5)
        humidity = st.slider("Humidity (%)", min_value=10, max_value=100, value=55, step=1)
        
    with col2:
        wind = st.slider("Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=4.5, step=0.1)
        visibility = st.slider("Visibility Range (meters)", min_value=1000, max_value=10000, value=8000, step=500)

    st.markdown("### Prediction Engine")
    if st.button("Calculate Expected AQI", type="primary"):
        # Wrap inputs exactly in the dataframe format the models trained on
        input_df = pd.DataFrame([[temp, humidity, wind, visibility]], 
                               columns=['temperature', 'humidity', 'wind_speed', 'visibility'])
        
        # Perform prediction
        prediction = model.predict(input_df)[0]
        aqi_val = int(prediction)
        
        # Metric layout
        st.markdown("#### Results:")
        st.metric(label="Predicted Air Quality Index (AQI)", value=f"{aqi_val}")
        
        # AQI Category Alerts based on international scale
        if aqi_val <= 100:
            st.success("🟢 **Good / Moderate:** The air quality is acceptable and poses little or no risk.")
        elif aqi_val <= 200:
            st.warning("🟡 **Unhealthy for Sensitive Groups:** Members of sensitive groups may experience health effects.")
        else:
            st.error("🔴 **Hazardous / Unhealthy:** Active children and adults, and people with respiratory disease should avoid outdoor exertion.")
else:
    st.error("Pipeline breakdown. Please ensure train_pipeline.py finishes correctly.")