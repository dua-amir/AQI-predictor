import os
import hopsworks
import joblib
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from dotenv import load_dotenv
from features_pipeline import fetch_aqi_and_weather_pipeline

load_dotenv()
os.environ["HOPSWORKS_BE_RE_TEMP_DIR"] = os.environ.get("TEMP", "C:\\Temp")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

st.set_page_config(page_title="Pearls AQI Engine", layout="wide", page_icon="🌤️")

st.title("🌤️ Pearls AQI Predictor Dashboard")
st.write("Real-time Serverless Machine Learning forecasting system for atmospheric pollution patterns over the next 3 days.")

@st.cache_resource
def download_ml_artifacts():
    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        model_list = mr.get_models("aqi_prediction_model")
        if model_list:
            latest_model = sorted(model_list, key=lambda m: m.version)[-1]
            model_dir = latest_model.download()
            
            loaded_model = joblib.load(os.path.join(model_dir, "model.pkl"))
            feat_imp = joblib.load(os.path.join(model_dir, "feature_importance.pkl"))
            return loaded_model, feat_imp, latest_model.version
    except Exception as e:
        st.warning("⚠️ Registry unavailable. Attempting to fall back to local disk cache.")
        if os.path.exists("saved_model/model.pkl"):
            return joblib.load("saved_model/model.pkl"), joblib.load("saved_model/feature_importance.pkl"), "Local Cache"
    return None, None, None

model, feature_importance, model_version = download_ml_artifacts()

if model is not None:
    st.sidebar.success(f"🤖 Active Engine Version: {model_version}")
    
    with st.spinner("Streaming data matrices..."):
        try:
            df_features = fetch_aqi_and_weather_pipeline()
        except Exception as e:
            st.error(f"Error fetching live API data: {e}")
            df_features = pd.DataFrame()

    if not df_features.empty:
        # Align with features used during the training setup
        input_features = ['hour', 'day', 'month', 'temperature', 'humidity', 'wind_speed', 'visibility', 'stagnation_index', 'aqi_change_rate']
        df_features['predicted_aqi'] = model.predict(df_features[input_features]).astype(int)
        
        df_features['datetime'] = pd.to_datetime(df_features['timestamp'], unit='s')
        df_features['date_label'] = df_features['datetime'].dt.strftime('%A, %b %d')

        tab_predictions, tab_explainability = st.tabs(["🔮 3-Day Forecast Panels", "📊 Feature Importance Breakdown"])
        
        with tab_predictions:
            st.markdown("### Next 3-Days Automated Air Quality Insights")
            
            unique_days = df_features['day'].unique()[:3]
            cols_layout = st.columns(3)
            day_titles = ["📅 Today", "📅 Tomorrow", "📅 Day After Tomorrow"]
            
            for i, target_day in enumerate(unique_days):
                if i >= len(cols_layout):
                    break
                
                day_subset = df_features[df_features['day'] == target_day]
                peak_row = day_subset.loc[day_subset['predicted_aqi'].idxmax()]
                pred_val = int(peak_row['predicted_aqi'])
                readable_date = peak_row['date_label']
                
                with cols_layout[i]:
                    st.metric(label=f"{day_titles[i]} ({readable_date})", value=f"AQI {pred_val}")
                    st.write(f"🌡️ **Temp:** {peak_row['temperature']:.1f}°C | 💧 **Humidity:** {peak_row['humidity']:.0f}%")
                    st.write(f"💨 **Wind:** {peak_row['wind_speed']:.1f} km/h")
                    
                    if pred_val <= 50:
                        st.success("🟢 **Good**\nMinimal health impacts.")
                    elif pred_val <= 100:
                        st.info("🟡 **Moderate**\nAcceptable parameters.")
                    elif pred_val <= 150:
                        st.warning("🟠 **Unhealthy for Sensitive Groups**\nWear a mask if sensitive.")
                    else:
                        st.error("🚨 **HAZARDOUS ATMOSPHERE ALERT**\nHigh levels of air pollution inversion detected.")

            st.markdown("### Forecast Trend Overview (Continuous Timeline)")
            trend_df = pd.DataFrame({
                'Timeline': df_features['datetime'],
                'Predicted AQI': df_features['predicted_aqi']
            })
            st.line_chart(data=trend_df, x='Timeline', y='Predicted AQI')

        with tab_explainability:
            st.markdown("### Model Explanations (Global Importance Metrics)")
            st.write("The chart below illustrates feature influence relative to calculated output targets:")
            
            imp_df = pd.DataFrame({
                'Feature': list(feature_importance.keys()),
                'Importance Score': list(feature_importance.values())
            }).sort_values(by='Importance Score', ascending=False)
            
            chart = alt.Chart(imp_df).mark_bar(color='#4F46E5').encode(
                x=alt.X('Importance Score:Q', title='Absolute Contribution Magnitude'),
                y=alt.Y('Feature:N', sort='-x', title='Predictive Input Parameters')
            ).properties(height=350)
            
            # Replaced deprecated use_container_width with standard structural string configuration parameter
            st.altair_chart(chart, width="stretch")
else:
    st.error("❌ Critical Infrastructure Failure: Missing trained analytical registry model parameters.")