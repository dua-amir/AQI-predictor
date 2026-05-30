import os
import hopsworks
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv()

# WINDOWS OS CRASH FIX
os.environ["HOPSWORKS_BE_RE_TEMP_DIR"] = os.environ.get("TEMP", "C:\\Temp")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

print("🔌 Logging into Hopsworks Central Register...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

print("📊 Fetching Feature Group reference...")
fg = fs.get_feature_group(name="weather_aqi_fg", version=8)

# --- ROBUST DATA READING LAYER ---
# Attempts to read from the stable offline store. If Hopsworks is still materializing 
# the Hudi metadata, it instantly falls back to the real-time online store cache.
try:
    print("📥 Attempting to read data matrices from Offline Storage...")
    df = fg.read()
except Exception as e:
    print("🔄 Offline data is still materializing on the server.")
    print("📥 Falling back to real-time Online Storage engine stream...")
    df = fg.read(online=True)

# Define structural features matching features_pipeline.py
feature_cols = [
    'hour', 'day', 'month', 'temperature', 'humidity', 
    'wind_speed', 'visibility', 'pm25', 'pm10', 
    'stagnation_index', 'aqi_change_rate'
]

X = df[feature_cols]
y = df['aqi_target']

# Ensure data types are uniform and clean out any missing entries
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = pd.to_numeric(y, errors='coerce').fillna(0)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models required by the assignment guidelines
candidate_models = {
    "Model_1_Ridge": Ridge(alpha=10.0), # Increased alpha for heavier linear smoothing
    "Model_2_Random_Forest": RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_leaf=4, random_state=42), # Reduced max_depth to avoid overfitting on sudden spikes
    "Model_3_Gradient_Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42) # Added subsample and reduced learning rate for stability
}

best_model_name = None
best_model_obj = None
lowest_rmse = float('inf')
best_metrics = {}

print("\n--- 🏁 Training & Benchmarking Candidate Architectures ---")
for name, model in candidate_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Calculate performance metrics required by guidelines
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    
    print(f"-> {name} Validation Summary | RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}")
    
    # Track the best performer based on Root Mean Squared Error (RMSE)
    if rmse < lowest_rmse:
        lowest_rmse = rmse
        best_model_name = name
        best_model_obj = model
        best_metrics = {"rmse": rmse, "mae": mae, "r2": r2}

print(f"\n🏆 Chosen Winner: {best_model_name} (Lowest RMSE: {lowest_rmse:.4f})")

# --- GLOBAL FEATURE IMPORTANCE (SHAP/LIME INTERPRETABILITY LAYER) ---
print("📊 Calculating input feature importance metrics...")
if hasattr(best_model_obj, "feature_importances_"):
    importances = best_model_obj.feature_importances_
else:
    # Fallback pattern for Linear models using normalized coefficients
    importances = np.abs(best_model_obj.coef_)
    if importances.sum() > 0:
        importances = importances / importances.sum()

importance_dict = {feature_cols[idx]: float(importances[idx]) for idx in range(len(feature_cols))}

# Save model artifacts locally before registry upload
local_dir = "saved_model"
os.makedirs(local_dir, exist_ok=True)
joblib.dump(best_model_obj, f"{local_dir}/model.pkl")
joblib.dump(importance_dict, f"{local_dir}/feature_importance.pkl")
print(f"💾 Saved artifacts locally in ./{local_dir}")

# Register champion model within Hopsworks Model Registry
print("🛰️ Connecting to Hopsworks Model Registry...")
mr = project.get_model_registry()

hw_model = mr.python.create_model(
    name="aqi_prediction_model",
    metrics=best_metrics,
    description=f"Auto-selected suite champion: {best_model_name}. Integrated with feature version 8 metrics."
)
hw_model.save(local_dir)
print("🎯 Champion model uploaded to Hopsworks Model Registry successfully!")