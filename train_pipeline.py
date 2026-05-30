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
os.environ["HOPSWORKS_BE_RE_TEMP_DIR"] = os.environ.get("TEMP", "C:\\Temp")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

print("🔌 Logging into Hopsworks Central Register...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

print("Fetching Feature Group reference...")
fg = fs.get_feature_group(name="weather_aqi_fg", version=9)

try:
    print("Attempting to read data matrices from Offline Storage...")
    df = fg.read()
except Exception as e:
    print("Falling back to real-time Online Storage engine stream...")
    df = fg.read(online=True)

# Explicitly excluded overlapping pm25/pm10 factors to prevent systemic data leaks
feature_cols = [
    'hour', 'day', 'month', 'temperature', 'humidity', 
    'wind_speed', 'visibility', 'stagnation_index', 'aqi_change_rate'
]

X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
y = pd.to_numeric(df['aqi_target'], errors='coerce').fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

candidate_models = {
    "Model_1_Ridge": Ridge(alpha=5.0),
    "Model_2_Random_Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
    "Model_3_Gradient_Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42)
}

best_model_name = None
best_model_obj = None
lowest_rmse = float('inf')
best_metrics = {}

print("\n--- Training & Benchmarking Candidate Architectures ---")
for name, model in candidate_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    
    print(f"-> {name} Validation Summary | RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}")
    
    if rmse < lowest_rmse:
        lowest_rmse = rmse
        best_model_name = name
        best_model_obj = model
        best_metrics = {"rmse": rmse, "mae": mae, "r2": r2}

print(f"\nChosen Winner: {best_model_name} (RMSE: {lowest_rmse:.4f})")

# Feature Importance Calculations
if hasattr(best_model_obj, "feature_importances_"):
    importances = best_model_obj.feature_importances_
else:
    importances = np.abs(best_model_obj.coef_)
    if importances.sum() > 0:
        importances = importances / importances.sum()

importance_dict = {feature_cols[idx]: float(importances[idx]) for idx in range(len(feature_cols))}

local_dir = "saved_model"
os.makedirs(local_dir, exist_ok=True)
joblib.dump(best_model_obj, f"{local_dir}/model.pkl")
joblib.dump(importance_dict, f"{local_dir}/feature_importance.pkl")

print("Connecting to Hopsworks Model Registry...")
mr = project.get_model_registry()
hw_model = mr.python.create_model(
    name="aqi_prediction_model",
    metrics=best_metrics,
    description=f"Clean pipeline suite champion: {best_model_name} optimized without target leak parameters."
)
hw_model.save(local_dir)
print("Champion model uploaded successfully!")