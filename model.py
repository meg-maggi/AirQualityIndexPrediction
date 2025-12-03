import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


# =========================
# 1. Load Dataset
# =========================
data_path = r"C:\Users\meghana\Desktop\AQI_Project\data\aqi_data.csv"
df = pd.read_csv(data_path)

print("Dataset Loaded Successfully!")
print("Columns:", df.columns.tolist())

# =========================
# 2. Clean Data
# =========================
df = df.dropna(subset=["pollutant_min", "pollutant_max", "pollutant_avg"])
df["pollutant_id"] = LabelEncoder().fit_transform(df["pollutant_id"])

# =========================
# 3. Features & Target
# =========================
features = ["pollutant_min", "pollutant_max", "latitude", "longitude", "pollutant_id"]
X = df[features]
y = df["pollutant_avg"]

# =========================
# 4. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. Train Multiple Models
# =========================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

results = {}

print("\n=========================")
print(" Training Models")
print("=========================\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    results[name] = r2
    print(f"{name} R² Score: {r2:.4f}")

# =========================
# 6. Select Best Model
# =========================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_score = results[best_model_name]

print("\n====================================")
print(f" BEST MODEL: {best_model_name} (R² = {best_score:.4f})")
print("====================================")

# =========================
# 7. Save Best Model
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/aqi_model.pkl")

print("\nBest Model Saved as models/aqi_model.pkl")
