"""
train_model.py
Fuel Delivery App - ETA Prediction Model Training

Ye script humara dataset (data/fuel_delivery_data.csv) leke ek ML model train karta hai
jo predict karta hai ki fuel delivery me kitna time (ETA) lagega.

Run karne ke liye:
    python train_model.py

Output:
    ml_model/eta_model.pkl        <- trained model file
    ml_model/encoders.pkl         <- text columns ko number me convert karne wala encoder
"""

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ---- Step 1: Dataset load karo ----
print("📂 Dataset load kar rahe hain...")
df = pd.read_csv("data/fuel_delivery_data.csv")
print(f"Total rows: {len(df)}")
print(df.head())

# ---- Step 2: Text columns ko number me convert karo (Encoding) ----
# ML model sirf numbers samajhta hai, isliye "Morning", "Low", "Clear" jaise
# text values ko humein numbers me badalna padega.
print("\n🔄 Text columns ko numbers me convert kar rahe hain...")

categorical_cols = ["time_of_day", "traffic_level", "weather"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col])
    encoders[col] = le  # baad me app me use karne ke liye save karke rakhenge
    print(f"  {col}: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

# ---- Step 3: Features (X) aur Target (y) alag karo ----
feature_cols = [
    "distance_km",
    "fuel_quantity",
    "time_of_day_encoded",
    "traffic_level_encoded",
    "weather_encoded",
]

X = df[feature_cols]
y = df["eta_minutes"]

# ---- Step 4: Data ko Train aur Test me split karo ----
# 80% data model seekhne (train) ke liye, 20% data test/check karne ke liye
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📊 Train rows: {len(X_train)}, Test rows: {len(X_test)}")

# ---- Step 5: Model banao aur train karo ----
print("\n🤖 Model train ho raha hai...")
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ---- Step 6: Model ko test karo (accuracy check) ----
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n✅ Model training complete!")
print(f"   Mean Absolute Error (MAE): {mae:.2f} minutes")
print(f"   (Matlab, average prediction {mae:.2f} minutes ke aas-paas sahi hoti hai)")
print(f"   R2 Score: {r2:.3f} (1.0 = perfect, jitna 1 ke paas utna accha)")

# ---- Step 7: Feature importance dikhao (kaunsa factor sabse zyada matter karta hai) ----
print("\n📌 Kaunsa factor ETA pe sabse zyada asar dalta hai:")
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importance)

# ---- Step 8: Model aur encoders ko save karo ----
os.makedirs("ml_model", exist_ok=True)

with open("ml_model/eta_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("ml_model/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("\n💾 Model save ho gaya: ml_model/eta_model.pkl")
print("💾 Encoders save ho gaye: ml_model/encoders.pkl")
print("\n🎉 Ab ye model app me use karne ke liye ready hai!")
