"""
generate_data.py
Fuel Delivery App - Synthetic Dataset Generator

Ye script ek fake (synthetic) dataset banata hai jisme fuel delivery requests ka data hota hai.
Isse hum ETA (Estimated Time of Arrival) predict karne wala ML model train karenge.

Run karne ke liye:
    python generate_data.py

Output:
    data/fuel_delivery_data.csv
"""

import numpy as np
import pandas as pd
import os

# Reproducibility ke liye seed fix kar rahe hain (har baar same data banega)
np.random.seed(42)

# Kitni rows ka data chahiye
NUM_ROWS = 1000

# ---- 1. Distance (km) ----
# Highway delivery hai to distance 1 km se 25 km tak rakh rahe hain
distance_km = np.round(np.random.uniform(1, 25, NUM_ROWS), 2)

# ---- 2. Time of Day ----
time_of_day = np.random.choice(
    ["Morning", "Afternoon", "Evening", "Night"],
    size=NUM_ROWS,
    p=[0.3, 0.3, 0.25, 0.15]  # Night thodi kam frequent
)

# ---- 3. Traffic Level ----
traffic_level = np.random.choice(
    ["Low", "Medium", "High"],
    size=NUM_ROWS,
    p=[0.5, 0.35, 0.15]  # Highway hai to zyada traffic kam hota hai
)

# ---- 4. Weather ----
weather = np.random.choice(
    ["Clear", "Rainy"],
    size=NUM_ROWS,
    p=[0.85, 0.15]
)

# ---- 5. Fuel Quantity (litres) ----
fuel_quantity = np.round(np.random.uniform(2, 20, NUM_ROWS), 1)

# ---- 6. ETA Calculation (target variable) ----
# Hum ek formula bana rahe hain jisse ETA realistic lage:
# Base time = distance ke hisaab se (average speed ~40km/h maan rahe hain highway pe)
base_time = (distance_km / 40) * 60  # minutes me

# Traffic ka effect
traffic_extra = pd.Series(traffic_level).map({"Low": 0, "Medium": 5, "High": 12}).values

# Time of day ka effect (night me thoda zyada time lagta hai - kam staff/speed limit)
time_extra = pd.Series(time_of_day).map(
    {"Morning": 0, "Afternoon": 2, "Evening": 4, "Night": 7}
).values

# Weather ka effect
weather_extra = pd.Series(weather).map({"Clear": 0, "Rainy": 8}).values

# Fuel quantity ka thoda effect (zyada fuel = thoda zyada prep time)
fuel_extra = fuel_quantity * 0.3

# Thoda random noise add kar rahe hain taaki data bilkul perfect na lage (real world jaisa)
noise = np.random.normal(0, 3, NUM_ROWS)

eta_minutes = base_time + traffic_extra + time_extra + weather_extra + fuel_extra + noise
eta_minutes = np.round(np.clip(eta_minutes, 5, 90), 1)  # 5 se 90 min ke beech clip kar rahe hain

# ---- DataFrame banao ----
df = pd.DataFrame({
    "distance_km": distance_km,
    "time_of_day": time_of_day,
    "traffic_level": traffic_level,
    "weather": weather,
    "fuel_quantity": fuel_quantity,
    "eta_minutes": eta_minutes
})

# ---- Save karo CSV me ----
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "fuel_delivery_data.csv")
df.to_csv(output_path, index=False)

print(f"✅ Dataset ban gaya! {NUM_ROWS} rows save hui: {output_path}")
print("\nPehli 5 rows ka preview:")
print(df.head())
print("\nColumn info:")
print(df.info())
