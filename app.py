"""
app.py
Fuel Delivery App - Backend (Flask API)

Ye server 2 kaam karta hai:
1. User ka location + fuel quantity leke, sabse nearby "delivery agent" dhundhta hai
   (agents ka data hum yahin fake bana rahe hain - real app me ye database se aayega)
2. Humara trained ML model use karke ETA (kitne minute lagenge) predict karta hai

Run karne ke liye:
    python app.py

Fir browser me kholo:
    http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import math
import random
from datetime import datetime

app = Flask(__name__)

# ---- Step 1: Trained model aur encoders load karo ----
with open("ml_model/eta_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("ml_model/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

print("✅ Model aur encoders load ho gaye!")

# ---- Step 2: Fake delivery agents ka data (demo ke liye) ----
# Real app me ye database (Firebase/MySQL) se aayega.
# Abhi hum 6 agents ko highway ke aas-paas manually rakh rahe hain (demo ke liye).

AGENTS = [
    {"id": 1, "name": "Agent Ramesh", "lat": 22.7196, "lng": 75.8577},
    {"id": 2, "name": "Agent Suresh", "lat": 22.7500, "lng": 75.8900},
    {"id": 3, "name": "Agent Mahesh", "lat": 22.6900, "lng": 75.8200},
    {"id": 4, "name": "Agent Deepak", "lat": 22.7700, "lng": 75.8000},
    {"id": 5, "name": "Agent Vikram", "lat": 22.6600, "lng": 75.8700},
    {"id": 6, "name": "Agent Anil",   "lat": 22.7300, "lng": 75.9200},
]

def calculate_distance(lat1, lng1, lat2, lng2):
    """
    Do GPS coordinates ke beech ki distance (km me) nikalta hai.
    Isse "Haversine formula" bolte hain - GPS distance nikalne ka standard tarika.
    """
    R = 6371  # Earth ki radius (km me)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def get_time_of_day():
    """Abhi ka time dekh kar Morning/Afternoon/Evening/Night return karta hai."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


# ---- Route 1: Homepage (frontend dikhayega) ----
@app.route("/")
def home():
    return render_template("index.html")


# ---- Route 2: Fuel request API (main logic yahan hai) ----
@app.route("/api/request-fuel", methods=["POST"])
def request_fuel():
    data = request.get_json()

    user_lat = float(data["lat"])
    user_lng = float(data["lng"])
    fuel_quantity = float(data["fuel_quantity"])

    # ---- Sabse nearby agent dhundo ----
    nearest_agent = None
    min_distance = float("inf")

    for agent in AGENTS:
        dist = calculate_distance(user_lat, user_lng, agent["lat"], agent["lng"])
        if dist < min_distance:
            min_distance = dist
            nearest_agent = agent

    # ---- Traffic aur weather (demo ke liye random - real app me API se aayega) ----
    traffic_level = random.choice(["Low", "Medium", "High"])
    weather = random.choice(["Clear", "Clear", "Clear", "Rainy"])  # zyada chance Clear ka
    time_of_day = get_time_of_day()

    # ---- Model ke liye input taiyar karo ----
    time_encoded = encoders["time_of_day"].transform([time_of_day])[0]
    traffic_encoded = encoders["traffic_level"].transform([traffic_level])[0]
    weather_encoded = encoders["weather"].transform([weather])[0]

    features = np.array([[
        min_distance,
        fuel_quantity,
        time_encoded,
        traffic_encoded,
        weather_encoded
    ]])

    # ---- ETA predict karo ----
    predicted_eta = model.predict(features)[0]

    # ---- Price calculate karo (simple logic - demo ke liye) ----
    base_price_per_litre = 100  # rupees (fuel ka price)
    delivery_charge = 50 + (min_distance * 5)  # base + distance ke hisaab se
    if time_of_day == "Night":
        delivery_charge *= 1.3  # night surge
    total_price = (fuel_quantity * base_price_per_litre) + delivery_charge

    # ---- Response bhejo ----
    return jsonify({
        "agent_name": nearest_agent["name"],
        "distance_km": round(min_distance, 2),
        "eta_minutes": round(float(predicted_eta), 1),
        "traffic_level": traffic_level,
        "weather": weather,
        "time_of_day": time_of_day,
        "delivery_charge": round(delivery_charge, 2),
        "total_price": round(total_price, 2),
    })


if __name__ == "__main__":
    app.run(debug=True)
