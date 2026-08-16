# ⛽ Highway Fuel Delivery App

A smart fuel delivery system for highway travelers — request fuel from your current
location and get it delivered by the nearest agent, with a machine learning model
predicting the estimated time of arrival (ETA).

Built as a final year project combining **Full-Stack Development**, **Machine Learning**,
and **Data Analytics**.

---

## 📌 Problem Statement

On long highway journeys, running out of fuel far from a petrol pump is a common and
stressful problem. This app solves that by letting users request fuel delivery to their
exact location — no need to find a pump.

---

## ✨ Features

- 📍 **Interactive map** — select your location with a single click (Leaflet.js)
- 🤖 **ML-powered ETA prediction** — a Random Forest model predicts delivery time based
  on distance, traffic, weather, time of day, and fuel quantity
- 🚚 **Nearest agent assignment** — automatically finds the closest delivery agent using
  GPS distance calculation
- 💰 **Dynamic pricing** — delivery charge calculated based on distance and time of day
- 📊 **Analytics dashboard** — business insights on demand patterns, peak hours, and
  factors affecting delivery time

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript, Leaflet.js |
| Backend | Flask (Python) |
| Machine Learning | Scikit-learn (Random Forest Regressor) |
| Data Analysis | Pandas, NumPy |
| Dashboard | Streamlit, Matplotlib, Seaborn |

---

## 📁 Project Structure

```
fuel-delivery-app/
├── app.py                    # Flask backend + API
├── generate_data.py          # Synthetic dataset generator
├── train_model.py            # ML model training script
├── dashboard.py               # Streamlit analytics dashboard
├── templates/
│   └── index.html             # Frontend (map + request form)
├── data/
│   └── fuel_delivery_data.csv # Generated dataset
├── ml_model/
│   ├── eta_model.pkl           # Trained ML model
│   └── encoders.pkl            # Label encoders for categorical features
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fuel-delivery-app.git
cd fuel-delivery-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the dataset
```bash
python generate_data.py
```

### 4. Train the ML model
```bash
python train_model.py
```

### 5. Run the web app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

### 6. Run the analytics dashboard (in a separate terminal)
```bash
streamlit run dashboard.py
```
Open **http://localhost:8501** in your browser.

---

## 🤖 Model Performance

- **Mean Absolute Error (MAE):** ~3 minutes
- **R² Score:** ~0.89

The most influential factor on ETA is distance, followed by traffic level.

---

## 🔮 Future Improvements

- Real-time GPS tracking of delivery agents
- Live traffic and weather data via APIs
- User authentication and order history
- Payment gateway integration
- Android/iOS app version

---

## 👤 Author

Final Year CSE Diploma Student — Data Science

---

## 📄 License

This project is created for academic purposes.
