from flask import Flask, request, jsonify
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)

# Load models
models = joblib.load("proactive_models.pkl")
scaler_features = joblib.load("scaler_features.pkl")
scaler_price = joblib.load("scaler_price.pkl")

feature_cols_len = 9  # number of features used in feature models

@app.route("/", methods=["GET"])
def home():
    return {"status": "ML server is running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        temp = float(data["temp"])
        humidity = float(data["humidity"])
        gas = float(data["gas"])
        weight = float(data["weight"])
        light = float(data["light"])
        days_stored = int(data["days_stored"])
        current_price = float(data["current_price"])
    except:
        return {"error": "Invalid input format"}, 400

    now = datetime.now()
    hour = now.hour
    day_of_week = now.weekday()

    # Simple spoilage formula (same as your app)
    current_spoilage = min(100, days_stored * 0.8 +
                           (temp - 27) * 2 +
                           (humidity - 67) * 1.5 +
                           (gas - 1200) / 100)
    current_spoilage = max(0, current_spoilage)

    # Build features
    features = np.array([[temp, humidity, gas, weight, light,
                          hour, day_of_week, days_stored, current_spoilage]])

    X_scaled = scaler_features.transform(features)

    # Predictions
    temp_24h = float(models["temp_24h"].predict(X_scaled)[0])
    humid_24h = float(models["humid_24h"].predict(X_scaled)[0])
    spoilage_rate = float(models["spoilage_rate"].predict(X_scaled)[0])
    days_critical = max(0, float(models["days_critical"].predict(X_scaled)[0]))

    # Price model → needs features + current price
    price_input = np.append(features[0], current_price).reshape(1, -1)
    price_scaled = scaler_price.transform(price_input)
    price_7d = float(models["price_7d"].predict(price_scaled)[0])

    return {
        "temp_24h": temp_24h,
        "humid_24h": humid_24h,
        "spoilage_rate": spoilage_rate,
        "days_critical": days_critical,
        "price_7d": price_7d,
        "current_spoilage": current_spoilage
    }
