# server.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
from datetime import datetime
import traceback
import os

app = Flask(__name__)

# paths
MODEL_PATH = "proactive_models.pkl"
SCALER_FEATURES_PATH = "scaler_features.pkl"
SCALER_PRICE_PATH = "scaler_price.pkl"

# lazy-loaded globals
_models = None
_scaler_features = None
_scaler_price = None

def load_models_once():
    global _models, _scaler_features, _scaler_price
    if _models is None or _scaler_features is None or _scaler_price is None:
        print("Loading ML models and scalers (this happens only once)...")
        try:
            _models = joblib.load(MODEL_PATH)
            _scaler_features = joblib.load(SCALER_FEATURES_PATH)
            _scaler_price = joblib.load(SCALER_PRICE_PATH)
            print("Models loaded.")
        except Exception as e:
            print("Error loading models:", e)
            traceback.print_exc()
            # keep None so predict endpoint returns an error instead of crashing worker
    return _models, _scaler_features, _scaler_price

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ML server is running", "time": datetime.utcnow().isoformat() + "Z"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # required fields with safe defaults
        temp = float(data.get("temp", 27.0))
        humidity = float(data.get("humidity", 67.0))
        gas = float(data.get("gas", 1200.0))
        weight = float(data.get("weight", 95.0))
        light = float(data.get("light", 2000.0))
        days_stored = int(data.get("days_stored", 0))
        current_price = float(data.get("current_price", 30.0))

        # compute timestamp features
        now = datetime.utcnow()
        hour = now.hour
        day_of_week = now.weekday()

        # quick spoilage heuristic (same as training)
        current_spoilage = min(100, days_stored * 0.8 +
                              (temp - 27) * 2 +
                              (humidity - 67) * 1.5 +
                              (gas - 1200) / 100)
        current_spoilage = max(0.0, current_spoilage)

        # ensure models are loaded lazily
        models, scaler_features, scaler_price = load_models_once()

        if models is None or scaler_features is None or scaler_price is None:
            return jsonify({"success": False, "error": "Models not available on server. Contact admin."}), 503

        # Prepare feature vector consistent with training
        feature_vector = np.array([[temp, humidity, gas, weight, light,
                                    hour, day_of_week, days_stored, current_spoilage]])
        X_scaled = scaler_features.transform(feature_vector)

        # Predictions
        temp_24h = float(models["temp_24h"].predict(X_scaled)[0])
        humid_24h = float(models["humid_24h"].predict(X_scaled)[0])
        spoilage_rate = float(models["spoilage_rate"].predict(X_scaled)[0])
        days_critical = float(models["days_critical"].predict(X_scaled)[0])
        days_critical = max(0.0, days_critical)

        # Price model expects features + current_price
        price_input = np.append(feature_vector[0], current_price).reshape(1, -1)
        price_scaled = scaler_price.transform(price_input)
        price_7d = float(models["price_7d"].predict(price_scaled)[0])

        # action
        action_code = int(models["action"].predict(X_scaled)[0])
        action_map = {0: "WAIT_STORE", 1: "SELL_NOW", 2: "PROCESS", 3: "DISCARD"}
        optimal_action = action_map.get(action_code, "SELL_NOW")

        # alerts
        alerts = []
        if temp_24h > 32:
            alerts.append({'type': 'WARNING', 'message': 'Temperature will increase in next 24 hours', 'action': 'Increase ventilation'})
        if humid_24h > 75:
            alerts.append({'type': 'WARNING', 'message': 'Humidity will increase in next 24 hours', 'action': 'Turn on dehumidifier'})
        if spoilage_rate > 2.0:
            alerts.append({'type': 'CRITICAL', 'message': f'Fast degradation ({spoilage_rate:.1f}%/day)', 'action': 'Intervene immediately'})
        if days_critical < 7:
            alerts.append({'type': 'URGENT', 'message': f'Only {int(days_critical)} days until critical', 'action': 'Plan quick sale'})

        response = {
            'success': True,
            'timestamp': datetime.utcnow().isoformat() + "Z",
            'current_state': {
                'temperature': temp,
                'humidity': humidity,
                'gas_level': gas,
                'weight': weight,
                'spoilage': round(current_spoilage, 2),
                'price': current_price
            },
            'predictions_24h': {
                'temperature': round(temp_24h, 1),
                'humidity': round(humid_24h, 1),
                'temp_change': round(temp_24h - temp, 1),
                'humid_change': round(humid_24h - humidity, 1)
            },
            'quality_forecast': {
                'spoilage_rate_per_day': round(spoilage_rate, 2),
                'days_until_critical': int(days_critical),
                'estimated_spoilage_7d': min(100, round(current_spoilage + (spoilage_rate * 7), 1)),
                'quality_trend': 'deteriorating' if spoilage_rate > 1.5 else 'stable'
            },
            'market_intelligence': {
                'current_price': round(current_price, 2),
                'predicted_price_7d': round(price_7d, 2),
                'price_change': round(price_7d - current_price, 2),
                'trend': 'up' if price_7d > current_price else 'down'
            },
            'optimal_decision': {
                'action': optimal_action,
                'confidence': 85
            },
            'proactive_alerts': alerts
        }

        return jsonify(response), 200

    except Exception as e:
        tb = traceback.format_exc()
        print("Error in /predict:", e)
        print(tb)
        return jsonify({'success': False, 'error': str(e)}), 500

# Optional endpoint to reload models (admin use)
@app.route("/reload-models", methods=["POST"])
def reload_models():
    global _models, _scaler_features, _scaler_price
    _models = None
    _scaler_features = None
    _scaler_price = None
    load_models_once()
    ready = _models is not None
    return jsonify({"reloaded": ready})

if __name__ == "__main__":
    # Debug only locally
    load_models_once()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
