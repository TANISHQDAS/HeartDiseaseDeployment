"""
app.py
------
Task 3: API Development

Flask REST API for Heart Disease Prediction.
Loads the trained model (model.pkl) and serves predictions as JSON.

This module exposes `app`, a plain Flask WSGI application. It is used:
  - directly for local development (`python app.py`)
  - via Procfile/gunicorn for Render deployment
  - imported by api/index.py for Vercel's serverless Python runtime

Author: Abhi Pandey (23BAI10909)
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------
# Load the trained model bundle once, at import/startup time
# ---------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURE_NAMES = bundle["feature_names"]
MODEL_ACCURACY = bundle.get("accuracy")


@app.route("/", methods=["GET"])
def home():
    """Simple landing page confirming the service is live."""
    return jsonify({
        "message": "Heart Disease Prediction API is running.",
        "usage": "POST patient details as JSON to /predict",
        "required_fields": FEATURE_NAMES,
        "model_accuracy": MODEL_ACCURACY
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint - useful for uptime monitors."""
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept patient details as JSON input and return the prediction as JSON.

    Example input:
    {
        "age": 58, "sex": 1, "cp": 0, "trestbps": 128, "chol": 216,
        "fbs": 0, "restecg": 0, "thalach": 131, "exang": 1,
        "oldpeak": 2.2, "slope": 1, "ca": 3, "thal": 3
    }
    """
    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    # Validate that all required fields are present
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return jsonify({
            "error": f"Missing required field(s): {missing}"
        }), 400

    # Validate that all fields are numeric before handing them to the model
    bad_fields = []
    row = {}
    for f in FEATURE_NAMES:
        try:
            row[f] = float(data[f])
        except (TypeError, ValueError):
            bad_fields.append(f)
    if bad_fields:
        return jsonify({
            "error": f"Field(s) must be numeric: {bad_fields}"
        }), 400

    try:
        # Build feature vector in the exact order the model was trained on
        features = pd.DataFrame([[row[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": result,
            "probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
