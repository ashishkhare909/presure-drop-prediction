from flask import Flask, request, jsonify
import numpy as np
import joblib

MODEL_PATH = "pressure_drop_rf_model.joblib"
METADATA_PATH = "pressure_drop_model_metadata.joblib"

model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)
feature_columns = metadata.get("feature_columns", [
    "diameter",
    "length",
    "bend_angle",
    "velocity",
])

app = Flask(__name__)


def get_risk_and_message(pressure_drop):
    if pressure_drop < 1000:
        return "Low", "System operating normally."
    if pressure_drop < 5000:
        return "Moderate", "Pressure drop is elevated; monitor system performance."
    return "High", "High pressure drop detected; inspect the system."


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    missing_keys = [key for key in feature_columns if key not in data]
    if missing_keys:
        return jsonify({"error": f"Missing keys: {', '.join(missing_keys)}"}), 400

    try:
        values = [float(data[key]) for key in feature_columns]
    except (TypeError, ValueError):
        return jsonify({"error": "All input features must be numeric"}), 400

    features = np.array([values])
    pressure_drop = float(model.predict(features)[0])
    risk, message = get_risk_and_message(pressure_drop)

    return jsonify({
        "pressure_drop": pressure_drop,
        "risk": risk,
        "message": message,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
