from flask import Flask, request, jsonify
import numpy as np
import joblib

MODEL_PATH = "pressure_drop_rf_model.joblib"
METADATA_PATH = "pressure_drop_model_metadata.joblib"

model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)
feature_columns = metadata.get("feature_columns", [
    "pipe_diameter",
    "pipe_length",
    "flow_rate",
    "inlet_pressure",
])

app = Flask(__name__)


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
    prediction = model.predict(features)[0]

    return jsonify({"predicted_pressure_drop": float(prediction)})


if __name__ == "__main__":
    app.run(debug=True)
