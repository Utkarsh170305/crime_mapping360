from flask import Flask, request, jsonify,render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load("models/crime_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define weather categories (must match training one-hot encoding)
weather_conditions = ["Clear", "Rainy", "Cloudy", "Snow", "Storm"]  # Ensure all trained categories are listed

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Ensure all required fields are present
    required_fields = ["latitude", "longitude", "time", "weather", "population_density"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Convert weather to one-hot encoding (match training format)
    weather_encoded = [1 if data["weather"] == w else 0 for w in weather_conditions]

    # Convert categorical 'Population_Density' to numeric
    try:
        population_density_encoded = int(data["population_density"])  # Ensure numeric conversion
    except ValueError:
        return jsonify({"error": "Invalid population_density value"}), 400

    # Prepare features array (must match trained model format)
    features = np.array([[data["latitude"], data["longitude"], data["time"], population_density_encoded] + weather_encoded])

    # Check feature length before scaling
    expected_features = scaler.n_features_in_
    if features.shape[1] != expected_features:
        return jsonify({"error": f"Feature mismatch: Expected {expected_features}, got {features.shape[1]}"}), 400

    # Scale the features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)
    predicted_crime = label_encoder.inverse_transform(prediction)[0]

    return jsonify({"predicted_crime": predicted_crime})

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)