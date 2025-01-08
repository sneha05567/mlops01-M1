from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
from constants import CLASS_LABELS, FEATURE_NAMES

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '../model/random_forest_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return "Random Forest Classifier API is Running!"

@app.route('/predict_species', methods=['POST'])
def predict():
    # Parse JSON request
    data = request.get_json()
    features = data.get('features')

    # Ensure all required features are present
    if not all(feature in features for feature in FEATURE_NAMES):
        return jsonify({"error": "Missing required features"}), 400

    # Extract feature values in the correct order
    feature_values = [features[feature] for feature in FEATURE_NAMES]
    feature_array = np.array([feature_values])  # Convert to 2D array for model prediction

    # Make prediction
    prediction = model.predict(feature_array)
    predicted_class_index = int(prediction[0])  # Convert NumPy int to regular int
    predicted_class_label = CLASS_LABELS[predicted_class_index]

    # Return predicted class label
    return jsonify({"predicted_class": predicted_class_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)