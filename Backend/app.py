import json
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import re
from symptoms_dict import symptom_variant  # Import symptom variations from symptoms_dict.py

app = Flask(__name__)

# Load the models, scalers, and label encoders
disease_model = load_model('disease_prediction_model.keras')  # Load the disease Keras model
stress_model = load_model('stress_prediction_model.keras')  # Load the stress Keras model
disease_scaler = joblib.load('scaler.pkl')  # Load the scaler for disease
stress_scaler = joblib.load('scaler_stress.pkl')  # Load the scaler for stress
label_encoder = joblib.load('label_encoder.pkl')  # Load the label encoder for diseases

# Load the symptom-disease mapping from the JSON file
def load_symptom_disease_mapping():
    with open('symptom_disease_mapping.json', 'r') as f:
        return json.load(f)

# Map stress levels to health conditions
def get_stress_label(stress_level):
    if stress_level == 0:
        return "Hypotension"
    elif stress_level == 1:
        return "Normal"
    elif stress_level == 2:
        return "Hypertension"
    else:
        return "Unknown"

# Function to normalize symptoms from user input
def normalize_symptoms(user_input, symptom_disease_map):
    # Convert the user input to lowercase
    user_input = user_input.lower()

    # Substitute variations in the user input using the symptom_variant dictionary
    for symptom, variations in symptom_variant.items():
        # Iterate over all variations (list) for a given symptom
        for variation in variations:
            # Use regex to substitute the variation with the normalized symptom name
            # The 'variation' is first escaped to safely use in regex, and we ensure that it matches the whole word (with word boundaries)
            user_input = re.sub(r'\b' + re.escape(variation) + r'\b', symptom, user_input)

    # Replace spaces with underscores to match the format in the symptom_disease_map
    user_input = user_input.replace(" ", "_")

    # Extract symptoms from the user input by checking against the symptom map
    detected_symptoms = []
    for symptom in symptom_disease_map:
        if symptom in user_input:
            detected_symptoms.append(symptom)

    return detected_symptoms


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        data = request.get_json()

        # Extract prediction type
        prediction_type = data.get('prediction_type')
        if prediction_type is None:
            return jsonify({'error': 'Missing required field: prediction_type'}), 400

        # Disease Prediction
        if prediction_type == 'disease':
            symptoms = data.get('symptoms')
            if symptoms is None:
                return jsonify({'error': 'Missing symptoms data'}), 400

            # Load the symptom-disease mapping
            symptom_disease_map = load_symptom_disease_mapping()

            # If symptoms are provided as a natural language string, normalize them
            if isinstance(symptoms, str):
                symptoms = normalize_symptoms(symptoms, symptom_disease_map)

            # If there are fewer than 17 symptoms, use the symptom-disease mapping
            if len(symptoms) < 17:
                potential_diseases = set()
                for symptom in symptoms:
                    # Add diseases for each symptom from the mapping
                    if symptom in symptom_disease_map:
                        potential_diseases.update(symptom_disease_map[symptom])

                return jsonify({
                    'potential_diseases_from_map': list(potential_diseases),
                    'symptoms_detected': symptoms,
                    'warning': 'Disease model expects exactly 17 symptoms. Only mapping-based diseases are returned.'
                })

            # If exactly 17 symptoms are provided, use the trained model
            if len(symptoms) != 17:
                return jsonify({'error': 'Disease model expects exactly 17 symptoms.'}), 400

            # Convert the symptoms into a numpy array and scale it
            input_features = np.array([symptoms]).astype(float)
            input_features_scaled = disease_scaler.transform(input_features)

            # Predict the disease using the trained model
            predictions = disease_model.predict(input_features_scaled)
            predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class with the highest probability
            predicted_disease = label_encoder.inverse_transform([predicted_class])[0]

            return jsonify({'predicted_disease': predicted_disease})

        # Stress Prediction
        elif prediction_type == 'stress':
            # Required features for stress prediction
            required_features = ['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP']
            stress_features = [data.get(feature) for feature in required_features]

            if None in stress_features:
                return jsonify({'error': f'Missing one or more required stress features: {required_features}'}), 400

            # Convert the stress features into a numpy array and scale it
            input_features = np.array([stress_features]).astype(float)
            input_features_scaled = stress_scaler.transform(input_features)

            # Predict the stress level using the trained model
            predictions = stress_model.predict(input_features_scaled)
            predicted_stress_level = int(np.argmax(predictions, axis=1)[0])  # Convert to standard Python int
            stress_label = get_stress_label(predicted_stress_level)  # Map stress level to health condition

            return jsonify({'predicted_stress_level': predicted_stress_level,
                            'stress_condition': stress_label})

        # Invalid prediction type
        else:
            return jsonify({'error': 'Invalid prediction type. Please use "disease" or "stress".'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
