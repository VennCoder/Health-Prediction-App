# Symptom extraction logic
def extract_symptoms(user_input, symptom_list):
    # Preprocess the input
    user_input = user_input.lower().strip()
    
    # Extract symptoms using keyword matching
    detected_symptoms = [symptom for symptom in symptom_list if symptom in user_input]
    
    # Ensure exactly 17 symptoms by padding with "not reported"
    while len(detected_symptoms) < 17:
        detected_symptoms.append("not reported")
    
    return detected_symptoms

symptom_list = ["abdominal pain", "back pain", "acidity", "fever", "headache", "cough"]
