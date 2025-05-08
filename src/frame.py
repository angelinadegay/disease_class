import joblib
import numpy as np
import xgboost as xgb
import os
import pandas as pd
from preprocessing import preprocess_input


# Disease-specific feature lists
DISEASE_FEATURES = {
    'diabetes': [
    'Pregnancies', 
    'Glucose', 
    'BloodPressure', 
    'SkinThickness', 
    'Insulin',
    'BMI', 
    'DiabetesPedigreeFunction', 
    'Age', 
    'BMI_Age',
    'Glucose_Insulin_Ratio', 
    'BP_BMI', 
    'Glucose_Squared', 
    'Skin_BMI'
]
,
   'stroke': [
    'gender', 
    'age', 
    'hypertension', 
    'heart_disease', 
    'ever_married',
    'work_type', 
    'Residence_type', 
    'avg_glucose_level', 
    'bmi',
    'smoking_status'
]
,
    'lung_cancer': [
    'GENDER', 
    'AGE', 
    'SMOKING', 
    'YELLOW_FINGERS', 
    'ANXIETY',
    'PEER_PRESSURE', 
    'CHRONIC DISEASE', 
    'FATIGUE ', 
    'ALLERGY ', 
    'WHEEZING',
    'ALCOHOL CONSUMING', 
    'COUGHING', 
    'SHORTNESS OF BREATH',
    'SWALLOWING DIFFICULTY', 
    'CHEST PAIN'
]
,
    'alzheimers': [
    'SleepQuality',
    'CholesterolTriglycerides',
    'BMI',
    'CholesterolTotal',
    'MMSE',
    'CholesterolLDL',
    'CholesterolHDL',
    'ADL',
    'AlcoholConsumption',
    'PhysicalActivity'
]
,
    'ckd': [
    'age', 'albumin', 'sugar', 'blood_glucose_random', 'blood_urea',
    'serum_creatinine', 'haemoglobin', 'packed_cell_volume',
    'white_blood_cell_count', 'hypertension'
]
}

class DiseasePredictor:
    def __init__(self):
        # Load your saved models here
        self.models = {
            'diabetes': joblib.load('models/diabetes_model.pkl'),
            'ckd': {
                'model': joblib.load('models/new_ckd_update.pkl'),
                'scaler': joblib.load('models/scaler.pkl')
            },
            'stroke': joblib.load('models/stroke_model.pkl'),
            'lung_cancer': joblib.load('models/lung_cancer_model.pkl'),
            'alzheimers': joblib.load('models/alzheimers_model.pkl')
        }

    def predict(self, patient_input):
        predictions = {}

        for disease, model in self.models.items():
            # Get only the features needed for this disease
            feature_names = DISEASE_FEATURES[disease]

            try:
                # Preprocess the input data for this specific disease
                processed_input = preprocess_input(disease, patient_input.copy())
                
                # Log the features we're looking for
                print(f"\nPredicting {disease} with features:")
                print(f"Required features: {feature_names}")
                print(f"Available features: {list(processed_input.keys())}")
                
                # Check for missing features
                missing_features = [f for f in feature_names if f not in processed_input]
                if missing_features:
                    print(f"Missing features for {disease}: {missing_features}")
                    continue

                # For CKD, apply special preprocessing
                if disease == 'ckd':
                    # Get features in correct order
                    features = [processed_input[col] for col in feature_names]
                    # Convert to numpy array and reshape
                    features_array = np.array(features).reshape(1, -1)
                    # Scale the features
                    scaled_features = model['scaler'].transform(features_array)
                    # Make prediction
                    prob = model['model'].predict_proba(scaled_features)[0][1]
                else:
                    # For other diseases, use standard prediction
                    features = [processed_input[col] for col in feature_names]
                    input_array = np.array(features).reshape(1, -1)
                    prob = model.predict_proba(input_array)[0][1]

                print(f"Predicted probability: {prob}")
                predictions[disease] = prob

            except Exception as e:
                print(f"Error predicting {disease}: {str(e)}")
                continue

        return predictions

import random


def generate_random_patient():
    patient_input = {}

    # ==== Common base fields ====
    patient_input['age'] = random.randint(30, 90)
    patient_input['gender'] = random.choice([0, 1])
    patient_input['bmi'] = round(random.uniform(18.0, 40.0), 1)
    patient_input['blood_pressure'] = random.randint(80, 180)
    patient_input['glucose'] = random.randint(70, 250)

    # ==== Diabetes Features ====
    patient_input['Pregnancies'] = random.randint(0, 10)
    patient_input['Glucose'] = random.randint(70, 250)
    patient_input['BloodPressure'] = random.randint(60, 140)
    patient_input['SkinThickness'] = random.uniform(5.0, 40.0)
    patient_input['Insulin'] = random.uniform(15.0, 300.0)
    patient_input['BMI'] = round(random.uniform(18.0, 45.0), 1)
    patient_input['DiabetesPedigreeFunction'] = round(random.uniform(0.1, 2.5), 2)
    patient_input['Age'] = random.randint(20, 80)

    # Engineering diabetes features
    patient_input['BMI_Age'] = patient_input['BMI'] * patient_input['Age']
    patient_input['Glucose_Insulin_Ratio'] = patient_input['Glucose'] / max(patient_input['Insulin'], 1)
    patient_input['BP_BMI'] = patient_input['BloodPressure'] / max(patient_input['BMI'], 1)
    patient_input['Glucose_Squared'] = patient_input['Glucose'] ** 2
    patient_input['Skin_BMI'] = patient_input['SkinThickness'] / max(patient_input['BMI'], 1)

    # ==== Lung Cancer Features ====
    patient_input['GENDER'] = random.choice([0, 1])
    patient_input['AGE'] = random.randint(30, 90)
    patient_input['SMOKING'] = random.choice([0, 1])
    patient_input['YELLOW_FINGERS'] = random.choice([0, 1])
    patient_input['ANXIETY'] = random.choice([0, 1])
    patient_input['PEER_PRESSURE'] = random.choice([0, 1])
    patient_input['CHRONIC DISEASE'] = random.choice([0, 1])
    patient_input['FATIGUE '] = random.choice([0, 1])  # notice trailing space
    patient_input['ALLERGY '] = random.choice([0, 1])
    patient_input['WHEEZING'] = random.choice([0, 1])
    patient_input['ALCOHOL CONSUMING'] = random.choice([0, 1])
    patient_input['COUGHING'] = random.choice([0, 1])
    patient_input['SHORTNESS OF BREATH'] = random.choice([0, 1])
    patient_input['SWALLOWING DIFFICULTY'] = random.choice([0, 1])
    patient_input['CHEST PAIN'] = random.choice([0, 1])

    # ==== Stroke Features ====
    patient_input['hypertension'] = random.choice([0, 1])
    patient_input['heart_disease'] = random.choice([0, 1])
    patient_input['ever_married'] = random.choice([0, 1])
    patient_input['work_type'] = random.choice([0, 1, 2])
    patient_input['Residence_type'] = random.choice([0, 1])
    patient_input['avg_glucose_level'] = round(random.uniform(70.0, 200.0), 1)
    patient_input['bmi'] = round(random.uniform(18.0, 40.0), 1)
    patient_input['smoking_status'] = random.choice([0, 1, 2])

    # ==== Alzheimer's Features ====
    patient_input['SleepQuality'] = random.uniform(0, 10)
    patient_input['CholesterolTriglycerides'] = random.uniform(100, 300)
    patient_input['CholesterolTotal'] = random.uniform(150, 300)
    patient_input['MMSE'] = random.uniform(10, 30)
    patient_input['CholesterolLDL'] = random.uniform(50, 200)
    patient_input['CholesterolHDL'] = random.uniform(30, 90)
    patient_input['ADL'] = random.uniform(0, 10)
    patient_input['AlcoholConsumption'] = random.uniform(0, 20)
    patient_input['PhysicalActivity'] = random.uniform(0, 10)

    # ==== CKD Features ====
    patient_input['albumin'] = round(random.uniform(3.0, 5.0), 1)
    patient_input['sugar'] = round(random.uniform(70.0, 200.0), 1)
    patient_input['blood_glucose_random'] = round(random.uniform(70.0, 200.0), 1)
    patient_input['blood_urea'] = round(random.uniform(10.0, 50.0), 1)
    patient_input['serum_creatinine'] = round(random.uniform(0.5, 2.0), 1)
    patient_input['haemoglobin'] = round(random.uniform(10.0, 18.0), 1)
    patient_input['packed_cell_volume'] = round(random.uniform(35.0, 50.0), 1)
    patient_input['white_blood_cell_count'] = round(random.uniform(4.0, 11.0), 1)
    patient_input['hypertension'] = random.choice([0, 1])

    return patient_input


# Example usage
if __name__ == "__main__":
    random_patient = generate_random_patient()
    predictor = DiseasePredictor()
    results = predictor.predict(random_patient)

    print("\n--- Health Risk Report ---")
    for disease, prob in results.items():
        risk_level = "Low" if prob < 0.3 else "Medium" if prob < 0.6 else "High"
        print(f"{disease.capitalize()} Risk: {risk_level} ({prob:.2%})")
        

def run_batch_prediction(n_patients=100, save_path="batch_patient_predictions.csv"):
    predictor = DiseasePredictor()
    all_patients = []

    for i in range(n_patients):
        random_patient = generate_random_patient()
        results = predictor.predict(random_patient)

        # Merge input features and prediction results
        save_data = random_patient.copy()
        for disease, prob in results.items():
            save_data[f"{disease}_prob"] = prob

        all_patients.append(save_data)

    # Save all patients to a single CSV
    batch_df = pd.DataFrame(all_patients)

    if not os.path.exists(save_path):
        batch_df.to_csv(save_path, index=False)
    else:
        batch_df.to_csv(save_path, mode='a', index=False, header=False)

    print(f"\n✅ Successfully generated and saved {n_patients} random patients to {save_path}!\n")

# --------------- Main Batch Test ---------------
if __name__ == "__main__":
    run_batch_prediction(n_patients=100)

