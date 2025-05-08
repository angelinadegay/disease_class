
def preprocess_ckd_input(input_dict):
    defaults = {
        'age': 50,
        'albumin': 1,
        'sugar': 0,
        'blood_glucose_random': 100,
        'blood_urea': 40,
        'serum_creatinine': 1.2,
        'haemoglobin': 13,
        'packed_cell_volume': 40,
        'white_blood_cell_count': 8000,
        'hypertension': 0
    }
    for key in defaults:
        if key not in input_dict or input_dict[key] == '' or input_dict[key] is None:
            input_dict[key] = defaults[key]
        else:
            try:
                input_dict[key] = float(input_dict[key])
            except ValueError:
                input_dict[key] = defaults[key]
    return input_dict

def preprocess_diabetes_input(input_dict):
    # Base features expected
    required = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    for key in required:
        if key not in input_dict or input_dict[key] in ('', None):
            input_dict[key] = 0.0
        else:
            input_dict[key] = float(input_dict[key])

    # Engineered features
    input_dict['BMI_Age'] = input_dict['BMI'] * input_dict['Age']
    input_dict['Glucose_Insulin_Ratio'] = input_dict['Glucose'] / max(input_dict['Insulin'], 1)
    input_dict['BP_BMI'] = input_dict['BloodPressure'] / max(input_dict['BMI'], 1)
    input_dict['Glucose_Squared'] = input_dict['Glucose'] ** 2
    input_dict['Skin_BMI'] = input_dict['SkinThickness'] / max(input_dict['BMI'], 1)

    return input_dict




def preprocess_alzheimers_input(input_dict):
    defaults = {
        'SleepQuality': 5,
        'CholesterolTriglycerides': 150,
        'BMI': 25,
        'CholesterolTotal': 200,
        'MMSE': 28,
        'CholesterolLDL': 100,
        'CholesterolHDL': 50,
        'ADL': 5,
        'AlcoholConsumption': 1,
        'PhysicalActivity': 3
    }
    for key in defaults:
        if key not in input_dict or input_dict[key] in ('', None):
            input_dict[key] = defaults[key]
        else:
            try:
                input_dict[key] = float(input_dict[key])
            except ValueError:
                input_dict[key] = defaults[key]
    return input_dict

def preprocess_lung_cancer_input(input_dict):
    binary_fields = [
        'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
        'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING',
        'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'
    ]
    input_dict['AGE'] = float(input_dict.get('AGE', 50))

    for key in binary_fields:
        input_dict[key] = int(input_dict.get(key, 0))
    return input_dict

def preprocess_stroke_input(input_dict):
    defaults = {
        'gender': 1,
        'age': 60,
        'hypertension': 0,
        'heart_disease': 0,
        'ever_married': 1,
        'work_type': 2,
        'Residence_type': 1,
        'avg_glucose_level': 100,
        'bmi': 25,
        'smoking_status': 1
    }
    for key in defaults:
        if key not in input_dict or input_dict[key] in ('', None):
            input_dict[key] = defaults[key]
        else:
            try:
                input_dict[key] = float(input_dict[key])
            except:
                input_dict[key] = defaults[key]
    return input_dict

# Extend dispatch function
def preprocess_input(disease, input_dict):
    if disease == 'ckd':
        return preprocess_ckd_input(input_dict)
    elif disease == 'diabetes':
        return preprocess_diabetes_input(input_dict)
    elif disease == 'alzheimers':
        return preprocess_alzheimers_input(input_dict)
    elif disease == 'lung_cancer':
        return preprocess_lung_cancer_input(input_dict)
    elif disease == 'stroke':
        return preprocess_stroke_input(input_dict)
    else:
        for key in input_dict:
            try:
                input_dict[key] = float(input_dict[key])
            except:
                input_dict[key] = 0.0
        return input_dict
