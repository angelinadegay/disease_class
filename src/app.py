from flask import Flask, render_template, request, jsonify
from frame import DiseasePredictor, generate_random_patient
import json
import time
from functools import lru_cache
import numpy as np
import warnings


# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
predictor = DiseasePredictor()

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

# Cache predictions to avoid redundant calculations
@lru_cache(maxsize=100)
def cached_predict(data_tuple):
    """Cache predictions based on input data."""
    # Convert tuple back to dict, preserving spaces in keys
    data = {k: v for k, v in data_tuple}
    # Convert string values to appropriate types
    for key, value in data.items():
        if value == '1':
            data[key] = 1
        elif value == '0':
            data[key] = 0
        else:
            try:
                data[key] = float(value)
            except (ValueError, TypeError):
                pass
    return predictor.predict(data)

@app.route('/')
def index():
    """Render the main form page."""
    # Get a random patient for testing
    sample_patient = generate_random_patient()
    return render_template('index.html', sample_patient=sample_patient)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return predictions."""
    start_time = time.time()
    
    try:
        # Get form data
        form_data = request.form.to_dict()
        print("\nReceived form data:")
        print(form_data)
        
        # Convert form data to tuple for caching, preserving spaces in keys
        data_tuple = tuple((k, v) for k, v in sorted(form_data.items()))
        
        # Get predictions using cached function
        predictions = cached_predict(data_tuple)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Use the custom encoder to serialize the response
        response_data = {
            'success': True,
            'predictions': predictions,
            'processing_time': round(processing_time, 2)
        }
        
        return json.dumps(response_data, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}
        
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({
            'success': False,
            'errors': [str(e)],
            'processing_time': round(time.time() - start_time, 2)
        })

if __name__ == '__main__':
    app.run(debug=True) 