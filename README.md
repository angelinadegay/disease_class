# Multi-Disease Medical Detection System

A comprehensive web-based medical detection system that uses machine learning to assess the risk of multiple diseases including Diabetes, Stroke, Chronic Kidney Disease (CKD), Lung Cancer, and Alzheimer's Disease.

## Features

- **Multi-Disease Risk Assessment**
  - Diabetes Risk Prediction
  - Stroke Risk Prediction
  - Chronic Kidney Disease (CKD) Risk Prediction
  - Lung Cancer Risk Prediction
  - Alzheimer's Disease Risk Prediction

- **User-Friendly Interface**
  - Modern, responsive design
  - Real-time form validation
  - Interactive loading states
  - Clear risk level indicators
  - Detailed input fields with helpful descriptions

- **Comprehensive Health Data Collection**
  - Personal Information
  - Vital Signs
  - Medical History
  - Lifestyle Factors
  - Lab Results
  - Symptoms

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-disease-detection.git
cd multi-disease-detection
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
multi-disease-detection/
├── src/
│   ├── templates/
│   │   └── index.html
│   ├── models/
│   │   ├── diabetes_model.pkl
│   │   ├── stroke_model.pkl
│   │   ├── ckd_model.pkl
│   │   ├── lung_cancer_model.pkl
│   │   └── alzheimers_model.pkl
│   ├── app.py
│   └── frame.py
├── requirements.txt
└── README.md
```

## Running the Application

1. Ensure you're in the project directory and your virtual environment is activated.

2. Start the Flask application:
```bash
python src/app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage Guide

1. **Fill in Patient Information**
   - Enter personal details (age, gender)
   - Input vital signs (blood pressure, glucose levels)
   - Provide medical history
   - Add lifestyle information
   - Include lab results if available

2. **Submit for Analysis**
   - Click the "Analyze Health Data" button
   - Wait for the system to process the information
   - View the risk assessment results

3. **Interpret Results**
   - Each disease risk is displayed with:
     - Risk level (Low/Medium/High)
     - Probability percentage
     - Color-coded indicators

## Model Information

The system uses pre-trained machine learning models for each disease:

- **Diabetes Model**: Predicts diabetes risk based on factors like glucose levels, BMI, and age
- **Stroke Model**: Assesses stroke risk using blood pressure, age, and medical history
- **CKD Model**: Evaluates kidney disease risk using lab results and vital signs
- **Lung Cancer Model**: Predicts lung cancer risk based on symptoms and lifestyle factors
- **Alzheimer's Model**: Assesses Alzheimer's risk using cognitive scores and health metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Support

For support, please open an issue in the GitHub repository or contact the development team.

## Acknowledgments

- Flask web framework
- Bootstrap for UI components
- Font Awesome for icons
- Various medical datasets used for model training 