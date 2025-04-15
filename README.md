# AI-Powered Bayesian Medical Diagnosis System

A medical diagnosis system that uses Bayesian inference to predict disease probabilities based on patient symptoms.

## Project Structure

```
.
├── data/               # Dataset storage
├── notebooks/          # Jupyter notebooks for EDA
├── src/               # Core logic
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── bayes_utils.py
│   ├── model.py
│   ├── evaluator.py
│   └── cli.py
└── app/               # Web interface
    ├── app.py
    ├── templates/
    └── static/
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your medical datasets in the `data/` directory

## Usage

### Command Line Interface
```bash
python src/cli.py
```

### Web Interface
```bash
python app/app.py
```

## Features

- Bayesian inference for disease diagnosis
- Support for multiple diseases
- Data preprocessing pipeline
- Web and CLI interfaces
- Model evaluation metrics

## License

MIT License 