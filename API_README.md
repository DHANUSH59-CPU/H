# ML Prediction API

A Flask-based REST API for serving machine learning model predictions with monitoring and logging capabilities.

## Features

- **Model Loading**: Automatically loads trained models from the models directory
- **Prediction Endpoint**: REST API for making predictions with input validation
- **Model Management**: Support for multiple models with automatic latest model selection
- **Prediction Logging**: Comprehensive logging of all prediction requests and responses
- **Error Handling**: Robust error handling with detailed error messages
- **Metrics Collection**: Real-time metrics on prediction volume and performance
- **Health Monitoring**: Health check endpoints for system status

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the API service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-18T14:08:21.972372",
  "service": "ML Prediction API"
}
```

### Model Status
```
GET /model/status
```
Returns information about available models and current model.

**Response:**
```json
{
  "status": "healthy",
  "total_models": 4,
  "current_model": {
    "model_id": "random_forest_clf_20250918_135521",
    "model_type": "random_forest_classifier",
    "created_at": "2025-09-18T13:55:21.981608",
    "validation_accuracy": 0.955,
    "feature_names": ["feature_1", "feature_2", "feature_3", "feature_4"]
  },
  "available_models": [...]
}
```

### Make Prediction
```
POST /predict
```
Make predictions using the loaded model.

**Request Body:**
```json
{
  "features": {
    "feature_1": 1.5,
    "feature_2": 2.0,
    "feature_3": 1.2,
    "feature_4": 0.8
  },
  "model_id": "optional_specific_model_id"
}
```

**Response:**
```json
{
  "prediction": "class_a",
  "model_id": "random_forest_clf_20250918_135521",
  "model_type": "random_forest_classifier",
  "confidence": 0.85,
  "response_time_ms": 150,
  "request_id": "req_1726665501972"
}
```

### Get Metrics
```
GET /metrics
```
Returns prediction metrics and recent activity.

**Response:**
```json
{
  "total_predictions": 25,
  "average_response_time_ms": 145.2,
  "model_usage": {
    "random_forest_clf_20250918_135521": 15,
    "logistic_regression_clf_20250918_135522": 10
  },
  "recent_predictions": [...]
}
```

## Usage Examples

### Starting the Server
```bash
python app.py
```
The server will start on `http://localhost:5000` by default.

### Making a Prediction (Python)
```python
import requests

# Prepare input data
data = {
    "features": {
        "feature_1": 1.5,
        "feature_2": 2.0,
        "feature_3": 1.2,
        "feature_4": 0.8
    }
}

# Make prediction
response = requests.post(
    "http://localhost:5000/predict",
    json=data,
    headers={'Content-Type': 'application/json'}
)

if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
else:
    print(f"Error: {response.text}")
```

### Making a Prediction (curl)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature_1": 1.5,
      "feature_2": 2.0,
      "feature_3": 1.2,
      "feature_4": 0.8
    }
  }'
```

## Error Handling

The API provides detailed error messages for various scenarios:

### Missing Features
```json
{
  "error": "Invalid input features",
  "details": ["Missing required features: ['feature_1', 'feature_2']"],
  "expected_features": ["feature_1", "feature_2", "feature_3", "feature_4"],
  "request_id": "req_1726665501972"
}
```

### Invalid JSON
```json
{
  "error": "Request must be JSON",
  "request_id": "req_1726665501972"
}
```

### Model Not Found
```json
{
  "error": "Model not found: invalid_model_id",
  "request_id": "req_1726665501972"
}
```

## Components

### ModelLoader
Handles loading and management of trained ML models:
- Automatic model discovery from the models directory
- Model caching for improved performance
- Input validation against expected features
- Support for multiple model formats

### PredictionLogger
Comprehensive logging of prediction requests:
- Logs all prediction requests and responses
- Tracks response times and model usage
- Provides recent prediction history
- JSON-formatted logs for easy parsing

### Input Validation
Robust input validation ensures data quality:
- Validates required features are present
- Checks for unexpected features
- Validates data types (numeric values)
- Provides detailed error messages

## Configuration

The API uses configuration from `config.py`:

```python
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}

MODEL_CONFIG = {
    'model_save_path': 'models/',
    'model_file_extension': '.joblib'
}

LOGGING_CONFIG = {
    'prediction_log_file': 'predictions.log',
    'log_level': 'INFO'
}
```

## Testing

### Component Tests
```bash
python test_api_components.py
```
Tests individual components without running the server.

### Full API Tests
```bash
python test_api.py --wait
```
Comprehensive tests of all API endpoints (requires server to be running).

### Demo
```bash
python demo_api.py
```
Interactive demonstration of all API features.

## Requirements

- Flask==2.3.3
- scikit-learn==1.3.0
- pandas==2.0.3
- joblib==1.3.2
- numpy==1.24.3
- requests==2.31.0

## Model Format

The API expects models to be saved with joblib and accompanied by metadata files:

```
models/
├── model_id.joblib              # Trained model
└── model_id_metadata.json       # Model metadata
```

Metadata format:
```json
{
  "model_id": "random_forest_clf_20250918_135521",
  "model_type": "random_forest_classifier",
  "feature_names": ["feature_1", "feature_2", "feature_3", "feature_4"],
  "target_name": "target",
  "created_at": "2025-09-18T13:55:21.981608",
  "validation_metrics": {
    "accuracy": 0.955,
    "precision": 0.955,
    "recall": 0.955,
    "f1_score": 0.955
  }
}
```

## Logging

Prediction logs are stored in JSON format:
```json
{
  "timestamp": "2025-09-18T14:08:21.972372",
  "model_id": "random_forest_clf_20250918_135521",
  "input_features": {"feature_1": 1.5, "feature_2": 2.0},
  "prediction": "class_a",
  "confidence": 0.85,
  "response_time_ms": 150,
  "request_id": "req_1726665501972"
}
```

## Performance

- Typical response times: 50-200ms
- Model caching reduces load times
- Concurrent request support
- Configurable timeout handling
- Response time tracking and monitoring