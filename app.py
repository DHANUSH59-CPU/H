"""
Flask API server for ML model predictions.
Provides REST endpoints for model predictions, status, and management.
"""

import os
import json
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

from config import API_CONFIG, LOGGING_CONFIG, MODEL_CONFIG
from models.data_processor import DataProcessor
from models.monitoring import get_metrics_collector
from models.retraining_system import get_retraining_system


@dataclass
class PredictionLog:
    """Log entry for prediction requests."""
    timestamp: str
    model_id: str
    input_features: Dict[str, Any]
    prediction: Any
    confidence: Optional[float]
    response_time_ms: int
    request_id: str


class ModelLoader:
    """
    Handles loading and management of trained ML models.
    """
    
    def __init__(self, model_dir: str = "models/"):
        """
        Initialize ModelLoader.
        
        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = model_dir
        self.loaded_models = {}  # Cache for loaded models
        self.model_metadata = {}  # Cache for model metadata
        self.data_processor = DataProcessor()
        
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in the model directory.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        if not os.path.exists(self.model_dir):
            return models
            
        for filename in os.listdir(self.model_dir):
            if filename.endswith('_metadata.json'):
                try:
                    metadata_path = os.path.join(self.model_dir, filename)
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    model_file = filename.replace('_metadata.json', '.joblib')
                    model_path = os.path.join(self.model_dir, model_file)
                    
                    if os.path.exists(model_path):
                        models.append({
                            'model_id': metadata['model_id'],
                            'model_type': metadata['model_type'],
                            'created_at': metadata['created_at'],
                            'validation_accuracy': metadata.get('validation_metrics', {}).get('accuracy'),
                            'feature_names': metadata['feature_names'],
                            'target_name': metadata['target_name']
                        })
                except Exception as e:
                    logging.warning(f"Error reading model metadata {filename}: {e}")
                    
        return sorted(models, key=lambda x: x['created_at'], reverse=True)
    
    def load_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a specific model by ID.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Tuple of (model_object, metadata_dict)
            
        Raises:
            FileNotFoundError: If model doesn't exist
            ValueError: If model cannot be loaded
        """
        # Check cache first
        if model_id in self.loaded_models:
            return self.loaded_models[model_id], self.model_metadata[model_id]
        
        # Find model files
        model_path = os.path.join(self.model_dir, f"{model_id}.joblib")
        metadata_path = os.path.join(self.model_dir, f"{model_id}_metadata.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Cache the loaded model
            self.loaded_models[model_id] = model
            self.model_metadata[model_id] = metadata
            
            logging.info(f"Model {model_id} loaded successfully")
            return model, metadata
            
        except Exception as e:
            raise ValueError(f"Error loading model {model_id}: {str(e)}")
    
    def get_latest_model(self, model_type: str = None) -> Tuple[str, Any, Dict[str, Any]]:
        """
        Get the most recently created model.
        
        Args:
            model_type: Optional filter by model type (e.g., 'classifier', 'regressor')
            
        Returns:
            Tuple of (model_id, model_object, metadata_dict)
            
        Raises:
            ValueError: If no models are available
        """
        available_models = self.list_available_models()
        
        if not available_models:
            raise ValueError("No models available")
        
        # Filter by model type if specified
        if model_type:
            filtered_models = [m for m in available_models if model_type in m['model_type']]
            if not filtered_models:
                raise ValueError(f"No models of type '{model_type}' available")
            available_models = filtered_models
        
        # Get the most recent model
        latest_model = available_models[0]
        model_id = latest_model['model_id']
        
        model, metadata = self.load_model(model_id)
        return model_id, model, metadata
    
    def validate_input_features(self, input_data: Dict[str, Any], 
                              expected_features: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate input features against expected model features.
        
        Args:
            input_data: Input data dictionary
            expected_features: List of expected feature names
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if all required features are present
        missing_features = set(expected_features) - set(input_data.keys())
        if missing_features:
            errors.append(f"Missing required features: {list(missing_features)}")
        
        # Check for extra features
        extra_features = set(input_data.keys()) - set(expected_features)
        if extra_features:
            errors.append(f"Unexpected features: {list(extra_features)}")
        
        # Check data types and values
        for feature in expected_features:
            if feature in input_data:
                value = input_data[feature]
                if not isinstance(value, (int, float)):
                    try:
                        float(value)  # Try to convert to float
                    except (ValueError, TypeError):
                        errors.append(f"Feature '{feature}' must be numeric, got: {type(value).__name__}")
        
        return len(errors) == 0, errors


class PredictionLogger:
    """
    Handles logging of prediction requests and responses.
    """
    
    def __init__(self, log_file: str = "predictions.log"):
        """
        Initialize PredictionLogger.
        
        Args:
            log_file: Path to prediction log file
        """
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration for predictions."""
        # Create a separate logger for predictions
        self.logger = logging.getLogger('predictions')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def log_prediction(self, prediction_log: PredictionLog):
        """
        Log a prediction request and response.
        
        Args:
            prediction_log: PredictionLog object containing request details
        """
        try:
            log_entry = {
                'timestamp': prediction_log.timestamp,
                'model_id': prediction_log.model_id,
                'input_features': prediction_log.input_features,
                'prediction': prediction_log.prediction,
                'confidence': prediction_log.confidence,
                'response_time_ms': prediction_log.response_time_ms,
                'request_id': prediction_log.request_id
            }
            
            self.logger.info(json.dumps(log_entry))
            
        except Exception as e:
            logging.error(f"Error logging prediction: {e}")
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent prediction logs.
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            List of prediction log dictionaries
        """
        predictions = []
        
        if not os.path.exists(self.log_file):
            return predictions
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Get the last 'limit' lines
            recent_lines = lines[-limit:] if len(lines) > limit else lines
            
            for line in recent_lines:
                try:
                    # Parse the log line
                    # Format: timestamp - log_message
                    parts = line.strip().split(' - ', 1)
                    if len(parts) == 2:
                        log_data = json.loads(parts[1])
                        predictions.append(log_data)
                except json.JSONDecodeError:
                    continue
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error reading prediction logs: {e}")
            return predictions


# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize components
model_loader = ModelLoader(MODEL_CONFIG.get('model_save_path', 'models/'))
prediction_logger = PredictionLogger(LOGGING_CONFIG.get('prediction_log_file', 'predictions.log'))
metrics_collector = get_metrics_collector()
retraining_system = get_retraining_system(metrics_collector=metrics_collector, config=MODEL_CONFIG)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG.get('log_level', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions using the loaded model.
    
    Expected JSON payload:
    {
        "features": {
            "feature_1": value1,
            "feature_2": value2,
            ...
        },
        "model_id": "optional_model_id"
    }
    """
    start_time = datetime.now()
    request_id = f"req_{int(start_time.timestamp() * 1000)}"
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'request_id': request_id
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'features' not in data:
            return jsonify({
                'error': 'Missing required field: features',
                'request_id': request_id
            }), 400
        
        features = data['features']
        model_id = data.get('model_id')
        
        # Load model
        if model_id:
            try:
                model, metadata = model_loader.load_model(model_id)
            except FileNotFoundError:
                return jsonify({
                    'error': f'Model not found: {model_id}',
                    'request_id': request_id
                }), 404
            except ValueError as e:
                return jsonify({
                    'error': f'Error loading model: {str(e)}',
                    'request_id': request_id
                }), 500
        else:
            # Use latest model
            try:
                model_id, model, metadata = model_loader.get_latest_model()
            except ValueError as e:
                return jsonify({
                    'error': f'No models available: {str(e)}',
                    'request_id': request_id
                }), 503
        
        # Validate input features
        expected_features = metadata['feature_names']
        is_valid, validation_errors = model_loader.validate_input_features(features, expected_features)
        
        if not is_valid:
            return jsonify({
                'error': 'Invalid input features',
                'details': validation_errors,
                'expected_features': expected_features,
                'request_id': request_id
            }), 400
        
        # Prepare input data
        input_array = np.array([[features[feature] for feature in expected_features]])
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        
        # Get prediction confidence/probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(input_array)[0]
                confidence = float(max(probabilities))
            except Exception:
                pass
        
        # Calculate response time
        end_time = datetime.now()
        response_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Convert prediction to serializable format
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        elif isinstance(prediction, (np.integer, np.floating)):
            prediction = prediction.item()
        
        # Log the prediction
        prediction_log = PredictionLog(
            timestamp=start_time.isoformat(),
            model_id=model_id,
            input_features=features,
            prediction=prediction,
            confidence=confidence,
            response_time_ms=response_time_ms,
            request_id=request_id
        )
        prediction_logger.log_prediction(prediction_log)
        
        # Log to monitoring system
        metrics_collector.log_prediction(asdict(prediction_log))
        
        # Return response
        response = {
            'prediction': prediction,
            'model_id': model_id,
            'model_type': metadata['model_type'],
            'confidence': confidence,
            'response_time_ms': response_time_ms,
            'request_id': request_id
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        
        end_time = datetime.now()
        response_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'response_time_ms': response_time_ms,
            'request_id': request_id
        }), 500


@app.route('/model/status', methods=['GET'])
def model_status():
    """
    Get status information about available models.
    """
    try:
        available_models = model_loader.list_available_models()
        
        # Get current model info
        current_model = None
        try:
            model_id, _, metadata = model_loader.get_latest_model()
            current_model = {
                'model_id': model_id,
                'model_type': metadata['model_type'],
                'created_at': metadata['created_at'],
                'validation_accuracy': metadata.get('validation_metrics', {}).get('accuracy'),
                'feature_names': metadata['feature_names']
            }
        except ValueError:
            pass
        
        return jsonify({
            'status': 'healthy',
            'total_models': len(available_models),
            'current_model': current_model,
            'available_models': available_models
        }), 200
        
    except Exception as e:
        logging.error(f"Status error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/model/retrain', methods=['POST'])
def retrain_model():
    """
    Trigger manual model retraining.
    
    Expected JSON payload:
    {
        "model_id": "model_identifier",
        "notes": "optional notes about retraining"
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        model_id = data.get('model_id')
        notes = data.get('notes', '')
        
        if not model_id:
            return jsonify({
                'error': 'Missing required field: model_id'
            }), 400
        
        # Trigger manual retraining
        result = retraining_system.trigger_manual_retraining(model_id, notes)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
        
    except Exception as e:
        logging.error(f"Retraining trigger error: {str(e)}")
        return jsonify({
            'error': 'Error triggering retraining',
            'details': str(e)
        }), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get comprehensive monitoring metrics and dashboard data.
    """
    try:
        # Get query parameters
        model_id = request.args.get('model_id')
        hours = request.args.get('hours', 24, type=int)
        
        # Get comprehensive dashboard data from monitoring system
        dashboard_data = metrics_collector.get_dashboard_data(model_id, hours)
        
        return jsonify(dashboard_data), 200
        
    except Exception as e:
        logging.error(f"Metrics error: {str(e)}")
        return jsonify({
            'error': 'Error retrieving metrics',
            'details': str(e)
        }), 500


@app.route('/metrics/drift/<model_id>', methods=['GET'])
def check_drift(model_id):
    """
    Check for data drift in a specific model.
    """
    try:
        hours = request.args.get('hours', 1, type=int)
        
        drift_result = metrics_collector.check_drift(model_id, hours)
        
        if drift_result is None:
            return jsonify({
                'message': 'Insufficient data for drift detection',
                'model_id': model_id
            }), 200
        
        return jsonify({
            'model_id': drift_result.model_id,
            'has_drift': drift_result.has_drift,
            'drift_score': drift_result.drift_score,
            'threshold': drift_result.threshold,
            'feature_drifts': drift_result.feature_drifts,
            'timestamp': drift_result.timestamp.isoformat(),
            'method': drift_result.method
        }), 200
        
    except Exception as e:
        logging.error(f"Drift check error: {str(e)}")
        return jsonify({
            'error': 'Error checking drift',
            'details': str(e)
        }), 500


@app.route('/alerts', methods=['GET'])
def get_alerts():
    """
    Get active alerts for monitoring.
    """
    try:
        model_id = request.args.get('model_id')
        
        alerts = metrics_collector.storage.get_active_alerts(model_id)
        
        return jsonify({
            'alerts': alerts,
            'count': len(alerts)
        }), 200
        
    except Exception as e:
        logging.error(f"Alerts error: {str(e)}")
        return jsonify({
            'error': 'Error retrieving alerts',
            'details': str(e)
        }), 500


@app.route('/alerts/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """
    Mark an alert as resolved.
    """
    try:
        success = metrics_collector.alert_manager.resolve_alert(alert_id)
        
        if success:
            return jsonify({
                'message': 'Alert resolved successfully',
                'alert_id': alert_id
            }), 200
        else:
            return jsonify({
                'error': 'Failed to resolve alert',
                'alert_id': alert_id
            }), 500
        
    except Exception as e:
        logging.error(f"Alert resolution error: {str(e)}")
        return jsonify({
            'error': 'Error resolving alert',
            'details': str(e)
        }), 500


@app.route('/retraining/status', methods=['GET'])
def retraining_status():
    """
    Get status of automated retraining system.
    """
    try:
        status = retraining_system.get_retraining_status()
        return jsonify(status), 200
        
    except Exception as e:
        logging.error(f"Retraining status error: {str(e)}")
        return jsonify({
            'error': 'Error retrieving retraining status',
            'details': str(e)
        }), 500


@app.route('/model/versions/<model_id>', methods=['GET'])
def get_model_versions(model_id):
    """
    Get version history for a specific model.
    """
    try:
        versions = retraining_system.version_manager.get_version_history(model_id)
        
        versions_data = []
        for version in versions:
            versions_data.append({
                'version_id': version.version_id,
                'version_number': version.version_number,
                'created_at': version.created_at.isoformat(),
                'is_active': version.is_active,
                'deployment_status': version.deployment_status,
                'validation_metrics': version.validation_metrics,
                'training_metrics': version.training_metrics,
                'parent_version': version.parent_version,
                'notes': version.notes
            })
        
        return jsonify({
            'model_id': model_id,
            'versions': versions_data,
            'total_versions': len(versions_data)
        }), 200
        
    except Exception as e:
        logging.error(f"Model versions error: {str(e)}")
        return jsonify({
            'error': 'Error retrieving model versions',
            'details': str(e)
        }), 500


@app.route('/model/versions/<version_id>/activate', methods=['POST'])
def activate_model_version(version_id):
    """
    Activate (deploy) a specific model version.
    """
    try:
        success = retraining_system.version_manager.activate_version(version_id)
        
        if success:
            return jsonify({
                'message': f'Version {version_id} activated successfully',
                'version_id': version_id
            }), 200
        else:
            return jsonify({
                'error': f'Failed to activate version {version_id}'
            }), 500
        
    except Exception as e:
        logging.error(f"Version activation error: {str(e)}")
        return jsonify({
            'error': 'Error activating version',
            'details': str(e)
        }), 500


@app.route('/model/versions/compare', methods=['POST'])
def compare_model_versions():
    """
    Compare two model versions.
    
    Expected JSON payload:
    {
        "version_1": "version_id_1",
        "version_2": "version_id_2"
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        version_1 = data.get('version_1')
        version_2 = data.get('version_2')
        
        if not version_1 or not version_2:
            return jsonify({
                'error': 'Missing required fields: version_1, version_2'
            }), 400
        
        comparison = retraining_system.version_manager.compare_versions(version_1, version_2)
        
        return jsonify(comparison), 200
        
    except Exception as e:
        logging.error(f"Version comparison error: {str(e)}")
        return jsonify({
            'error': 'Error comparing versions',
            'details': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Basic health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'ML Prediction API',
        'version': '1.0.0'
    }), 200


@app.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """
    Detailed health check endpoint with system status.
    """
    try:
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'ML Prediction API',
            'version': '1.0.0',
            'components': {}
        }
        
        # Check model availability
        try:
            available_models = model_loader.list_available_models()
            current_model = None
            try:
                model_id, _, metadata = model_loader.get_latest_model()
                current_model = {
                    'model_id': model_id,
                    'model_type': metadata['model_type'],
                    'created_at': metadata['created_at']
                }
            except ValueError:
                pass
            
            health_data['components']['models'] = {
                'status': 'healthy' if available_models else 'warning',
                'total_models': len(available_models),
                'current_model': current_model
            }
        except Exception as e:
            health_data['components']['models'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Check monitoring system
        try:
            # Test metrics collector
            dashboard_data = metrics_collector.get_dashboard_data(hours=1)
            health_data['components']['monitoring'] = {
                'status': 'healthy',
                'predictions_tracked': dashboard_data.get('total_predictions', 0)
            }
        except Exception as e:
            health_data['components']['monitoring'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Check retraining system
        try:
            retraining_status = retraining_system.get_retraining_status()
            health_data['components']['retraining'] = {
                'status': 'healthy',
                'is_monitoring': retraining_status.get('is_monitoring', False)
            }
        except Exception as e:
            health_data['components']['retraining'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Check file system access
        try:
            model_dir_exists = os.path.exists(MODEL_CONFIG.get('model_save_path', 'models/'))
            log_file_writable = True
            try:
                log_file = LOGGING_CONFIG.get('prediction_log_file', 'predictions.log')
                with open(log_file, 'a') as f:
                    pass  # Test write access
            except Exception:
                log_file_writable = False
            
            health_data['components']['filesystem'] = {
                'status': 'healthy' if model_dir_exists and log_file_writable else 'warning',
                'model_directory': model_dir_exists,
                'log_file_writable': log_file_writable
            }
        except Exception as e:
            health_data['components']['filesystem'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Determine overall status
        component_statuses = [comp['status'] for comp in health_data['components'].values()]
        if 'error' in component_statuses:
            health_data['status'] = 'unhealthy'
        elif 'warning' in component_statuses:
            health_data['status'] = 'degraded'
        
        status_code = 200 if health_data['status'] == 'healthy' else 503
        return jsonify(health_data), status_code
        
    except Exception as e:
        logging.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'service': 'ML Prediction API',
            'error': str(e)
        }), 500


@app.route('/health/readiness', methods=['GET'])
def readiness_check():
    """
    Readiness check endpoint - checks if service is ready to serve requests.
    """
    try:
        # Check if we have at least one model available
        available_models = model_loader.list_available_models()
        
        if not available_models:
            return jsonify({
                'status': 'not_ready',
                'timestamp': datetime.now().isoformat(),
                'reason': 'No models available for predictions'
            }), 503
        
        # Try to load the latest model
        try:
            model_id, model, metadata = model_loader.get_latest_model()
        except Exception as e:
            return jsonify({
                'status': 'not_ready',
                'timestamp': datetime.now().isoformat(),
                'reason': f'Cannot load model: {str(e)}'
            }), 503
        
        return jsonify({
            'status': 'ready',
            'timestamp': datetime.now().isoformat(),
            'ready_model': {
                'model_id': model_id,
                'model_type': metadata['model_type']
            }
        }), 200
        
    except Exception as e:
        logging.error(f"Readiness check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500


@app.route('/health/liveness', methods=['GET'])
def liveness_check():
    """
    Liveness check endpoint - checks if service is alive and responding.
    """
    return jsonify({
        'status': 'alive',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': int((datetime.now() - app.start_time).total_seconds()) if hasattr(app, 'start_time') else None
    }), 200


# Web Dashboard Routes
@app.route('/')
def dashboard():
    """
    Main dashboard page.
    """
    return render_template('dashboard.html')


@app.route('/models')
def models():
    """
    Models management page.
    """
    return render_template('models.html')


@app.route('/api-test')
def api_test():
    """
    API testing interface page.
    """
    return render_template('api_test.html')


@app.route('/api-docs')
def api_docs():
    """
    Serve the static API documentation.
    """
    return app.send_static_file('api-documentation.html')


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this endpoint'
    }), 405


if __name__ == '__main__':
    # Record start time for uptime tracking
    app.start_time = datetime.now()
    
    # Ensure log directory exists
    log_file = LOGGING_CONFIG.get('prediction_log_file', 'predictions.log')
    log_dir = os.path.dirname(log_file)
    if log_dir:  # Only create directory if there is one
        os.makedirs(log_dir, exist_ok=True)
    
    # Start background monitoring
    metrics_collector.start_monitoring(interval=60)  # Monitor every 60 seconds
    
    # Start automated retraining monitoring
    retraining_system.start_monitoring()
    
    try:
        # Start the Flask application
        app.run(
            host=API_CONFIG.get('host', '0.0.0.0'),
            port=API_CONFIG.get('port', 5000),
            debug=API_CONFIG.get('debug', True)
        )
    finally:
        # Stop monitoring when app shuts down
        metrics_collector.stop_monitoring()
        retraining_system.stop_monitoring()