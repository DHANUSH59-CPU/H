"""
Configuration file for ML deployment and monitoring system
Contains model parameters, thresholds, and system settings
"""

import os

# Model Configuration
MODEL_CONFIG = {
    'default_algorithm': 'random_forest',
    'test_size': 0.2,
    'random_state': 42,
    'cross_validation_folds': 5,
    'model_save_path': 'models/',
    'model_file_extension': '.joblib',
    # Retraining configuration
    'retraining_check_interval': 300,  # seconds (5 minutes)
    'min_predictions_for_retraining': 50,  # minimum predictions needed to trigger retraining
    'retraining_data_window_hours': 24,  # hours of data to consider for retraining
    'retraining_data_limit': 10000,  # maximum number of predictions to use for retraining
    'min_improvement_threshold': 0.01,  # minimum improvement required to deploy new version
    'scheduled_retraining_hours': None,  # hours between scheduled retraining (None = disabled)
    'original_training_data': 'data/sample_classification.csv'  # fallback training data
}

# Monitoring Thresholds
MONITORING_THRESHOLDS = {
    'accuracy_threshold': 0.85,  # Minimum acceptable accuracy
    'drift_threshold': 0.1,      # Statistical drift detection threshold
    'prediction_volume_alert': 1000,  # Alert when predictions exceed this number
    'response_time_threshold': 2.0,   # Maximum response time in seconds
    'retraining_accuracy_drop': 0.05  # Trigger retraining when accuracy drops by this amount
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'prediction_endpoint': '/predict',
    'metrics_endpoint': '/metrics',
    'retrain_endpoint': '/model/retrain',
    'status_endpoint': '/model/status'
}

# Data Configuration
DATA_CONFIG = {
    'data_path': 'data/',
    'training_data_file': 'training_data.csv',
    'test_data_file': 'test_data.csv',
    'sample_data_size': 1000,
    'feature_columns': [],  # Will be populated based on dataset
    'target_column': 'target'
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'refresh_interval': 5000,  # Milliseconds
    'max_chart_points': 100,
    'chart_colors': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'],
    'metrics_history_limit': 1000
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'ml_system.log',
    'prediction_log_file': 'predictions.log',
    'max_log_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Database Configuration (SQLite for simplicity)
DATABASE_CONFIG = {
    'database_path': 'ml_monitoring.db',
    'metrics_table': 'performance_metrics',
    'predictions_table': 'prediction_logs',
    'models_table': 'model_metadata'
}

# Alert Configuration
ALERT_CONFIG = {
    'enable_alerts': True,
    'alert_methods': ['console', 'log'],  # Can extend to email, slack, etc.
    'alert_cooldown': 300,  # Seconds between similar alerts
    'critical_threshold_multiplier': 1.5
}

# Environment-specific overrides
if os.getenv('FLASK_ENV') == 'production':
    API_CONFIG['debug'] = False
    LOGGING_CONFIG['log_level'] = 'WARNING'

if os.getenv('DEMO_MODE') == 'true':
    # Demo-specific settings for hackathon presentation
    DATA_CONFIG['sample_data_size'] = 500
    MONITORING_THRESHOLDS['accuracy_threshold'] = 0.75
    DASHBOARD_CONFIG['refresh_interval'] = 2000