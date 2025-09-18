"""
Monitoring and metrics collection system for ML model deployment.
Handles performance tracking, drift detection, and alerting.
"""

import os
import sqlite3
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from scipy import stats

from config import MONITORING_THRESHOLDS, DATABASE_CONFIG, ALERT_CONFIG


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model at a specific time."""
    model_id: str
    timestamp: datetime
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    prediction_count: int = 0
    avg_response_time_ms: float = 0.0
    confidence_avg: Optional[float] = None


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    model_id: str
    has_drift: bool
    drift_score: float
    threshold: float
    feature_drifts: Dict[str, float]
    timestamp: datetime
    method: str = "ks_test"


@dataclass
class Alert:
    """Alert notification for monitoring events."""
    alert_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    model_id: str
    timestamp: datetime
    data: Dict[str, Any]
    resolved: bool = False


class MetricsStorage:
    """
    SQLite-based storage for monitoring metrics and alerts.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize metrics storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or DATABASE_CONFIG.get('database_path', 'ml_monitoring.db')
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Performance metrics table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall REAL,
                        f1_score REAL,
                        mse REAL,
                        mae REAL,
                        r2_score REAL,
                        prediction_count INTEGER DEFAULT 0,
                        avg_response_time_ms REAL DEFAULT 0.0,
                        confidence_avg REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Prediction logs table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS prediction_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        input_features TEXT,
                        prediction TEXT,
                        confidence REAL,
                        response_time_ms INTEGER,
                        request_id TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Drift detection results table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS drift_detection (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        has_drift BOOLEAN,
                        drift_score REAL,
                        threshold_value REAL,
                        feature_drifts TEXT,
                        method TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Alerts table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE NOT NULL,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        data TEXT,
                        resolved BOOLEAN DEFAULT FALSE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_model_time ON performance_metrics(model_id, timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model_time ON prediction_logs(model_id, timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_model_time ON alerts(model_id, timestamp)')
                
                conn.commit()
                logging.info("Database initialized successfully")
                
            except Exception as e:
                logging.error(f"Error initializing database: {e}")
                raise
            finally:
                conn.close()
    
    def store_metrics(self, metrics: PerformanceMetrics) -> bool:
        """
        Store performance metrics in database.
        
        Args:
            metrics: PerformanceMetrics object to store
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    INSERT INTO performance_metrics 
                    (model_id, timestamp, accuracy, precision_score, recall, f1_score, 
                     mse, mae, r2_score, prediction_count, avg_response_time_ms, confidence_avg)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.model_id,
                    metrics.timestamp.isoformat(),
                    metrics.accuracy,
                    metrics.precision,
                    metrics.recall,
                    metrics.f1_score,
                    metrics.mse,
                    metrics.mae,
                    metrics.r2_score,
                    metrics.prediction_count,
                    metrics.avg_response_time_ms,
                    metrics.confidence_avg
                ))
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                logging.error(f"Error storing metrics: {e}")
                return False
    
    def store_prediction_log(self, log_data: Dict[str, Any]) -> bool:
        """
        Store prediction log entry.
        
        Args:
            log_data: Dictionary containing prediction log data
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    INSERT INTO prediction_logs 
                    (timestamp, model_id, input_features, prediction, confidence, 
                     response_time_ms, request_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    log_data.get('timestamp'),
                    log_data.get('model_id'),
                    json.dumps(log_data.get('input_features')),
                    json.dumps(log_data.get('prediction')),
                    log_data.get('confidence'),
                    log_data.get('response_time_ms'),
                    log_data.get('request_id')
                ))
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                logging.error(f"Error storing prediction log: {e}")
                return False
    
    def store_drift_result(self, drift_result: DriftDetectionResult) -> bool:
        """
        Store drift detection result.
        
        Args:
            drift_result: DriftDetectionResult object to store
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    INSERT INTO drift_detection 
                    (model_id, timestamp, has_drift, drift_score, threshold_value, 
                     feature_drifts, method)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    drift_result.model_id,
                    drift_result.timestamp.isoformat(),
                    drift_result.has_drift,
                    drift_result.drift_score,
                    drift_result.threshold,
                    json.dumps(drift_result.feature_drifts),
                    drift_result.method
                ))
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                logging.error(f"Error storing drift result: {e}")
                return False
    
    def store_alert(self, alert: Alert) -> bool:
        """
        Store alert in database.
        
        Args:
            alert: Alert object to store
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    INSERT OR REPLACE INTO alerts 
                    (alert_id, alert_type, severity, message, model_id, timestamp, data, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id,
                    alert.alert_type,
                    alert.severity,
                    alert.message,
                    alert.model_id,
                    alert.timestamp.isoformat(),
                    json.dumps(alert.data),
                    alert.resolved
                ))
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                logging.error(f"Error storing alert: {e}")
                return False
    
    def get_recent_metrics(self, model_id: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent performance metrics.
        
        Args:
            model_id: Optional model ID filter
            hours: Number of hours to look back
            
        Returns:
            List of metrics dictionaries
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                
                cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
                
                if model_id:
                    cursor = conn.execute('''
                        SELECT * FROM performance_metrics 
                        WHERE model_id = ? AND timestamp >= ?
                        ORDER BY timestamp DESC
                    ''', (model_id, cutoff_time))
                else:
                    cursor = conn.execute('''
                        SELECT * FROM performance_metrics 
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                    ''', (cutoff_time,))
                
                results = [dict(row) for row in cursor.fetchall()]
                conn.close()
                return results
                
            except Exception as e:
                logging.error(f"Error retrieving metrics: {e}")
                return []
    
    def get_recent_predictions(self, model_id: str = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get recent prediction logs.
        
        Args:
            model_id: Optional model ID filter
            limit: Maximum number of records to return
            
        Returns:
            List of prediction log dictionaries
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                
                if model_id:
                    cursor = conn.execute('''
                        SELECT * FROM prediction_logs 
                        WHERE model_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (model_id, limit))
                else:
                    cursor = conn.execute('''
                        SELECT * FROM prediction_logs 
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (limit,))
                
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    # Parse JSON fields
                    if row_dict['input_features']:
                        row_dict['input_features'] = json.loads(row_dict['input_features'])
                    if row_dict['prediction']:
                        row_dict['prediction'] = json.loads(row_dict['prediction'])
                    results.append(row_dict)
                
                conn.close()
                return results
                
            except Exception as e:
                logging.error(f"Error retrieving predictions: {e}")
                return []
    
    def get_active_alerts(self, model_id: str = None) -> List[Dict[str, Any]]:
        """
        Get active (unresolved) alerts.
        
        Args:
            model_id: Optional model ID filter
            
        Returns:
            List of alert dictionaries
        """
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                
                if model_id:
                    cursor = conn.execute('''
                        SELECT * FROM alerts 
                        WHERE model_id = ? AND resolved = FALSE
                        ORDER BY timestamp DESC
                    ''', (model_id,))
                else:
                    cursor = conn.execute('''
                        SELECT * FROM alerts 
                        WHERE resolved = FALSE
                        ORDER BY timestamp DESC
                    ''')
                
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    if row_dict['data']:
                        row_dict['data'] = json.loads(row_dict['data'])
                    results.append(row_dict)
                
                conn.close()
                return results
                
            except Exception as e:
                logging.error(f"Error retrieving alerts: {e}")
                return []


class DriftDetector:
    """
    Simple statistical drift detection for monitoring data distribution changes.
    """
    
    def __init__(self, threshold: float = None):
        """
        Initialize drift detector.
        
        Args:
            threshold: Drift detection threshold (default from config)
        """
        self.threshold = threshold or MONITORING_THRESHOLDS.get('drift_threshold', 0.1)
        self.reference_data = {}  # Store reference distributions per model
    
    def set_reference_data(self, model_id: str, reference_features: Dict[str, List[float]]):
        """
        Set reference data distribution for a model.
        
        Args:
            model_id: Model identifier
            reference_features: Dictionary of feature_name -> list of values
        """
        self.reference_data[model_id] = {}
        
        for feature_name, values in reference_features.items():
            if len(values) > 0:
                self.reference_data[model_id][feature_name] = {
                    'values': np.array(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
    
    def detect_drift(self, model_id: str, new_features: Dict[str, List[float]]) -> DriftDetectionResult:
        """
        Detect drift between reference and new data using statistical tests.
        
        Args:
            model_id: Model identifier
            new_features: Dictionary of feature_name -> list of new values
            
        Returns:
            DriftDetectionResult object
        """
        if model_id not in self.reference_data:
            # No reference data available
            return DriftDetectionResult(
                model_id=model_id,
                has_drift=False,
                drift_score=0.0,
                threshold=self.threshold,
                feature_drifts={},
                timestamp=datetime.now(),
                method="no_reference"
            )
        
        reference = self.reference_data[model_id]
        feature_drifts = {}
        drift_scores = []
        
        for feature_name, new_values in new_features.items():
            if feature_name not in reference or len(new_values) == 0:
                continue
            
            ref_values = reference[feature_name]['values']
            new_array = np.array(new_values)
            
            # Perform Kolmogorov-Smirnov test
            try:
                ks_statistic, p_value = stats.ks_2samp(ref_values, new_array)
                
                # Use KS statistic as drift score (0 = no drift, 1 = maximum drift)
                drift_score = ks_statistic
                feature_drifts[feature_name] = drift_score
                drift_scores.append(drift_score)
                
            except Exception as e:
                logging.warning(f"Error calculating drift for feature {feature_name}: {e}")
                continue
        
        # Overall drift score is the maximum feature drift
        overall_drift_score = max(drift_scores) if drift_scores else 0.0
        has_drift = overall_drift_score > self.threshold
        
        return DriftDetectionResult(
            model_id=model_id,
            has_drift=has_drift,
            drift_score=overall_drift_score,
            threshold=self.threshold,
            feature_drifts=feature_drifts,
            timestamp=datetime.now(),
            method="ks_test"
        )
    
    def update_reference_from_predictions(self, model_id: str, predictions_data: List[Dict[str, Any]]):
        """
        Update reference data using recent prediction inputs.
        
        Args:
            model_id: Model identifier
            predictions_data: List of prediction log dictionaries
        """
        if not predictions_data:
            return
        
        # Collect feature values from predictions
        feature_values = defaultdict(list)
        
        for pred in predictions_data:
            input_features = pred.get('input_features', {})
            if isinstance(input_features, dict):
                for feature_name, value in input_features.items():
                    try:
                        feature_values[feature_name].append(float(value))
                    except (ValueError, TypeError):
                        continue
        
        # Set as reference data
        if feature_values:
            self.set_reference_data(model_id, dict(feature_values))
            logging.info(f"Updated reference data for model {model_id} with {len(predictions_data)} predictions")


class AlertManager:
    """
    Manages threshold-based alerts and notifications.
    """
    
    def __init__(self, storage: MetricsStorage):
        """
        Initialize alert manager.
        
        Args:
            storage: MetricsStorage instance for persisting alerts
        """
        self.storage = storage
        self.alert_cooldowns = {}  # Track alert cooldowns to prevent spam
        self.alert_methods = ALERT_CONFIG.get('alert_methods', ['console', 'log'])
        self.cooldown_seconds = ALERT_CONFIG.get('alert_cooldown', 300)
    
    def check_performance_thresholds(self, metrics: PerformanceMetrics) -> List[Alert]:
        """
        Check performance metrics against thresholds and generate alerts.
        
        Args:
            metrics: PerformanceMetrics to check
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Check accuracy threshold
        if metrics.accuracy is not None:
            accuracy_threshold = MONITORING_THRESHOLDS.get('accuracy_threshold', 0.85)
            if metrics.accuracy < accuracy_threshold:
                alert = self._create_alert(
                    alert_type='accuracy_drop',
                    severity='high',
                    message=f'Model accuracy ({metrics.accuracy:.3f}) below threshold ({accuracy_threshold})',
                    model_id=metrics.model_id,
                    data={'accuracy': metrics.accuracy, 'threshold': accuracy_threshold}
                )
                alerts.append(alert)
        
        # Check response time threshold
        response_threshold = MONITORING_THRESHOLDS.get('response_time_threshold', 2.0) * 1000  # Convert to ms
        if metrics.avg_response_time_ms > response_threshold:
            alert = self._create_alert(
                alert_type='slow_response',
                severity='medium',
                message=f'Average response time ({metrics.avg_response_time_ms:.1f}ms) exceeds threshold ({response_threshold}ms)',
                model_id=metrics.model_id,
                data={'response_time_ms': metrics.avg_response_time_ms, 'threshold_ms': response_threshold}
            )
            alerts.append(alert)
        
        # Check prediction volume
        volume_threshold = MONITORING_THRESHOLDS.get('prediction_volume_alert', 1000)
        if metrics.prediction_count > volume_threshold:
            alert = self._create_alert(
                alert_type='high_volume',
                severity='low',
                message=f'High prediction volume detected ({metrics.prediction_count} predictions)',
                model_id=metrics.model_id,
                data={'prediction_count': metrics.prediction_count, 'threshold': volume_threshold}
            )
            alerts.append(alert)
        
        return alerts
    
    def check_drift_alert(self, drift_result: DriftDetectionResult) -> Optional[Alert]:
        """
        Check drift detection result and generate alert if needed.
        
        Args:
            drift_result: DriftDetectionResult to check
            
        Returns:
            Alert if drift detected, None otherwise
        """
        if drift_result.has_drift:
            return self._create_alert(
                alert_type='data_drift',
                severity='high',
                message=f'Data drift detected (score: {drift_result.drift_score:.3f}, threshold: {drift_result.threshold})',
                model_id=drift_result.model_id,
                data={
                    'drift_score': drift_result.drift_score,
                    'threshold': drift_result.threshold,
                    'feature_drifts': drift_result.feature_drifts,
                    'method': drift_result.method
                }
            )
        return None
    
    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert using configured methods.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if alert was sent successfully
        """
        # Check cooldown
        cooldown_key = f"{alert.model_id}_{alert.alert_type}"
        now = datetime.now()
        
        if cooldown_key in self.alert_cooldowns:
            last_sent = self.alert_cooldowns[cooldown_key]
            if (now - last_sent).total_seconds() < self.cooldown_seconds:
                logging.debug(f"Alert {alert.alert_id} skipped due to cooldown")
                return False
        
        # Store alert in database
        stored = self.storage.store_alert(alert)
        if not stored:
            logging.error(f"Failed to store alert {alert.alert_id}")
            return False
        
        # Send via configured methods
        success = True
        
        if 'console' in self.alert_methods:
            self._send_console_alert(alert)
        
        if 'log' in self.alert_methods:
            self._send_log_alert(alert)
        
        # Update cooldown
        self.alert_cooldowns[cooldown_key] = now
        
        return success
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved.
        
        Args:
            alert_id: ID of alert to resolve
            
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.storage.db_path)
            conn.execute(
                'UPDATE alerts SET resolved = TRUE WHERE alert_id = ?',
                (alert_id,)
            )
            conn.commit()
            conn.close()
            logging.info(f"Alert {alert_id} marked as resolved")
            return True
            
        except Exception as e:
            logging.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def _create_alert(self, alert_type: str, severity: str, message: str, 
                     model_id: str, data: Dict[str, Any]) -> Alert:
        """Create a new alert with unique ID."""
        timestamp = datetime.now()
        alert_id = f"{alert_type}_{model_id}_{int(timestamp.timestamp())}"
        
        return Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            model_id=model_id,
            timestamp=timestamp,
            data=data
        )
    
    def _send_console_alert(self, alert: Alert):
        """Send alert to console."""
        severity_colors = {
            'low': '\033[94m',      # Blue
            'medium': '\033[93m',   # Yellow
            'high': '\033[91m',     # Red
            'critical': '\033[95m'  # Magenta
        }
        reset_color = '\033[0m'
        
        color = severity_colors.get(alert.severity, '')
        print(f"{color}[{alert.severity.upper()} ALERT] {alert.message}{reset_color}")
        print(f"  Model: {alert.model_id}")
        print(f"  Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Alert ID: {alert.alert_id}")
    
    def _send_log_alert(self, alert: Alert):
        """Send alert to log file."""
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        logging.log(log_level, f"ALERT [{alert.alert_type}] {alert.message} (Model: {alert.model_id}, ID: {alert.alert_id})")


class MetricsCollector:
    """
    Main metrics collection and monitoring coordinator.
    """
    
    def __init__(self, storage: MetricsStorage = None):
        """
        Initialize metrics collector.
        
        Args:
            storage: Optional MetricsStorage instance
        """
        self.storage = storage or MetricsStorage()
        self.drift_detector = DriftDetector()
        self.alert_manager = AlertManager(self.storage)
        
        # In-memory caches for real-time metrics
        self.prediction_cache = defaultdict(deque)  # Recent predictions per model
        self.metrics_cache = defaultdict(deque)     # Recent metrics per model
        self.cache_size = 1000  # Maximum items in cache
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 60  # seconds
    
    def log_prediction(self, prediction_data: Dict[str, Any]):
        """
        Log a prediction and update real-time metrics.
        
        Args:
            prediction_data: Dictionary containing prediction log data
        """
        # Store in database
        self.storage.store_prediction_log(prediction_data)
        
        # Update cache
        model_id = prediction_data.get('model_id', 'unknown')
        self.prediction_cache[model_id].append(prediction_data)
        
        # Maintain cache size
        if len(self.prediction_cache[model_id]) > self.cache_size:
            self.prediction_cache[model_id].popleft()
        
        # Update drift detector reference data periodically
        if len(self.prediction_cache[model_id]) % 100 == 0:  # Every 100 predictions
            recent_predictions = list(self.prediction_cache[model_id])
            self.drift_detector.update_reference_from_predictions(model_id, recent_predictions)
    
    def calculate_current_metrics(self, model_id: str) -> Optional[PerformanceMetrics]:
        """
        Calculate current performance metrics for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            PerformanceMetrics object or None if insufficient data
        """
        # Get recent predictions from cache and database
        cached_predictions = list(self.prediction_cache.get(model_id, []))
        
        if len(cached_predictions) < 10:  # Need minimum predictions for meaningful metrics
            # Try to get from database
            db_predictions = self.storage.get_recent_predictions(model_id, limit=100)
            if len(db_predictions) < 10:
                return None
            predictions = db_predictions
        else:
            predictions = cached_predictions
        
        # Calculate metrics from predictions
        prediction_count = len(predictions)
        
        # Response time metrics
        response_times = [p.get('response_time_ms', 0) for p in predictions if p.get('response_time_ms')]
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        # Confidence metrics
        confidences = [p.get('confidence') for p in predictions if p.get('confidence') is not None]
        avg_confidence = np.mean(confidences) if confidences else None
        
        # Create metrics object (accuracy will be calculated separately if ground truth is available)
        metrics = PerformanceMetrics(
            model_id=model_id,
            timestamp=datetime.now(),
            prediction_count=prediction_count,
            avg_response_time_ms=avg_response_time,
            confidence_avg=avg_confidence
        )
        
        return metrics
    
    def check_drift(self, model_id: str, recent_hours: int = 1) -> Optional[DriftDetectionResult]:
        """
        Check for data drift in recent predictions.
        
        Args:
            model_id: Model identifier
            recent_hours: Hours of recent data to analyze
            
        Returns:
            DriftDetectionResult or None if insufficient data
        """
        # Get recent predictions
        cutoff_time = datetime.now() - timedelta(hours=recent_hours)
        recent_predictions = []
        
        # Check cache first
        for pred in self.prediction_cache.get(model_id, []):
            pred_time = datetime.fromisoformat(pred.get('timestamp', ''))
            if pred_time >= cutoff_time:
                recent_predictions.append(pred)
        
        # If not enough in cache, get from database
        if len(recent_predictions) < 50:
            db_predictions = self.storage.get_recent_predictions(model_id, limit=200)
            recent_predictions = [
                p for p in db_predictions 
                if datetime.fromisoformat(p.get('timestamp', '')) >= cutoff_time
            ]
        
        if len(recent_predictions) < 20:  # Need minimum samples for drift detection
            return None
        
        # Extract features for drift detection
        feature_values = defaultdict(list)
        for pred in recent_predictions:
            input_features = pred.get('input_features', {})
            if isinstance(input_features, dict):
                for feature_name, value in input_features.items():
                    try:
                        feature_values[feature_name].append(float(value))
                    except (ValueError, TypeError):
                        continue
        
        if not feature_values:
            return None
        
        # Perform drift detection
        drift_result = self.drift_detector.detect_drift(model_id, dict(feature_values))
        
        # Store result
        self.storage.store_drift_result(drift_result)
        
        return drift_result
    
    def run_monitoring_cycle(self):
        """Run a single monitoring cycle for all active models."""
        try:
            # Get list of active models (models with recent predictions)
            recent_predictions = self.storage.get_recent_predictions(limit=1000)
            active_models = set(p.get('model_id') for p in recent_predictions if p.get('model_id'))
            
            for model_id in active_models:
                # Calculate current metrics
                metrics = self.calculate_current_metrics(model_id)
                if metrics:
                    # Store metrics
                    self.storage.store_metrics(metrics)
                    
                    # Check performance thresholds
                    performance_alerts = self.alert_manager.check_performance_thresholds(metrics)
                    for alert in performance_alerts:
                        self.alert_manager.send_alert(alert)
                
                # Check for drift
                drift_result = self.check_drift(model_id)
                if drift_result:
                    drift_alert = self.alert_manager.check_drift_alert(drift_result)
                    if drift_alert:
                        self.alert_manager.send_alert(drift_alert)
            
            logging.info(f"Monitoring cycle completed for {len(active_models)} models")
            
        except Exception as e:
            logging.error(f"Error in monitoring cycle: {e}")
    
    def start_monitoring(self, interval: int = None):
        """
        Start background monitoring thread.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            logging.warning("Monitoring already active")
            return
        
        if interval:
            self.monitoring_interval = interval
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                self.run_monitoring_cycle()
                time.sleep(self.monitoring_interval)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logging.info(f"Background monitoring started (interval: {self.monitoring_interval}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring thread."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logging.info("Background monitoring stopped")
    
    def get_dashboard_data(self, model_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive monitoring data for dashboard display.
        
        Args:
            model_id: Optional model ID filter
            hours: Hours of historical data to include
            
        Returns:
            Dictionary containing dashboard data
        """
        # Get recent metrics
        metrics_data = self.storage.get_recent_metrics(model_id, hours)
        
        # Get recent predictions
        predictions_data = self.storage.get_recent_predictions(model_id, limit=500)
        
        # Get active alerts
        alerts_data = self.storage.get_active_alerts(model_id)
        
        # Calculate summary statistics
        total_predictions = len(predictions_data)
        
        # Response time statistics
        response_times = [p.get('response_time_ms', 0) for p in predictions_data if p.get('response_time_ms')]
        avg_response_time = np.mean(response_times) if response_times else 0
        
        # Model usage statistics
        model_usage = defaultdict(int)
        for pred in predictions_data:
            model_usage[pred.get('model_id', 'unknown')] += 1
        
        # Recent drift results
        drift_data = []
        try:
            conn = sqlite3.connect(self.storage.db_path)
            conn.row_factory = sqlite3.Row
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            if model_id:
                cursor = conn.execute('''
                    SELECT * FROM drift_detection 
                    WHERE model_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''', (model_id, cutoff_time))
            else:
                cursor = conn.execute('''
                    SELECT * FROM drift_detection 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''', (cutoff_time,))
            
            for row in cursor.fetchall():
                row_dict = dict(row)
                if row_dict['feature_drifts']:
                    row_dict['feature_drifts'] = json.loads(row_dict['feature_drifts'])
                drift_data.append(row_dict)
            
            conn.close()
            
        except Exception as e:
            logging.error(f"Error retrieving drift data: {e}")
        
        return {
            'summary': {
                'total_predictions': total_predictions,
                'avg_response_time_ms': round(avg_response_time, 2),
                'active_alerts': len(alerts_data),
                'models_monitored': len(model_usage)
            },
            'metrics_history': metrics_data,
            'recent_predictions': predictions_data[-50:],  # Last 50 predictions
            'alerts': alerts_data,
            'model_usage': dict(model_usage),
            'drift_results': drift_data,
            'timestamp': datetime.now().isoformat()
        }


# Global instance for easy access
_metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    """Get global MetricsCollector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector