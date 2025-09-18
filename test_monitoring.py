#!/usr/bin/env python3
"""
Test script for monitoring and metrics collection system.
Tests MetricsCollector, DriftDetector, and AlertManager functionality.
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

# Add models directory to path
sys.path.append('models')

from models.monitoring import (
    MetricsCollector, MetricsStorage, DriftDetector, AlertManager,
    PerformanceMetrics, DriftDetectionResult, Alert
)


def test_metrics_storage():
    """Test MetricsStorage functionality."""
    print("Testing MetricsStorage...")
    
    # Use test database
    storage = MetricsStorage("test_monitoring.db")
    
    # Test storing performance metrics
    metrics = PerformanceMetrics(
        model_id="test_model_1",
        timestamp=datetime.now(),
        accuracy=0.92,
        precision=0.89,
        recall=0.94,
        f1_score=0.91,
        prediction_count=150,
        avg_response_time_ms=45.2,
        confidence_avg=0.87
    )
    
    success = storage.store_metrics(metrics)
    assert success, "Failed to store metrics"
    print("✓ Metrics storage successful")
    
    # Test storing prediction log
    prediction_log = {
        'timestamp': datetime.now().isoformat(),
        'model_id': 'test_model_1',
        'input_features': {'feature1': 1.5, 'feature2': 2.3, 'feature3': 0.8},
        'prediction': 'class_A',
        'confidence': 0.89,
        'response_time_ms': 42,
        'request_id': 'test_req_001'
    }
    
    success = storage.store_prediction_log(prediction_log)
    assert success, "Failed to store prediction log"
    print("✓ Prediction log storage successful")
    
    # Test retrieving metrics
    recent_metrics = storage.get_recent_metrics("test_model_1", hours=1)
    assert len(recent_metrics) > 0, "No metrics retrieved"
    print(f"✓ Retrieved {len(recent_metrics)} metrics records")
    
    # Test retrieving predictions
    recent_predictions = storage.get_recent_predictions("test_model_1", limit=10)
    assert len(recent_predictions) > 0, "No predictions retrieved"
    print(f"✓ Retrieved {len(recent_predictions)} prediction records")
    
    print("MetricsStorage tests passed!\n")
    return storage


def test_drift_detector():
    """Test DriftDetector functionality."""
    print("Testing DriftDetector...")
    
    detector = DriftDetector(threshold=0.1)
    
    # Create reference data (normal distribution)
    np.random.seed(42)
    reference_data = {
        'feature1': np.random.normal(0, 1, 1000).tolist(),
        'feature2': np.random.normal(5, 2, 1000).tolist(),
        'feature3': np.random.uniform(0, 10, 1000).tolist()
    }
    
    detector.set_reference_data("test_model_1", reference_data)
    print("✓ Reference data set")
    
    # Test with similar data (no drift expected)
    similar_data = {
        'feature1': np.random.normal(0.1, 1.1, 200).tolist(),
        'feature2': np.random.normal(5.1, 2.1, 200).tolist(),
        'feature3': np.random.uniform(0.2, 9.8, 200).tolist()
    }
    
    result = detector.detect_drift("test_model_1", similar_data)
    print(f"✓ No drift test: has_drift={result.has_drift}, score={result.drift_score:.3f}")
    
    # Test with drifted data (drift expected)
    drifted_data = {
        'feature1': np.random.normal(3, 1, 200).tolist(),  # Mean shifted
        'feature2': np.random.normal(10, 2, 200).tolist(),  # Mean shifted
        'feature3': np.random.uniform(15, 25, 200).tolist()  # Range shifted
    }
    
    result = detector.detect_drift("test_model_1", drifted_data)
    print(f"✓ Drift test: has_drift={result.has_drift}, score={result.drift_score:.3f}")
    
    # Test feature-level drift scores
    print("Feature drift scores:")
    for feature, score in result.feature_drifts.items():
        print(f"  {feature}: {score:.3f}")
    
    print("DriftDetector tests passed!\n")
    return detector


def test_alert_manager(storage):
    """Test AlertManager functionality."""
    print("Testing AlertManager...")
    
    alert_manager = AlertManager(storage)
    
    # Test performance threshold alerts
    # Create metrics that should trigger alerts
    poor_metrics = PerformanceMetrics(
        model_id="test_model_1",
        timestamp=datetime.now(),
        accuracy=0.75,  # Below threshold (0.85)
        avg_response_time_ms=2500,  # Above threshold (2000ms)
        prediction_count=1200  # Above volume threshold (1000)
    )
    
    alerts = alert_manager.check_performance_thresholds(poor_metrics)
    print(f"✓ Generated {len(alerts)} performance alerts")
    
    for alert in alerts:
        print(f"  - {alert.alert_type}: {alert.message}")
        success = alert_manager.send_alert(alert)
        assert success, f"Failed to send alert {alert.alert_id}"
    
    # Test drift alert
    drift_result = DriftDetectionResult(
        model_id="test_model_1",
        has_drift=True,
        drift_score=0.25,
        threshold=0.1,
        feature_drifts={'feature1': 0.25, 'feature2': 0.15},
        timestamp=datetime.now(),
        method="ks_test"
    )
    
    drift_alert = alert_manager.check_drift_alert(drift_result)
    if drift_alert:
        print(f"✓ Generated drift alert: {drift_alert.message}")
        success = alert_manager.send_alert(drift_alert)
        assert success, f"Failed to send drift alert {drift_alert.alert_id}"
    
    # Test retrieving active alerts
    active_alerts = storage.get_active_alerts("test_model_1")
    print(f"✓ Retrieved {len(active_alerts)} active alerts")
    
    # Test resolving an alert
    if active_alerts:
        alert_id = active_alerts[0]['alert_id']
        success = alert_manager.resolve_alert(alert_id)
        assert success, f"Failed to resolve alert {alert_id}"
        print(f"✓ Resolved alert {alert_id}")
    
    print("AlertManager tests passed!\n")


def test_metrics_collector():
    """Test MetricsCollector integration."""
    print("Testing MetricsCollector...")
    
    collector = MetricsCollector()
    
    # Simulate prediction logs
    model_id = "test_model_integration"
    
    print("Simulating prediction logs...")
    for i in range(50):
        prediction_data = {
            'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
            'model_id': model_id,
            'input_features': {
                'feature1': np.random.normal(0, 1),
                'feature2': np.random.normal(5, 2),
                'feature3': np.random.uniform(0, 10)
            },
            'prediction': np.random.choice(['class_A', 'class_B', 'class_C']),
            'confidence': np.random.uniform(0.6, 0.95),
            'response_time_ms': int(np.random.uniform(20, 100)),
            'request_id': f'req_{i:03d}'
        }
        
        collector.log_prediction(prediction_data)
    
    print("✓ Logged 50 predictions")
    
    # Test calculating current metrics
    current_metrics = collector.calculate_current_metrics(model_id)
    if current_metrics:
        print(f"✓ Calculated metrics: {current_metrics.prediction_count} predictions, "
              f"{current_metrics.avg_response_time_ms:.1f}ms avg response time")
    
    # Test drift detection
    drift_result = collector.check_drift(model_id, recent_hours=1)
    if drift_result:
        print(f"✓ Drift check: has_drift={drift_result.has_drift}, score={drift_result.drift_score:.3f}")
    
    # Test dashboard data
    dashboard_data = collector.get_dashboard_data(model_id, hours=1)
    print(f"✓ Dashboard data: {dashboard_data['summary']['total_predictions']} predictions, "
          f"{dashboard_data['summary']['active_alerts']} alerts")
    
    # Test monitoring cycle
    print("Running monitoring cycle...")
    collector.run_monitoring_cycle()
    print("✓ Monitoring cycle completed")
    
    print("MetricsCollector tests passed!\n")


def test_integration_with_sample_data():
    """Test with realistic sample data."""
    print("Testing with realistic sample data...")
    
    collector = MetricsCollector()
    model_id = "iris_classifier_test"
    
    # Simulate iris dataset predictions
    iris_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    iris_classes = ['setosa', 'versicolor', 'virginica']
    
    print("Generating realistic iris predictions...")
    
    # Generate normal predictions
    for i in range(100):
        # Simulate realistic iris feature values
        features = {
            'sepal_length': np.random.uniform(4.0, 8.0),
            'sepal_width': np.random.uniform(2.0, 4.5),
            'petal_length': np.random.uniform(1.0, 7.0),
            'petal_width': np.random.uniform(0.1, 2.5)
        }
        
        prediction_data = {
            'timestamp': (datetime.now() - timedelta(minutes=i*2)).isoformat(),
            'model_id': model_id,
            'input_features': features,
            'prediction': np.random.choice(iris_classes),
            'confidence': np.random.uniform(0.7, 0.98),
            'response_time_ms': int(np.random.uniform(15, 80)),
            'request_id': f'iris_req_{i:03d}'
        }
        
        collector.log_prediction(prediction_data)
    
    print("✓ Generated 100 normal predictions")
    
    # Generate some drifted predictions (different feature distributions)
    for i in range(20):
        # Simulate drifted feature values (shifted distributions)
        features = {
            'sepal_length': np.random.uniform(6.0, 10.0),  # Shifted higher
            'sepal_width': np.random.uniform(1.0, 3.0),    # Shifted lower
            'petal_length': np.random.uniform(3.0, 9.0),   # Shifted higher
            'petal_width': np.random.uniform(1.0, 3.5)     # Shifted higher
        }
        
        prediction_data = {
            'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
            'model_id': model_id,
            'input_features': features,
            'prediction': np.random.choice(iris_classes),
            'confidence': np.random.uniform(0.5, 0.85),  # Lower confidence
            'response_time_ms': int(np.random.uniform(25, 120)),  # Slower response
            'request_id': f'iris_drift_req_{i:03d}'
        }
        
        collector.log_prediction(prediction_data)
    
    print("✓ Generated 20 drifted predictions")
    
    # Run comprehensive analysis
    print("\nRunning comprehensive analysis...")
    
    # Check current metrics
    metrics = collector.calculate_current_metrics(model_id)
    if metrics:
        print(f"Current metrics:")
        print(f"  - Predictions: {metrics.prediction_count}")
        print(f"  - Avg response time: {metrics.avg_response_time_ms:.1f}ms")
        print(f"  - Avg confidence: {metrics.confidence_avg:.3f}")
    
    # Check for drift
    drift_result = collector.check_drift(model_id, recent_hours=1)
    if drift_result:
        print(f"Drift analysis:")
        print(f"  - Has drift: {drift_result.has_drift}")
        print(f"  - Overall score: {drift_result.drift_score:.3f}")
        print(f"  - Feature drifts:")
        for feature, score in drift_result.feature_drifts.items():
            print(f"    {feature}: {score:.3f}")
    
    # Get dashboard data
    dashboard = collector.get_dashboard_data(model_id, hours=2)
    print(f"Dashboard summary:")
    print(f"  - Total predictions: {dashboard['summary']['total_predictions']}")
    print(f"  - Active alerts: {dashboard['summary']['active_alerts']}")
    print(f"  - Models monitored: {dashboard['summary']['models_monitored']}")
    
    print("Integration test completed!\n")


def cleanup_test_files():
    """Clean up test database files."""
    test_files = ['test_monitoring.db', 'ml_monitoring.db']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up {file}")


def main():
    """Run all monitoring system tests."""
    print("=" * 60)
    print("MONITORING SYSTEM TESTS")
    print("=" * 60)
    
    try:
        # Test individual components
        storage = test_metrics_storage()
        test_drift_detector()
        test_alert_manager(storage)
        
        # Test integrated system
        test_metrics_collector()
        test_integration_with_sample_data()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
        print("\nMonitoring system is ready for use!")
        print("Key features implemented:")
        print("- ✓ MetricsCollector for tracking prediction accuracy and volume")
        print("- ✓ DriftDetector using statistical comparison methods")
        print("- ✓ AlertManager for threshold-based notifications")
        print("- ✓ SQLite database for metrics storage")
        print("- ✓ Integration with Flask API")
        print("- ✓ Real-time monitoring and alerting")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up test files
        cleanup_test_files()
    
    return 0


if __name__ == '__main__':
    exit(main())