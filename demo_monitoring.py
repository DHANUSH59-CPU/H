#!/usr/bin/env python3
"""
Demo script for monitoring and metrics collection system.
Shows how the monitoring system works with the ML API.
"""

import requests
import json
import time
import numpy as np
from datetime import datetime
import threading


def test_api_with_monitoring():
    """Test the API endpoints with monitoring enabled."""
    base_url = "http://localhost:5000"
    
    print("=" * 60)
    print("MONITORING SYSTEM DEMO")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            print("❌ API is not running. Please start the Flask app first:")
            print("   python app.py")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please start the Flask app first:")
        print("   python app.py")
        return
    
    print("✓ API is running")
    
    # Check model status
    print("\n1. Checking model status...")
    response = requests.get(f"{base_url}/model/status")
    if response.status_code == 200:
        status = response.json()
        print(f"✓ {status['total_models']} models available")
        if status['current_model']:
            current_model = status['current_model']
            print(f"✓ Current model: {current_model['model_id']}")
            print(f"  - Type: {current_model['model_type']}")
            print(f"  - Features: {current_model['feature_names']}")
        else:
            print("❌ No models available. Please train a model first:")
            print("   python train_model.py")
            return
    else:
        print("❌ Failed to get model status")
        return
    
    # Get feature names for predictions
    feature_names = current_model['feature_names']
    
    print(f"\n2. Making predictions to generate monitoring data...")
    print(f"   Using features: {feature_names}")
    
    # Make normal predictions
    print("   Making 20 normal predictions...")
    for i in range(20):
        # Generate realistic feature values
        features = {}
        for feature in feature_names:
            if 'sepal' in feature.lower():
                features[feature] = round(np.random.uniform(4.0, 8.0), 2)
            elif 'petal' in feature.lower():
                features[feature] = round(np.random.uniform(1.0, 7.0), 2)
            else:
                features[feature] = round(np.random.uniform(0.0, 10.0), 2)
        
        payload = {"features": features}
        
        response = requests.post(f"{base_url}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"     Prediction {i+1}: {result['prediction']} (confidence: {result.get('confidence', 'N/A')}, "
                  f"time: {result['response_time_ms']}ms)")
        else:
            print(f"     ❌ Prediction {i+1} failed: {response.text}")
        
        time.sleep(0.1)  # Small delay
    
    # Make some predictions with drifted data
    print("   Making 10 drifted predictions...")
    for i in range(10):
        # Generate drifted feature values (shifted distributions)
        features = {}
        for feature in feature_names:
            if 'sepal' in feature.lower():
                features[feature] = round(np.random.uniform(8.0, 12.0), 2)  # Shifted higher
            elif 'petal' in feature.lower():
                features[feature] = round(np.random.uniform(7.0, 12.0), 2)  # Shifted higher
            else:
                features[feature] = round(np.random.uniform(10.0, 20.0), 2)  # Shifted higher
        
        payload = {"features": features}
        
        response = requests.post(f"{base_url}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"     Drifted {i+1}: {result['prediction']} (confidence: {result.get('confidence', 'N/A')}, "
                  f"time: {result['response_time_ms']}ms)")
        
        time.sleep(0.1)
    
    print("✓ Completed 30 predictions")
    
    # Wait a moment for monitoring to process
    print("\n3. Waiting for monitoring system to process data...")
    time.sleep(2)
    
    # Check metrics
    print("\n4. Checking monitoring metrics...")
    response = requests.get(f"{base_url}/metrics")
    if response.status_code == 200:
        metrics = response.json()
        summary = metrics['summary']
        
        print(f"✓ Monitoring Summary:")
        print(f"  - Total predictions: {summary['total_predictions']}")
        print(f"  - Average response time: {summary['avg_response_time_ms']}ms")
        print(f"  - Active alerts: {summary['active_alerts']}")
        print(f"  - Models monitored: {summary['models_monitored']}")
        
        # Show model usage
        if metrics.get('model_usage'):
            print(f"  - Model usage:")
            for model_id, count in metrics['model_usage'].items():
                print(f"    {model_id}: {count} predictions")
        
        # Show recent predictions
        recent = metrics.get('recent_predictions', [])
        if recent:
            print(f"  - Last prediction: {recent[-1]['prediction']} at {recent[-1]['timestamp']}")
    
    else:
        print(f"❌ Failed to get metrics: {response.text}")
    
    # Check for drift
    print("\n5. Checking for data drift...")
    model_id = current_model['model_id']
    response = requests.get(f"{base_url}/metrics/drift/{model_id}?hours=1")
    if response.status_code == 200:
        drift = response.json()
        print(f"✓ Drift Analysis:")
        print(f"  - Has drift: {drift['has_drift']}")
        print(f"  - Drift score: {drift['drift_score']:.3f}")
        print(f"  - Threshold: {drift['threshold']}")
        print(f"  - Method: {drift['method']}")
        
        if drift.get('feature_drifts'):
            print(f"  - Feature drift scores:")
            for feature, score in drift['feature_drifts'].items():
                status = "⚠️ DRIFT" if score > drift['threshold'] else "✓ OK"
                print(f"    {feature}: {score:.3f} {status}")
    else:
        print(f"❌ Failed to check drift: {response.text}")
    
    # Check alerts
    print("\n6. Checking active alerts...")
    response = requests.get(f"{base_url}/alerts")
    if response.status_code == 200:
        alerts_data = response.json()
        alerts = alerts_data['alerts']
        
        if alerts:
            print(f"✓ Found {len(alerts)} active alerts:")
            for alert in alerts:
                severity_emoji = {
                    'low': '🔵',
                    'medium': '🟡', 
                    'high': '🔴',
                    'critical': '🟣'
                }.get(alert['severity'], '⚪')
                
                print(f"  {severity_emoji} [{alert['severity'].upper()}] {alert['message']}")
                print(f"    Type: {alert['alert_type']}, Time: {alert['timestamp']}")
                print(f"    Alert ID: {alert['alert_id']}")
        else:
            print("✓ No active alerts")
    else:
        print(f"❌ Failed to get alerts: {response.text}")
    
    print("\n" + "=" * 60)
    print("MONITORING DEMO COMPLETED!")
    print("=" * 60)
    
    print("\nKey monitoring features demonstrated:")
    print("- ✓ Real-time prediction logging and metrics collection")
    print("- ✓ Performance monitoring (response time, volume)")
    print("- ✓ Data drift detection using statistical methods")
    print("- ✓ Threshold-based alerting system")
    print("- ✓ SQLite database storage for metrics and alerts")
    print("- ✓ REST API endpoints for monitoring data")
    
    print("\nAvailable monitoring endpoints:")
    print(f"- GET {base_url}/metrics - Comprehensive monitoring dashboard data")
    print(f"- GET {base_url}/metrics/drift/<model_id> - Check data drift for specific model")
    print(f"- GET {base_url}/alerts - Get active alerts")
    print(f"- POST {base_url}/alerts/<alert_id>/resolve - Resolve specific alert")
    
    print("\nNext steps:")
    print("- Monitor the system in real-time using the /metrics endpoint")
    print("- Set up automated retraining when drift is detected")
    print("- Integrate with external alerting systems (email, Slack, etc.)")
    print("- Build a web dashboard for visual monitoring")


def continuous_monitoring_demo():
    """Run continuous monitoring demo."""
    print("\n" + "=" * 60)
    print("CONTINUOUS MONITORING DEMO")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    
    try:
        while True:
            # Get current metrics
            response = requests.get(f"{base_url}/metrics")
            if response.status_code == 200:
                metrics = response.json()
                summary = metrics['summary']
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Predictions: {summary['total_predictions']}, "
                      f"Avg Response: {summary['avg_response_time_ms']:.1f}ms, "
                      f"Alerts: {summary['active_alerts']}")
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n✓ Continuous monitoring stopped")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        continuous_monitoring_demo()
    else:
        test_api_with_monitoring()
        
        # Ask if user wants continuous monitoring
        try:
            response = input("\nWould you like to run continuous monitoring? (y/n): ")
            if response.lower().startswith('y'):
                continuous_monitoring_demo()
        except KeyboardInterrupt:
            print("\n✓ Demo completed")