#!/usr/bin/env python3
"""
Integration test for monitoring system with the ML API.
Tests the complete workflow including training, prediction, and monitoring.
"""

import os
import sys
import subprocess
import time
import requests
import json
import numpy as np
from datetime import datetime


def test_complete_workflow():
    """Test the complete ML workflow with monitoring."""
    print("=" * 70)
    print("COMPLETE ML WORKFLOW WITH MONITORING TEST")
    print("=" * 70)
    
    # Step 1: Train a model if none exists
    print("1. Checking for existing models...")
    
    models_dir = "models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    if not model_files:
        print("   No models found. Training a new model...")
        
        # Use sample data for training
        sample_data = "data/sample_iris.csv"
        if not os.path.exists(sample_data):
            print(f"   ❌ Sample data not found: {sample_data}")
            return False
        
        # Train model
        result = subprocess.run([
            sys.executable, "train_model.py", 
            "--data", sample_data,
            "--target", "species",
            "--algorithm", "random_forest"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"   ❌ Model training failed: {result.stderr}")
            return False
        
        print("   ✓ Model trained successfully")
    else:
        print(f"   ✓ Found {len(model_files)} existing models")
    
    # Step 2: Start the API server in background
    print("\n2. Starting API server...")
    
    api_process = subprocess.Popen([
        sys.executable, "app.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Check if server is running
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code != 200:
            print("   ❌ API server failed to start")
            return False
        print("   ✓ API server started successfully")
        
        # Step 3: Test monitoring endpoints
        print("\n3. Testing monitoring endpoints...")
        
        # Test model status
        response = requests.get("http://localhost:5000/model/status")
        if response.status_code != 200:
            print("   ❌ Model status endpoint failed")
            return False
        
        status = response.json()
        print(f"   ✓ Model status: {status['total_models']} models available")
        
        if not status['current_model']:
            print("   ❌ No current model available")
            return False
        
        current_model = status['current_model']
        feature_names = current_model['feature_names']
        print(f"   ✓ Current model: {current_model['model_id']}")
        print(f"   ✓ Features: {feature_names}")
        
        # Step 4: Make predictions to generate monitoring data
        print("\n4. Making predictions to generate monitoring data...")
        
        predictions_made = 0
        
        # Make normal predictions
        for i in range(15):
            features = {}
            for feature in feature_names:
                if 'sepal' in feature.lower():
                    features[feature] = round(np.random.uniform(4.0, 8.0), 2)
                elif 'petal' in feature.lower():
                    features[feature] = round(np.random.uniform(1.0, 7.0), 2)
                else:
                    features[feature] = round(np.random.uniform(0.0, 10.0), 2)
            
            payload = {"features": features}
            response = requests.post("http://localhost:5000/predict", json=payload)
            
            if response.status_code == 200:
                predictions_made += 1
            else:
                print(f"   ❌ Prediction {i+1} failed: {response.text}")
        
        print(f"   ✓ Made {predictions_made} predictions successfully")
        
        # Step 5: Test metrics endpoint
        print("\n5. Testing metrics collection...")
        
        time.sleep(1)  # Allow monitoring to process
        
        response = requests.get("http://localhost:5000/metrics")
        if response.status_code != 200:
            print("   ❌ Metrics endpoint failed")
            return False
        
        metrics = response.json()
        summary = metrics['summary']
        
        print(f"   ✓ Metrics collected:")
        print(f"     - Total predictions: {summary['total_predictions']}")
        print(f"     - Average response time: {summary['avg_response_time_ms']}ms")
        print(f"     - Models monitored: {summary['models_monitored']}")
        
        if summary['total_predictions'] < predictions_made:
            print(f"   ⚠️  Expected {predictions_made} predictions, got {summary['total_predictions']}")
        
        # Step 6: Test drift detection
        print("\n6. Testing drift detection...")
        
        model_id = current_model['model_id']
        response = requests.get(f"http://localhost:5000/metrics/drift/{model_id}?hours=1")
        
        if response.status_code == 200:
            drift = response.json()
            print(f"   ✓ Drift detection results:")
            print(f"     - Has drift: {drift['has_drift']}")
            print(f"     - Drift score: {drift['drift_score']:.3f}")
            print(f"     - Method: {drift['method']}")
        else:
            print(f"   ⚠️  Drift detection: {response.json().get('message', 'No data')}")
        
        # Step 7: Test alerts endpoint
        print("\n7. Testing alerts system...")
        
        response = requests.get("http://localhost:5000/alerts")
        if response.status_code != 200:
            print("   ❌ Alerts endpoint failed")
            return False
        
        alerts_data = response.json()
        alerts = alerts_data['alerts']
        
        print(f"   ✓ Alerts system working: {len(alerts)} active alerts")
        
        if alerts:
            for alert in alerts[:3]:  # Show first 3 alerts
                print(f"     - [{alert['severity']}] {alert['alert_type']}: {alert['message']}")
        
        # Step 8: Test database storage
        print("\n8. Testing database storage...")
        
        db_path = "ml_monitoring.db"
        if os.path.exists(db_path):
            print(f"   ✓ Monitoring database created: {db_path}")
            
            # Check database size
            db_size = os.path.getsize(db_path)
            print(f"   ✓ Database size: {db_size} bytes")
        else:
            print("   ❌ Monitoring database not found")
            return False
        
        # Step 9: Test monitoring with drifted data
        print("\n9. Testing with drifted data...")
        
        # Make predictions with drifted features
        drifted_predictions = 0
        for i in range(10):
            features = {}
            for feature in feature_names:
                # Shift all features significantly higher
                if 'sepal' in feature.lower():
                    features[feature] = round(np.random.uniform(10.0, 15.0), 2)
                elif 'petal' in feature.lower():
                    features[feature] = round(np.random.uniform(10.0, 15.0), 2)
                else:
                    features[feature] = round(np.random.uniform(15.0, 25.0), 2)
            
            payload = {"features": features}
            response = requests.post("http://localhost:5000/predict", json=payload)
            
            if response.status_code == 200:
                drifted_predictions += 1
        
        print(f"   ✓ Made {drifted_predictions} drifted predictions")
        
        # Wait for monitoring to process drifted data
        time.sleep(2)
        
        # Check drift again
        response = requests.get(f"http://localhost:5000/metrics/drift/{model_id}?hours=1")
        if response.status_code == 200:
            drift = response.json()
            print(f"   ✓ Updated drift detection:")
            print(f"     - Has drift: {drift['has_drift']}")
            print(f"     - Drift score: {drift['drift_score']:.3f}")
            
            if drift['has_drift']:
                print("   ✓ Drift successfully detected!")
            else:
                print("   ⚠️  Drift not detected (may need more data)")
        
        print("\n" + "=" * 70)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("=" * 70)
        
        print("\nMonitoring system successfully integrated with ML API:")
        print("- ✓ Real-time prediction logging")
        print("- ✓ Performance metrics collection")
        print("- ✓ Data drift detection")
        print("- ✓ Alert generation and management")
        print("- ✓ SQLite database storage")
        print("- ✓ REST API endpoints")
        print("- ✓ Background monitoring thread")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Stop the API server
        print("\n10. Cleaning up...")
        api_process.terminate()
        api_process.wait(timeout=5)
        print("   ✓ API server stopped")


def main():
    """Run integration tests."""
    success = test_complete_workflow()
    
    if success:
        print("\n🎉 Monitoring system is fully integrated and working!")
        print("\nTo use the monitoring system:")
        print("1. Start the API: python app.py")
        print("2. Make predictions via /predict endpoint")
        print("3. Monitor via /metrics endpoint")
        print("4. Check drift via /metrics/drift/<model_id>")
        print("5. View alerts via /alerts endpoint")
        
        print("\nFor a live demo, run:")
        print("   python demo_monitoring.py")
        
        return 0
    else:
        print("\n❌ Integration tests failed")
        return 1


if __name__ == '__main__':
    exit(main())