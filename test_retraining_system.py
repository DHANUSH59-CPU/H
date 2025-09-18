"""
Test script for automated retraining system.
Tests the complete retraining workflow including triggers, versioning, and deployment.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta

# Add models directory to path
sys.path.append('models')

from models.retraining_system import AutomatedRetrainingSystem, ModelVersionManager, RetrainingTrigger
from models.monitoring import MetricsCollector, PerformanceMetrics
from models.model_trainer import TrainingScript
from config import MODEL_CONFIG, MONITORING_THRESHOLDS


def setup_logging():
    """Setup logging for test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_model_version_manager():
    """Test model version management functionality."""
    print("\n=== Testing Model Version Manager ===")
    
    # Initialize version manager
    version_manager = ModelVersionManager()
    
    # Create test versions
    print("Creating test model versions...")
    
    version1 = version_manager.create_version(
        model_id='test_model',
        model_file_path='models/test_model_v1.joblib',
        metadata_file_path='models/test_model_v1_metadata.json',
        training_metrics={'accuracy': 0.85, 'f1_score': 0.82},
        validation_metrics={'accuracy': 0.83, 'f1_score': 0.80},
        notes='Initial version'
    )
    
    print(f"Created version 1: {version1.version_id}")
    
    version2 = version_manager.create_version(
        model_id='test_model',
        model_file_path='models/test_model_v2.joblib',
        metadata_file_path='models/test_model_v2_metadata.json',
        training_metrics={'accuracy': 0.87, 'f1_score': 0.84},
        validation_metrics={'accuracy': 0.85, 'f1_score': 0.82},
        parent_version=version1.version_id,
        notes='Retrained version'
    )
    
    print(f"Created version 2: {version2.version_id}")
    
    # Test version activation
    print("Testing version activation...")
    success = version_manager.activate_version(version1.version_id)
    print(f"Activated version 1: {success}")
    
    active_version = version_manager.get_active_version('test_model')
    print(f"Active version: {active_version.version_id if active_version else 'None'}")
    
    # Test version history
    print("Testing version history...")
    history = version_manager.get_version_history('test_model')
    print(f"Version history: {len(history)} versions")
    for v in history:
        print(f"  - {v.version_id} (v{v.version_number}) - Active: {v.is_active}")
    
    # Test version comparison
    print("Testing version comparison...")
    comparison = version_manager.compare_versions(version1.version_id, version2.version_id)
    print(f"Comparison result: {json.dumps(comparison, indent=2)}")
    
    return version_manager


def test_retraining_triggers():
    """Test retraining trigger detection."""
    print("\n=== Testing Retraining Triggers ===")
    
    # Initialize components
    metrics_collector = MetricsCollector()
    retraining_system = AutomatedRetrainingSystem(
        metrics_collector=metrics_collector,
        config=MODEL_CONFIG
    )
    
    # Create test model version
    version_manager = retraining_system.version_manager
    test_version = version_manager.create_version(
        model_id='trigger_test_model',
        model_file_path='models/trigger_test.joblib',
        metadata_file_path='models/trigger_test_metadata.json',
        training_metrics={'accuracy': 0.90},
        validation_metrics={'accuracy': 0.88},
        notes='Test model for trigger testing'
    )
    
    # Activate the test version
    version_manager.activate_version(test_version.version_id)
    
    # Simulate performance metrics that would trigger retraining
    print("Simulating performance degradation...")
    
    # Add metrics showing accuracy drop
    for i in range(10):
        # Simulate degrading performance
        accuracy = 0.88 - (i * 0.01)  # Gradual accuracy drop
        
        metrics = PerformanceMetrics(
            model_id='trigger_test_model',
            timestamp=datetime.now() - timedelta(hours=i),
            accuracy=accuracy,
            precision=accuracy - 0.02,
            recall=accuracy - 0.01,
            f1_score=accuracy - 0.015,
            prediction_count=100
        )
        
        # Store metrics
        metrics_collector.storage.store_metrics(metrics)
    
    print("Stored degraded performance metrics")
    
    # Test manual trigger
    print("Testing manual retraining trigger...")
    result = retraining_system.trigger_manual_retraining(
        'trigger_test_model', 
        'Manual test trigger'
    )
    print(f"Manual trigger result: {result}")
    
    # Get retraining status
    status = retraining_system.get_retraining_status()
    print(f"Retraining status: {json.dumps(status, indent=2)}")
    
    return retraining_system


def test_complete_retraining_workflow():
    """Test complete retraining workflow with actual model training."""
    print("\n=== Testing Complete Retraining Workflow ===")
    
    # Ensure we have training data
    training_data_path = 'data/sample_classification.csv'
    if not os.path.exists(training_data_path):
        print(f"Training data not found at {training_data_path}")
        print("Creating sample training data...")
        
        # Create sample data
        import pandas as pd
        import numpy as np
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_classes=2, 
            random_state=42
        )
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(10)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save to file
        os.makedirs('data', exist_ok=True)
        df.to_csv(training_data_path, index=False)
        print(f"Created sample training data: {training_data_path}")
    
    # Train initial model
    print("Training initial model...")
    training_script = TrainingScript(MODEL_CONFIG)
    result = training_script.train_and_save_model(
        training_data_path, 
        'target', 
        'random_forest'
    )
    
    if not result['success']:
        print(f"Initial training failed: {result}")
        return
    
    initial_model_id = result['model_id']
    print(f"Initial model trained: {initial_model_id}")
    
    # Initialize retraining system
    metrics_collector = MetricsCollector()
    retraining_system = AutomatedRetrainingSystem(
        metrics_collector=metrics_collector,
        config=MODEL_CONFIG
    )
    
    # Create version for the initial model
    model_path = os.path.join('models', f"{initial_model_id}.joblib")
    metadata_path = os.path.join('models', f"{initial_model_id}_metadata.json")
    
    # Use a consistent model_id for versioning
    base_model_id = 'workflow_test_model'
    
    initial_version = retraining_system.version_manager.create_version(
        model_id=base_model_id,
        model_file_path=model_path,
        metadata_file_path=metadata_path,
        training_metrics=result['training_metrics'],
        validation_metrics=result['validation_metrics'],
        notes='Initial model for workflow test'
    )
    
    # Activate initial version
    retraining_system.version_manager.activate_version(initial_version.version_id)
    print(f"Activated initial version: {initial_version.version_id}")
    
    # Simulate some predictions and performance degradation
    print("Simulating predictions and performance degradation...")
    
    # Add prediction logs
    import numpy as np
    for i in range(100):
        prediction_log = {
            'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
            'model_id': base_model_id,
            'input_features': {f'feature_{j}': np.random.random() for j in range(10)},
            'prediction': np.random.choice([0, 1]),
            'confidence': 0.7 + np.random.random() * 0.3,
            'response_time_ms': 50 + np.random.randint(0, 100),
            'request_id': f'test_req_{i}'
        }
        metrics_collector.storage.store_prediction_log(prediction_log)
    
    # Add degraded performance metrics
    for i in range(20):
        # Simulate accuracy dropping below threshold
        accuracy = 0.88 - (i * 0.003)  # Gradual drop
        
        metrics = PerformanceMetrics(
            model_id=base_model_id,
            timestamp=datetime.now() - timedelta(minutes=i * 5),
            accuracy=accuracy,
            precision=accuracy - 0.01,
            recall=accuracy - 0.005,
            f1_score=accuracy - 0.008,
            prediction_count=5
        )
        
        metrics_collector.storage.store_metrics(metrics)
    
    print("Added prediction logs and degraded metrics")
    
    # Trigger manual retraining to test the workflow
    print("Triggering manual retraining...")
    trigger_result = retraining_system.trigger_manual_retraining(
        base_model_id, 
        'Testing complete workflow'
    )
    print(f"Trigger result: {trigger_result}")
    
    # Wait a bit for retraining to start
    print("Waiting for retraining to complete...")
    time.sleep(5)
    
    # Check status
    status = retraining_system.get_retraining_status()
    print(f"Final status: {json.dumps(status, indent=2)}")
    
    # Check version history
    versions = retraining_system.version_manager.get_version_history(base_model_id)
    print(f"\nFinal version history ({len(versions)} versions):")
    for v in versions:
        print(f"  - {v.version_id} (v{v.version_number}) - Active: {v.is_active} - Status: {v.deployment_status}")
        print(f"    Validation Accuracy: {v.validation_metrics.get('accuracy', 'N/A')}")
    
    return retraining_system


def main():
    """Run all tests."""
    setup_logging()
    
    print("Starting Automated Retraining System Tests")
    print("=" * 50)
    
    try:
        # Test 1: Model Version Manager
        version_manager = test_model_version_manager()
        
        # Test 2: Retraining Triggers
        retraining_system = test_retraining_triggers()
        
        # Test 3: Complete Workflow
        complete_system = test_complete_retraining_workflow()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
        # Clean up test files
        print("\nCleaning up test files...")
        test_files = [
            'models/model_versions.json',
            'models/test_model_v1.joblib',
            'models/test_model_v1_metadata.json',
            'models/test_model_v2.joblib',
            'models/test_model_v2_metadata.json'
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Could not remove {file_path}: {e}")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()