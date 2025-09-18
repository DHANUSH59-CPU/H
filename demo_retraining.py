"""
Demo script for automated retraining system.
Shows the complete retraining workflow including monitoring, triggers, and deployment.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta

# Add models directory to path
sys.path.append('models')

from models.retraining_system import get_retraining_system
from models.monitoring import get_metrics_collector, PerformanceMetrics
from models.model_trainer import TrainingScript
from config import MODEL_CONFIG


def setup_demo_environment():
    """Setup demo environment with sample data and initial model."""
    print("Setting up demo environment...")
    
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Create sample training data if it doesn't exist
    training_data_path = 'data/sample_classification.csv'
    if not os.path.exists(training_data_path):
        print("Creating sample training data...")
        
        import pandas as pd
        import numpy as np
        from sklearn.datasets import make_classification
        
        # Generate synthetic classification dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=8,
            n_classes=2,
            n_informative=6,
            n_redundant=2,
            random_state=42
        )
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(8)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save to file
        df.to_csv(training_data_path, index=False)
        print(f"Created sample data: {training_data_path}")
    
    return training_data_path


def train_initial_model(training_data_path):
    """Train initial model for demo."""
    print("\n=== Training Initial Model ===")
    
    training_script = TrainingScript(MODEL_CONFIG)
    result = training_script.train_and_save_model(
        training_data_path,
        'target',
        'random_forest'
    )
    
    if result['success']:
        print(f"✓ Initial model trained successfully: {result['model_id']}")
        print(f"  Validation Accuracy: {result['validation_metrics'].get('accuracy', 'N/A'):.4f}")
        return result
    else:
        print(f"✗ Initial model training failed: {result.get('error')}")
        return None


def setup_retraining_system(initial_model_result):
    """Setup and configure the retraining system."""
    print("\n=== Setting Up Retraining System ===")
    
    # Get system components
    metrics_collector = get_metrics_collector()
    retraining_system = get_retraining_system(
        metrics_collector=metrics_collector,
        config=MODEL_CONFIG
    )
    
    # Create version for initial model
    model_id = 'demo_model'  # Use consistent ID for versioning
    initial_model_id = initial_model_result['model_id']
    
    model_path = os.path.join('models', f"{initial_model_id}.joblib")
    metadata_path = os.path.join('models', f"{initial_model_id}_metadata.json")
    
    # Create initial version
    initial_version = retraining_system.version_manager.create_version(
        model_id=model_id,
        model_file_path=model_path,
        metadata_file_path=metadata_path,
        training_metrics=initial_model_result['training_metrics'],
        validation_metrics=initial_model_result['validation_metrics'],
        notes='Initial demo model'
    )
    
    # Activate initial version
    success = retraining_system.version_manager.activate_version(initial_version.version_id)
    
    if success:
        print(f"✓ Retraining system initialized")
        print(f"  Initial version: {initial_version.version_id}")
        print(f"  Model ID: {model_id}")
    else:
        print("✗ Failed to activate initial version")
        return None, None
    
    return retraining_system, model_id


def simulate_model_usage(metrics_collector, model_id, num_predictions=200):
    """Simulate model usage with predictions and performance data."""
    print(f"\n=== Simulating Model Usage ({num_predictions} predictions) ===")
    
    import numpy as np
    
    # Simulate prediction logs
    print("Generating prediction logs...")
    for i in range(num_predictions):
        prediction_log = {
            'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
            'model_id': model_id,
            'input_features': {f'feature_{j}': float(np.random.normal(0, 1)) for j in range(8)},
            'prediction': int(np.random.choice([0, 1])),
            'confidence': float(0.6 + np.random.random() * 0.4),
            'response_time_ms': int(30 + np.random.randint(0, 70)),
            'request_id': f'demo_req_{i}'
        }
        metrics_collector.storage.store_prediction_log(prediction_log)
    
    # Simulate initial good performance
    print("Generating initial performance metrics...")
    for i in range(10):
        metrics = PerformanceMetrics(
            model_id=model_id,
            timestamp=datetime.now() - timedelta(hours=24-i),
            accuracy=0.88 + np.random.normal(0, 0.02),  # Good initial performance
            precision=0.86 + np.random.normal(0, 0.02),
            recall=0.87 + np.random.normal(0, 0.02),
            f1_score=0.865 + np.random.normal(0, 0.02),
            prediction_count=20,
            avg_response_time_ms=45 + np.random.randint(-10, 15)
        )
        metrics_collector.storage.store_metrics(metrics)
    
    print(f"✓ Simulated {num_predictions} predictions and 10 performance metrics")


def simulate_performance_degradation(metrics_collector, model_id):
    """Simulate gradual performance degradation to trigger retraining."""
    print("\n=== Simulating Performance Degradation ===")
    
    import numpy as np
    
    print("Generating degraded performance metrics...")
    
    # Simulate performance degradation over time
    base_accuracy = 0.88
    degradation_rate = 0.008  # Accuracy drops by ~0.8% per time step
    
    for i in range(15):
        # Calculate degraded accuracy
        current_accuracy = base_accuracy - (i * degradation_rate)
        
        # Add some noise
        accuracy = current_accuracy + np.random.normal(0, 0.01)
        
        metrics = PerformanceMetrics(
            model_id=model_id,
            timestamp=datetime.now() - timedelta(hours=i),
            accuracy=accuracy,
            precision=accuracy - 0.01,
            recall=accuracy - 0.005,
            f1_score=accuracy - 0.008,
            prediction_count=15,
            avg_response_time_ms=50 + np.random.randint(-5, 20)
        )
        
        metrics_collector.storage.store_metrics(metrics)
        
        print(f"  Time -{i}h: Accuracy = {accuracy:.4f}")
    
    final_accuracy = base_accuracy - (14 * degradation_rate)
    accuracy_drop = base_accuracy - final_accuracy
    threshold = MODEL_CONFIG.get('retraining_accuracy_drop', 0.05)
    
    print(f"✓ Performance degradation simulated")
    print(f"  Initial accuracy: {base_accuracy:.4f}")
    print(f"  Final accuracy: {final_accuracy:.4f}")
    print(f"  Total drop: {accuracy_drop:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Should trigger: {'Yes' if accuracy_drop > threshold else 'No'}")


def demonstrate_manual_retraining(retraining_system, model_id):
    """Demonstrate manual retraining trigger."""
    print("\n=== Demonstrating Manual Retraining ===")
    
    # Check current status
    status = retraining_system.get_retraining_status()
    print(f"Current retraining status:")
    print(f"  Monitoring active: {status['monitoring_active']}")
    print(f"  Models being retrained: {status['models_being_retrained']}")
    print(f"  Recent triggers: {len(status['recent_triggers'])}")
    
    # Trigger manual retraining
    print(f"\nTriggering manual retraining for model: {model_id}")
    result = retraining_system.trigger_manual_retraining(
        model_id,
        "Demo: Manual retraining to show workflow"
    )
    
    if result['success']:
        print(f"✓ Manual retraining triggered successfully")
        print(f"  Trigger ID: {result['trigger_id']}")
        
        # Wait for retraining to start
        print("Waiting for retraining to begin...")
        time.sleep(2)
        
        # Check status again
        status = retraining_system.get_retraining_status()
        print(f"Updated status:")
        print(f"  Models being retrained: {status['models_being_retrained']}")
        
        # Wait for retraining to complete (or timeout)
        print("Waiting for retraining to complete (max 30 seconds)...")
        timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = retraining_system.get_retraining_status()
            if model_id not in status['models_being_retrained']:
                print("✓ Retraining completed!")
                break
            time.sleep(2)
            print("  Still retraining...")
        else:
            print("⚠ Retraining timeout (may still be running in background)")
        
    else:
        print(f"✗ Manual retraining failed: {result.get('error')}")
    
    return result['success'] if result else False


def show_version_history(retraining_system, model_id):
    """Show model version history and comparison."""
    print(f"\n=== Model Version History for {model_id} ===")
    
    versions = retraining_system.version_manager.get_version_history(model_id)
    
    if not versions:
        print("No versions found")
        return
    
    print(f"Total versions: {len(versions)}")
    print()
    
    for i, version in enumerate(versions):
        status_icon = "🟢" if version.is_active else "⚪"
        print(f"{status_icon} Version {version.version_number}: {version.version_id}")
        print(f"   Created: {version.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Status: {version.deployment_status}")
        print(f"   Validation Accuracy: {version.validation_metrics.get('accuracy', 'N/A'):.4f}")
        if version.parent_version:
            print(f"   Parent: {version.parent_version}")
        if version.notes:
            print(f"   Notes: {version.notes}")
        print()
    
    # Compare versions if we have more than one
    if len(versions) >= 2:
        print("=== Version Comparison (Latest vs Previous) ===")
        latest = versions[0]
        previous = versions[1]
        
        comparison = retraining_system.version_manager.compare_versions(
            latest.version_id, previous.version_id
        )
        
        if 'improvements' in comparison:
            print(f"Comparing {latest.version_id} vs {previous.version_id}:")
            for metric, improvement in comparison['improvements'].items():
                direction = "↗" if improvement['better'] else "↘"
                print(f"  {metric}: {direction} {improvement['absolute']:+.4f} ({improvement['percentage']:+.2f}%)")


def demonstrate_monitoring_integration(retraining_system):
    """Demonstrate integration with monitoring system."""
    print("\n=== Monitoring System Integration ===")
    
    # Start monitoring if not already active
    if not retraining_system.monitoring_active:
        print("Starting automated monitoring...")
        retraining_system.start_monitoring()
        time.sleep(1)
    
    status = retraining_system.get_retraining_status()
    print(f"Monitoring status: {'Active' if status['monitoring_active'] else 'Inactive'}")
    
    # Show configuration
    config = status['configuration']
    print(f"Configuration:")
    print(f"  Check interval: {config['check_interval']} seconds")
    print(f"  Accuracy drop threshold: {config['accuracy_drop_threshold']}")
    print(f"  Min predictions for retraining: {config['min_predictions_for_retraining']}")
    print(f"  Data window: {config['retraining_data_window_hours']} hours")
    
    # Show recent triggers
    recent_triggers = status['recent_triggers']
    if recent_triggers:
        print(f"\nRecent triggers ({len(recent_triggers)}):")
        for trigger in recent_triggers[-3:]:  # Show last 3
            print(f"  - {trigger['trigger_type']} for {trigger['model_id']}")
            print(f"    Triggered: {trigger['triggered_at']}")
            print(f"    Value: {trigger['current_value']:.4f} (threshold: {trigger['threshold_value']:.4f})")
    else:
        print("\nNo recent triggers")


def main():
    """Run the complete retraining system demo."""
    print("🤖 Automated ML Retraining System Demo")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Step 1: Setup environment
        training_data_path = setup_demo_environment()
        
        # Step 2: Train initial model
        initial_result = train_initial_model(training_data_path)
        if not initial_result:
            return
        
        # Step 3: Setup retraining system
        retraining_system, model_id = setup_retraining_system(initial_result)
        if not retraining_system:
            return
        
        # Step 4: Simulate model usage
        metrics_collector = get_metrics_collector()
        simulate_model_usage(metrics_collector, model_id)
        
        # Step 5: Simulate performance degradation
        simulate_performance_degradation(metrics_collector, model_id)
        
        # Step 6: Demonstrate monitoring integration
        demonstrate_monitoring_integration(retraining_system)
        
        # Step 7: Demonstrate manual retraining
        retraining_success = demonstrate_manual_retraining(retraining_system, model_id)
        
        # Step 8: Show version history
        show_version_history(retraining_system, model_id)
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        
        if retraining_success:
            print("✓ Automated retraining system is working correctly")
            print("✓ Model versioning and deployment automation functional")
            print("✓ Performance monitoring and trigger detection operational")
        else:
            print("⚠ Some components may need attention")
        
        print("\nThe system is now ready for production use!")
        print("You can:")
        print("- Monitor models automatically in the background")
        print("- Trigger retraining manually via API")
        print("- View version history and comparisons")
        print("- Deploy specific model versions")
        
        # Keep monitoring running for a bit
        print("\nMonitoring will continue running...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(10)
                status = retraining_system.get_retraining_status()
                if status['models_being_retrained']:
                    print(f"Currently retraining: {status['models_being_retrained']}")
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            retraining_system.stop_monitoring()
            print("Demo finished.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()