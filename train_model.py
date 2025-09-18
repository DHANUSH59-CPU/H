#!/usr/bin/env python3
"""
Standalone training script for the automated ML deployment system.
Can be triggered programmatically or run from command line.
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional

# Add models directory to path
sys.path.append(os.path.dirname(__file__))

from models.model_trainer import TrainingScript
from models.data_processor import SampleDataGenerator
from config import MODEL_CONFIG, DATA_CONFIG


def create_sample_data(data_type: str = 'classification', 
                      output_path: str = None) -> str:
    """
    Create sample data for training and testing.
    
    Args:
        data_type: Type of data to generate ('classification', 'regression', 'iris')
        output_path: Path to save the data (optional)
        
    Returns:
        Path to created data file
    """
    if output_path is None:
        data_dir = DATA_CONFIG.get('data_path', 'data/')
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, f'sample_{data_type}.csv')
    
    print(f"Generating sample {data_type} data...")
    
    if data_type == 'classification':
        data = SampleDataGenerator.generate_classification_data(
            n_samples=DATA_CONFIG.get('sample_data_size', 1000),
            random_state=MODEL_CONFIG.get('random_state', 42)
        )
    elif data_type == 'regression':
        data = SampleDataGenerator.generate_regression_data(
            n_samples=DATA_CONFIG.get('sample_data_size', 1000),
            random_state=MODEL_CONFIG.get('random_state', 42)
        )
    elif data_type == 'iris':
        data = SampleDataGenerator.generate_iris_like_data(
            n_samples=150,
            random_state=MODEL_CONFIG.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    data.to_csv(output_path, index=False)
    print(f"Sample data saved to: {output_path}")
    
    return output_path


def train_model_from_config(data_path: str = None, 
                          config_overrides: Dict[str, Any] = None,
                          target_column: str = None) -> Dict[str, Any]:
    """
    Train model using configuration settings.
    
    Args:
        data_path: Path to training data (optional, will use sample data if not provided)
        config_overrides: Dictionary of config values to override
        target_column: Target column name (optional, will use config default)
        
    Returns:
        Training result dictionary
    """
    # Merge configurations
    config = {**MODEL_CONFIG}
    if config_overrides:
        config.update(config_overrides)
    
    # Create training script
    script = TrainingScript(config)
    
    # Use sample data if no data path provided
    if data_path is None:
        print("No data path provided, generating sample classification data...")
        data_path = create_sample_data('classification')
    
    # Get target column from parameter, config, or use default
    if target_column is None:
        target_column = DATA_CONFIG.get('target_column', 'target')
    
    # Get algorithm from config
    algorithm = config.get('default_algorithm', 'random_forest')
    
    print(f"\nStarting training with configuration:")
    print(f"  Data: {data_path}")
    print(f"  Target: {target_column}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Test size: {config.get('test_size', 0.2)}")
    print(f"  Random state: {config.get('random_state', 42)}")
    
    # Train model
    result = script.train_and_save_model(
        data_path=data_path,
        target_column=target_column,
        algorithm=algorithm
    )
    
    return result


def retrain_existing_model(model_id: str, new_data_path: str = None) -> Dict[str, Any]:
    """
    Retrain an existing model with new data.
    
    Args:
        model_id: ID of existing model to retrain
        new_data_path: Path to new training data
        
    Returns:
        Retraining result dictionary
    """
    config = MODEL_CONFIG.copy()
    script = TrainingScript(config)
    
    if new_data_path is None:
        print("No new data path provided, generating fresh sample data...")
        new_data_path = create_sample_data('classification')
    
    print(f"\nRetraining model {model_id} with new data: {new_data_path}")
    
    result = script.retrain_model(model_id, new_data_path)
    
    return result


def demo_training_pipeline():
    """
    Demonstrate the complete training pipeline with different scenarios.
    """
    print("=" * 60)
    print("ML Model Training Pipeline Demo")
    print("=" * 60)
    
    results = []
    
    # Demo 1: Classification with Random Forest
    print("\n1. Training Classification Model (Random Forest)")
    print("-" * 50)
    
    clf_data_path = create_sample_data('classification')
    clf_result = train_model_from_config(
        data_path=clf_data_path,
        config_overrides={'default_algorithm': 'random_forest'}
    )
    results.append(('Classification (RF)', clf_result))
    
    # Demo 2: Classification with Logistic Regression
    print("\n2. Training Classification Model (Logistic Regression)")
    print("-" * 50)
    
    clf_lr_result = train_model_from_config(
        data_path=clf_data_path,
        config_overrides={'default_algorithm': 'logistic_regression'}
    )
    results.append(('Classification (LR)', clf_lr_result))
    
    # Demo 3: Regression with Random Forest
    print("\n3. Training Regression Model (Random Forest)")
    print("-" * 50)
    
    reg_data_path = create_sample_data('regression')
    reg_result = train_model_from_config(
        data_path=reg_data_path,
        config_overrides={'default_algorithm': 'random_forest'}
    )
    results.append(('Regression (RF)', reg_result))
    
    # Demo 4: Iris-like dataset
    print("\n4. Training on Iris-like Dataset")
    print("-" * 50)
    
    iris_data_path = create_sample_data('iris')
    iris_result = train_model_from_config(
        data_path=iris_data_path,
        config_overrides={'default_algorithm': 'random_forest'},
        target_column='species'
    )
    results.append(('Iris Dataset', iris_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    
    for name, result in results:
        if result['success']:
            print(f"\n✓ {name}")
            print(f"  Model ID: {result['model_id']}")
            print(f"  Model Type: {result['model_type']}")
            
            # Show key metrics
            val_metrics = result['validation_metrics']
            if 'accuracy' in val_metrics:
                print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            if 'r2_score' in val_metrics:
                print(f"  R² Score: {val_metrics['r2_score']:.4f}")
        else:
            print(f"\n✗ {name}")
            print(f"  Error: {result['error']}")
    
    print(f"\nDemo completed! Check the 'models/' directory for saved models.")
    
    return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Train ML models for automated deployment system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with sample data
  python train_model.py --demo
  
  # Train with custom data
  python train_model.py --data data/my_data.csv --target my_target --algorithm random_forest
  
  # Generate sample data only
  python train_model.py --generate-data classification --output data/sample.csv
  
  # Retrain existing model
  python train_model.py --retrain model_id --new-data data/new_data.csv
        """
    )
    
    # Main action arguments
    parser.add_argument('--demo', action='store_true',
                       help='Run complete training demo with sample data')
    parser.add_argument('--data', type=str,
                       help='Path to training data file')
    parser.add_argument('--target', type=str, default='target',
                       help='Target column name (default: target)')
    parser.add_argument('--algorithm', type=str, default='random_forest',
                       choices=['random_forest', 'logistic_regression', 'svm'],
                       help='Algorithm to use (default: random_forest)')
    
    # Data generation arguments
    parser.add_argument('--generate-data', type=str,
                       choices=['classification', 'regression', 'iris'],
                       help='Generate sample data of specified type')
    parser.add_argument('--output', type=str,
                       help='Output path for generated data')
    
    # Retraining arguments
    parser.add_argument('--retrain', type=str,
                       help='Model ID to retrain')
    parser.add_argument('--new-data', type=str,
                       help='Path to new training data for retraining')
    
    # Configuration arguments
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    try:
        if args.demo:
            # Run demo
            demo_training_pipeline()
            
        elif args.generate_data:
            # Generate sample data
            output_path = create_sample_data(args.generate_data, args.output)
            print(f"Sample data generated: {output_path}")
            
        elif args.retrain:
            # Retrain existing model
            if not args.new_data:
                print("Error: --new-data is required for retraining")
                return 1
            
            result = retrain_existing_model(args.retrain, args.new_data)
            
            if result['success']:
                print(f"Retraining successful! New model ID: {result['model_id']}")
                return 0
            else:
                print(f"Retraining failed: {result['error']}")
                return 1
                
        elif args.data:
            # Train with provided data
            config_overrides = {
                'default_algorithm': args.algorithm,
                'test_size': args.test_size,
                'random_state': args.random_state
            }
            
            result = train_model_from_config(args.data, config_overrides)
            
            if result['success']:
                print(f"Training successful! Model ID: {result['model_id']}")
                return 0
            else:
                print(f"Training failed: {result['error']}")
                return 1
                
        else:
            # No specific action, train with sample data
            print("No data provided, training with sample data...")
            result = train_model_from_config()
            
            if result['success']:
                print(f"Training successful! Model ID: {result['model_id']}")
                return 0
            else:
                print(f"Training failed: {result['error']}")
                return 1
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())