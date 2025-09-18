#!/usr/bin/env python3
"""
Demo script showing how to use the DataProcessor class.
This demonstrates the core data processing functionality for the ML system.
"""

import sys
import os
sys.path.append('models')

from data_processor import DataProcessor, SampleDataGenerator
import pandas as pd


def demo_data_loading():
    """Demonstrate data loading functionality."""
    print("=== Data Loading Demo ===")
    
    processor = DataProcessor()
    
    # Load sample classification data
    print("Loading sample classification data...")
    data = processor.load_data('data/sample_classification.csv')
    print(f"Loaded data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"First few rows:\n{data.head()}")
    print()


def demo_data_validation():
    """Demonstrate data validation functionality."""
    print("=== Data Validation Demo ===")
    
    processor = DataProcessor()
    
    # Load and validate data
    data = processor.load_data('data/sample_iris.csv')
    validation_result = processor.validate_data(data, 'species')
    
    print(f"Validation result: {'PASSED' if validation_result.is_valid else 'FAILED'}")
    if validation_result.errors:
        print(f"Errors: {validation_result.errors}")
    if validation_result.warnings:
        print(f"Warnings: {validation_result.warnings}")
    print()


def demo_data_preprocessing():
    """Demonstrate data preprocessing functionality."""
    print("=== Data Preprocessing Demo ===")
    
    processor = DataProcessor({'min_rows': 5})  # Lower threshold for demo
    
    # Load and preprocess iris data
    data = processor.load_data('data/sample_iris.csv')
    print(f"Original data shape: {data.shape}")
    
    processed = processor.preprocess(data, 'species', test_size=0.3)
    
    print(f"Training features shape: {processed.X_train.shape}")
    print(f"Test features shape: {processed.X_test.shape}")
    print(f"Training target shape: {processed.y_train.shape}")
    print(f"Test target shape: {processed.y_test.shape}")
    print(f"Feature names: {processed.feature_names}")
    print(f"Target name: {processed.target_name}")
    print()
    
    return processor, processed


def demo_new_data_transformation():
    """Demonstrate transforming new data."""
    print("=== New Data Transformation Demo ===")
    
    # First, fit the processor
    processor, _ = demo_data_preprocessing()
    
    # Create some new data to transform
    new_data = pd.DataFrame({
        'sepal_length': [5.1, 6.2, 7.3],
        'sepal_width': [3.5, 2.8, 3.1],
        'petal_length': [1.4, 4.5, 6.2],
        'petal_width': [0.2, 1.5, 2.1]
    })
    
    print(f"New data to transform:\n{new_data}")
    
    transformed = processor.transform_new_data(new_data)
    print(f"Transformed data shape: {transformed.shape}")
    print(f"Transformed data (first row): {transformed[0]}")
    print()


def demo_sample_data_generation():
    """Demonstrate sample data generation."""
    print("=== Sample Data Generation Demo ===")
    
    # Generate different types of sample data
    print("Generating classification data...")
    class_data = SampleDataGenerator.generate_classification_data(n_samples=100, n_features=3, n_classes=2)
    print(f"Classification data shape: {class_data.shape}")
    print(f"Target distribution:\n{class_data['target'].value_counts()}")
    print()
    
    print("Generating regression data...")
    reg_data = SampleDataGenerator.generate_regression_data(n_samples=100, n_features=4)
    print(f"Regression data shape: {reg_data.shape}")
    print(f"Target statistics:\n{reg_data['target'].describe()}")
    print()
    
    print("Generating iris-like data...")
    iris_data = SampleDataGenerator.generate_iris_like_data(n_samples=150)
    print(f"Iris-like data shape: {iris_data.shape}")
    print(f"Species distribution:\n{iris_data['species'].value_counts()}")
    print()


def main():
    """Run all demos."""
    print("DataProcessor Demo Script")
    print("=" * 50)
    print()
    
    try:
        demo_data_loading()
        demo_data_validation()
        demo_data_preprocessing()
        demo_new_data_transformation()
        demo_sample_data_generation()
        
        print("All demos completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()