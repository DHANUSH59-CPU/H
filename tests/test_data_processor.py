"""
Unit tests for data processing functionality.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Add the models directory to the path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.data_processor import DataProcessor, SampleDataGenerator, ValidationResult, ProcessedData


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use config with lower minimum rows for testing
        self.processor = DataProcessor({'min_rows': 3})
        
        # Create sample data for testing (more rows to meet default requirements)
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'feature2': [2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8, 10.9, 11.0, 12.1, 13.2],
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'A'],
            'target': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        
        # Create data with missing values
        self.data_with_missing = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'feature2': [2.1, np.nan, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8, 10.9, 11.0, 12.1, 13.2],
            'category': ['A', 'B', None, 'C', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'A'],
            'target': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
        })
    
    def test_load_data_csv(self):
        """Test loading CSV data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            self.sample_data.to_csv(temp_file, index=False)
        
        try:
            loaded_data = self.processor.load_data(temp_file)
            
            self.assertIsInstance(loaded_data, pd.DataFrame)
            self.assertEqual(len(loaded_data), len(self.sample_data))
            self.assertEqual(list(loaded_data.columns), list(self.sample_data.columns))
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_load_data_json(self):
        """Test loading JSON data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            self.sample_data.to_json(temp_file, orient='records')
        
        try:
            loaded_data = self.processor.load_data(temp_file)
            
            self.assertIsInstance(loaded_data, pd.DataFrame)
            self.assertEqual(len(loaded_data), len(self.sample_data))
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_load_data_file_not_found(self):
        """Test loading non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_data('non_existent_file.csv')
    
    def test_load_data_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name
            f.write('some text')
        
        try:
            with self.assertRaises(ValueError):
                self.processor.load_data(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_validate_data_valid(self):
        """Test validation of valid data."""
        result = self.processor.validate_data(self.sample_data, 'target')
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_data_empty(self):
        """Test validation of empty data."""
        empty_data = pd.DataFrame()
        result = self.processor.validate_data(empty_data)
        
        self.assertFalse(result.is_valid)
        self.assertIn("Dataset is empty", result.errors)
    
    def test_validate_data_missing_target(self):
        """Test validation with missing target column."""
        result = self.processor.validate_data(self.sample_data, 'non_existent_target')
        
        self.assertFalse(result.is_valid)
        self.assertIn("Target column 'non_existent_target' not found", result.errors[0])
    
    def test_validate_data_with_missing_values(self):
        """Test validation of data with missing values."""
        # Create data with more significant missing values to trigger warnings
        data_with_many_missing = pd.DataFrame({
            'feature1': [1, 2, np.nan, np.nan, np.nan, 6, 7, 8, 9, 10, 11, 12],
            'feature2': [2.1, np.nan, np.nan, np.nan, 6.5, 7.6, 8.7, 9.8, 10.9, 11.0, 12.1, 13.2],
            'category': ['A', 'B', None, None, 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'A'],
            'target': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        
        result = self.processor.validate_data(data_with_many_missing, 'target')
        
        # Should still be valid but have warnings
        self.assertTrue(result.is_valid)
        self.assertTrue(len(result.warnings) > 0)
    
    def test_validate_data_target_missing_values(self):
        """Test validation when target has missing values."""
        data_missing_target = self.sample_data.copy()
        data_missing_target.loc[0, 'target'] = np.nan
        
        result = self.processor.validate_data(data_missing_target, 'target')
        
        self.assertFalse(result.is_valid)
        self.assertIn("Target column 'target' has 1 missing values", result.errors[0])
    
    def test_validate_data_too_few_rows(self):
        """Test validation with too few rows."""
        processor_with_config = DataProcessor({'min_rows': 10})
        small_data = self.sample_data.head(2)
        
        result = processor_with_config.validate_data(small_data)
        
        self.assertFalse(result.is_valid)
        self.assertIn("Dataset has only 2 rows", result.errors[0])
    
    def test_preprocess_classification_data(self):
        """Test preprocessing for classification data."""
        processed = self.processor.preprocess(self.sample_data, 'target')
        
        self.assertIsInstance(processed, ProcessedData)
        self.assertEqual(processed.X_train.shape[1], processed.X_test.shape[1])
        self.assertTrue(len(processed.y_train) > 0)
        self.assertTrue(len(processed.y_test) > 0)
        self.assertEqual(processed.target_name, 'target')
        self.assertIsNotNone(processed.scaler)
    
    def test_preprocess_with_missing_values(self):
        """Test preprocessing data with missing values."""
        processed = self.processor.preprocess(self.data_with_missing, 'target')
        
        self.assertIsInstance(processed, ProcessedData)
        # Check that no NaN values remain
        self.assertFalse(np.isnan(processed.X_train).any())
        self.assertFalse(np.isnan(processed.X_test).any())
    
    def test_preprocess_invalid_data(self):
        """Test preprocessing with invalid data."""
        invalid_data = pd.DataFrame({'feature1': [1, 2], 'target': [0, np.nan]})
        
        with self.assertRaises(ValueError):
            self.processor.preprocess(invalid_data, 'target')
    
    def test_transform_new_data(self):
        """Test transforming new data after fitting."""
        # First fit the processor
        self.processor.preprocess(self.sample_data, 'target')
        
        # Create new data (without target) - include all categories seen during training
        new_data = pd.DataFrame({
            'feature1': [6, 7, 8],
            'feature2': [7.6, 8.7, 9.8],
            'category': ['A', 'B', 'C']  # Include all categories from training
        })
        
        transformed = self.processor.transform_new_data(new_data)
        
        self.assertIsInstance(transformed, np.ndarray)
        self.assertEqual(transformed.shape[0], 3)
    
    def test_transform_new_data_not_fitted(self):
        """Test transforming new data without fitting first."""
        new_data = pd.DataFrame({'feature1': [1, 2]})
        
        with self.assertRaises(ValueError):
            self.processor.transform_new_data(new_data)
    
    def test_handle_missing_values_numeric(self):
        """Test handling missing values in numeric columns."""
        data_with_nan = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5],
            'other': [1, 2, 3, 4, 5]
        })
        
        cleaned = self.processor._handle_missing_values(data_with_nan)
        
        self.assertFalse(cleaned['numeric'].isnull().any())
        self.assertEqual(cleaned['numeric'].iloc[2], data_with_nan['numeric'].median())
    
    def test_handle_missing_values_categorical(self):
        """Test handling missing values in categorical columns."""
        data_with_nan = pd.DataFrame({
            'categorical': ['A', 'B', None, 'A', 'A'],
            'other': [1, 2, 3, 4, 5]
        })
        
        cleaned = self.processor._handle_missing_values(data_with_nan)
        
        self.assertFalse(cleaned['categorical'].isnull().any())
        self.assertEqual(cleaned['categorical'].iloc[2], 'A')  # Mode is 'A'
    
    def test_encode_categorical_features(self):
        """Test encoding of categorical features."""
        data_with_cat = pd.DataFrame({
            'numeric': [1, 2, 3],
            'categorical': ['A', 'B', 'C']
        })
        
        encoded = self.processor._encode_categorical_features(data_with_cat)
        
        # Should have more columns due to one-hot encoding
        self.assertGreater(encoded.shape[1], data_with_cat.shape[1])
        # Original numeric column should remain
        self.assertIn('numeric', encoded.columns)
    
    def test_encode_target_categorical(self):
        """Test encoding categorical target."""
        categorical_target = pd.Series(['class_A', 'class_B', 'class_A', 'class_C'])
        
        encoded = self.processor._encode_target(categorical_target)
        
        self.assertIsInstance(encoded, np.ndarray)
        self.assertTrue(hasattr(self.processor, '_target_encoded'))
        self.assertTrue(self.processor._target_encoded)
    
    def test_encode_target_numeric(self):
        """Test encoding numeric target."""
        numeric_target = pd.Series([1.5, 2.3, 3.1, 4.2])
        
        encoded = self.processor._encode_target(numeric_target)
        
        self.assertIsInstance(encoded, np.ndarray)
        np.testing.assert_array_equal(encoded, numeric_target.values)


class TestSampleDataGenerator(unittest.TestCase):
    """Test cases for SampleDataGenerator class."""
    
    def test_generate_classification_data(self):
        """Test generation of classification data."""
        df = SampleDataGenerator.generate_classification_data(n_samples=100, n_features=3, n_classes=2)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertEqual(len(df.columns), 4)  # 3 features + 1 target
        self.assertIn('target', df.columns)
        self.assertEqual(df['target'].nunique(), 2)
    
    def test_generate_regression_data(self):
        """Test generation of regression data."""
        df = SampleDataGenerator.generate_regression_data(n_samples=50, n_features=2)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 50)
        self.assertEqual(len(df.columns), 3)  # 2 features + 1 target
        self.assertIn('target', df.columns)
        self.assertTrue(df['target'].dtype in ['float64', 'int64'])
    
    def test_generate_iris_like_data(self):
        """Test generation of iris-like data."""
        df = SampleDataGenerator.generate_iris_like_data(n_samples=150)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 150)
        self.assertEqual(len(df.columns), 5)  # 4 features + 1 target
        self.assertIn('species', df.columns)
        self.assertEqual(df['species'].nunique(), 3)
        
        expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        self.assertEqual(list(df.columns), expected_columns)
    
    def test_save_sample_data_classification(self):
        """Test saving classification sample data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test_classification.csv')
            
            SampleDataGenerator.save_sample_data('classification', file_path)
            
            self.assertTrue(os.path.exists(file_path))
            
            # Load and verify the saved data
            df = pd.read_csv(file_path)
            self.assertGreater(len(df), 0)
            self.assertIn('target', df.columns)
    
    def test_save_sample_data_regression(self):
        """Test saving regression sample data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test_regression.csv')
            
            SampleDataGenerator.save_sample_data('regression', file_path)
            
            self.assertTrue(os.path.exists(file_path))
            
            # Load and verify the saved data
            df = pd.read_csv(file_path)
            self.assertGreater(len(df), 0)
            self.assertIn('target', df.columns)
    
    def test_save_sample_data_iris(self):
        """Test saving iris-like sample data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test_iris.csv')
            
            SampleDataGenerator.save_sample_data('iris', file_path)
            
            self.assertTrue(os.path.exists(file_path))
            
            # Load and verify the saved data
            df = pd.read_csv(file_path)
            self.assertEqual(len(df), 150)
            self.assertIn('species', df.columns)
    
    def test_save_sample_data_invalid_type(self):
        """Test saving with invalid data type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test_invalid.csv')
            
            with self.assertRaises(ValueError):
                SampleDataGenerator.save_sample_data('invalid_type', file_path)
    
    def test_reproducibility(self):
        """Test that generated data is reproducible with same random state."""
        df1 = SampleDataGenerator.generate_classification_data(random_state=42)
        df2 = SampleDataGenerator.generate_classification_data(random_state=42)
        
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == '__main__':
    # Create tests directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    unittest.main()