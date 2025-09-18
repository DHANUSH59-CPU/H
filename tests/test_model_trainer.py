"""
Unit tests for model training pipeline.
Tests ModelTrainer class and related functionality.
"""

import unittest
import tempfile
import os
import shutil
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add models directory to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model_trainer import ModelTrainer, TrainingScript, ModelResult, MetricsDict
from models.data_processor import SampleDataGenerator


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'model_save_path': self.temp_dir,
            'test_size': 0.2,
            'random_state': 42
        }
        self.trainer = ModelTrainer(self.config)
        
        # Create sample data files
        self.classification_data_path = os.path.join(self.temp_dir, 'classification_data.csv')
        self.regression_data_path = os.path.join(self.temp_dir, 'regression_data.csv')
        
        # Generate and save sample data
        clf_data = SampleDataGenerator.generate_classification_data(n_samples=100, random_state=42)
        clf_data.to_csv(self.classification_data_path, index=False)
        
        reg_data = SampleDataGenerator.generate_regression_data(n_samples=100, random_state=42)
        reg_data.to_csv(self.regression_data_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_train_classification_model(self):
        """Test training a classification model."""
        result = self.trainer.train_model(
            self.classification_data_path, 
            target_column='target',
            algorithm='random_forest'
        )
        
        # Check result structure
        self.assertIsInstance(result, ModelResult)
        self.assertIsNotNone(result.model)
        self.assertTrue(result.model_id.startswith('random_forest_clf_'))
        self.assertEqual(result.model_type, 'random_forest_classifier')
        
        # Check metrics
        self.assertIn('accuracy', result.training_metrics)
        self.assertIn('precision', result.training_metrics)
        self.assertIn('recall', result.training_metrics)
        self.assertIn('f1_score', result.training_metrics)
        
        # Check accuracy is reasonable
        self.assertGreater(result.training_metrics['accuracy'], 0.5)
        self.assertGreater(result.validation_metrics['accuracy'], 0.5)
    
    def test_train_regression_model(self):
        """Test training a regression model."""
        result = self.trainer.train_model(
            self.regression_data_path,
            target_column='target',
            algorithm='random_forest'
        )
        
        # Check result structure
        self.assertIsInstance(result, ModelResult)
        self.assertIsNotNone(result.model)
        self.assertTrue(result.model_id.startswith('random_forest_reg_'))
        self.assertEqual(result.model_type, 'random_forest_regressor')
        
        # Check metrics
        self.assertIn('mse', result.training_metrics)
        self.assertIn('mae', result.training_metrics)
        self.assertIn('r2_score', result.training_metrics)
        
        # Check R² score is reasonable
        self.assertGreater(result.training_metrics['r2_score'], 0.5)
    
    def test_different_algorithms(self):
        """Test training with different algorithms."""
        algorithms = ['random_forest', 'logistic_regression']
        
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                result = self.trainer.train_model(
                    self.classification_data_path,
                    target_column='target',
                    algorithm=algorithm
                )
                
                self.assertIsInstance(result, ModelResult)
                self.assertTrue(result.model_id.startswith(f'{algorithm}_clf_'))
                self.assertIn('accuracy', result.training_metrics)
    
    def test_invalid_algorithm(self):
        """Test training with invalid algorithm."""
        with self.assertRaises(ValueError):
            self.trainer.train_model(
                self.classification_data_path,
                target_column='target',
                algorithm='invalid_algorithm'
            )
    
    def test_invalid_data_path(self):
        """Test training with invalid data path."""
        with self.assertRaises(FileNotFoundError):
            self.trainer.train_model(
                'nonexistent_file.csv',
                target_column='target'
            )
    
    def test_evaluate_model_classification(self):
        """Test model evaluation for classification."""
        # Train a simple model first
        result = self.trainer.train_model(
            self.classification_data_path,
            target_column='target',
            algorithm='random_forest'
        )
        
        # Load test data
        data = pd.read_csv(self.classification_data_path)
        processed_data = self.trainer.data_processor.preprocess(data, 'target')
        
        # Evaluate model
        metrics = self.trainer.evaluate_model(
            result.model, 
            processed_data.X_test, 
            processed_data.y_test,
            is_classification=True
        )
        
        self.assertIsInstance(metrics, MetricsDict)
        self.assertIsNotNone(metrics.accuracy)
        self.assertIsNotNone(metrics.precision)
        self.assertIsNotNone(metrics.recall)
        self.assertIsNotNone(metrics.f1_score)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
    
    def test_evaluate_model_regression(self):
        """Test model evaluation for regression."""
        # Train a simple model first
        result = self.trainer.train_model(
            self.regression_data_path,
            target_column='target',
            algorithm='random_forest'
        )
        
        # Load test data
        data = pd.read_csv(self.regression_data_path)
        processed_data = self.trainer.data_processor.preprocess(data, 'target')
        
        # Evaluate model
        metrics = self.trainer.evaluate_model(
            result.model,
            processed_data.X_test,
            processed_data.y_test,
            is_classification=False
        )
        
        self.assertIsInstance(metrics, MetricsDict)
        self.assertIsNotNone(metrics.mse)
        self.assertIsNotNone(metrics.mae)
        self.assertIsNotNone(metrics.r2_score)
        
        # Check that MSE and MAE are non-negative
        self.assertGreaterEqual(metrics.mse, 0.0)
        self.assertGreaterEqual(metrics.mae, 0.0)
    
    def test_save_and_load_model(self):
        """Test saving and loading models."""
        # Train a model
        result = self.trainer.train_model(
            self.classification_data_path,
            target_column='target',
            algorithm='random_forest'
        )
        
        # Save model
        save_success = self.trainer.save_model(result)
        self.assertTrue(save_success)
        
        # Check files exist
        model_path = os.path.join(self.temp_dir, f"{result.model_id}.joblib")
        metadata_path = os.path.join(self.temp_dir, f"{result.model_id}_metadata.json")
        
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(metadata_path))
        
        # Load model
        loaded_model, metadata = self.trainer.load_model(model_path)
        
        self.assertIsNotNone(loaded_model)
        self.assertEqual(metadata['model_id'], result.model_id)
        self.assertEqual(metadata['model_type'], result.model_type)
        
        # Test predictions are the same
        test_data = np.array([[1, 2, 3, 4]])
        original_pred = result.model.predict(test_data)
        loaded_pred = loaded_model.predict(test_data)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
    
    def test_load_nonexistent_model(self):
        """Test loading a model that doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.trainer.load_model('nonexistent_model.joblib')
    
    def test_is_classification_problem(self):
        """Test classification problem detection."""
        # Test with string labels (classification)
        y_str = np.array(['class_0', 'class_1', 'class_0'])
        self.assertTrue(self.trainer._is_classification_problem(y_str))
        
        # Test with integer labels (classification)
        y_int = np.array([0, 1, 2, 0, 1])
        self.assertTrue(self.trainer._is_classification_problem(y_int))
        
        # Test with continuous values (regression)
        y_float = np.random.randn(100)
        self.assertFalse(self.trainer._is_classification_problem(y_float))
    
    def test_generate_model_id(self):
        """Test model ID generation."""
        model_id_clf = self.trainer._generate_model_id('random_forest', True)
        model_id_reg = self.trainer._generate_model_id('random_forest', False)
        
        self.assertTrue(model_id_clf.startswith('random_forest_clf_'))
        self.assertTrue(model_id_reg.startswith('random_forest_reg_'))
        
        # IDs should be different
        self.assertNotEqual(model_id_clf, model_id_reg)
    
    def test_metrics_dict_to_dict(self):
        """Test MetricsDict conversion to dictionary."""
        metrics = MetricsDict(
            accuracy=0.95,
            precision=0.90,
            recall=None,  # Should be excluded
            f1_score=0.92
        )
        
        result_dict = metrics.to_dict()
        
        self.assertIn('accuracy', result_dict)
        self.assertIn('precision', result_dict)
        self.assertIn('f1_score', result_dict)
        self.assertNotIn('recall', result_dict)  # None values excluded
        
        self.assertEqual(result_dict['accuracy'], 0.95)


class TestTrainingScript(unittest.TestCase):
    """Test cases for TrainingScript class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'model_save_path': self.temp_dir,
            'test_size': 0.2,
            'random_state': 42
        }
        self.script = TrainingScript(self.config)
        
        # Create sample data
        self.data_path = os.path.join(self.temp_dir, 'test_data.csv')
        data = SampleDataGenerator.generate_classification_data(n_samples=100, random_state=42)
        data.to_csv(self.data_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_train_and_save_model_success(self):
        """Test successful model training and saving."""
        result = self.script.train_and_save_model(
            self.data_path,
            target_column='target',
            algorithm='random_forest'
        )
        
        self.assertTrue(result['success'])
        self.assertIn('model_id', result)
        self.assertIn('training_metrics', result)
        self.assertIn('validation_metrics', result)
        self.assertTrue(result['saved'])
    
    def test_train_and_save_model_invalid_data(self):
        """Test training with invalid data path."""
        result = self.script.train_and_save_model(
            'nonexistent_file.csv',
            target_column='target'
        )
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    @patch('builtins.print')  # Mock print to avoid output during tests
    def test_retrain_model_nonexistent(self, mock_print):
        """Test retraining a model that doesn't exist."""
        result = self.script.retrain_model('nonexistent_model', self.data_path)
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    @patch('builtins.print')  # Mock print to avoid output during tests
    def test_retrain_model_success(self, mock_print):
        """Test successful model retraining."""
        # First train a model
        initial_result = self.script.train_and_save_model(
            self.data_path,
            target_column='target',
            algorithm='random_forest'
        )
        
        self.assertTrue(initial_result['success'])
        
        # Create new training data
        new_data_path = os.path.join(self.temp_dir, 'new_data.csv')
        new_data = SampleDataGenerator.generate_classification_data(n_samples=150, random_state=123)
        new_data.to_csv(new_data_path, index=False)
        
        # Retrain model
        retrain_result = self.script.retrain_model(
            initial_result['model_id'],
            new_data_path
        )
        
        self.assertTrue(retrain_result['success'])
        self.assertNotEqual(retrain_result['model_id'], initial_result['model_id'])


if __name__ == '__main__':
    unittest.main()