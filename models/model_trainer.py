"""
Model training pipeline for automated ML deployment and monitoring system.
Handles model training, evaluation, and serialization using scikit-learn.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score

from .data_processor import DataProcessor, ProcessedData


@dataclass
class ModelResult:
    """Result of model training process."""
    model: Any
    model_id: str
    model_type: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    feature_names: List[str]
    target_name: str
    created_at: datetime
    training_data_size: int
    test_data_size: int


@dataclass
class MetricsDict:
    """Container for model evaluation metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    cross_val_scores: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ModelTrainer:
    """
    Handles model training, evaluation, and management for the ML pipeline.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config: Dictionary containing training parameters
        """
        self.config = config or {}
        self.data_processor = DataProcessor(config)
        
        # Available algorithms
        self.algorithms = {
            'random_forest': {
                'classifier': RandomForestClassifier,
                'regressor': RandomForestRegressor,
                'params': {'n_estimators': 100, 'random_state': 42}
            },
            'logistic_regression': {
                'classifier': LogisticRegression,
                'regressor': LinearRegression,
                'params': {'random_state': 42}
            },
            'svm': {
                'classifier': SVC,
                'regressor': SVR,
                'params': {'random_state': 42}
            }
        }
    
    def train_model(self, data_path: str, target_column: str = 'target', 
                   algorithm: str = 'random_forest') -> ModelResult:
        """
        Train a machine learning model on the provided data.
        
        Args:
            data_path: Path to training data file
            target_column: Name of target column
            algorithm: Algorithm to use for training
            
        Returns:
            ModelResult containing trained model and metrics
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If algorithm is not supported or data is invalid
        """
        # Load and preprocess data
        raw_data = self.data_processor.load_data(data_path)
        processed_data = self.data_processor.preprocess(
            raw_data, 
            target_column,
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42)
        )
        
        # Determine if this is classification or regression
        is_classification = self._is_classification_problem(processed_data.y_train)
        
        # Get appropriate model
        model = self._get_model(algorithm, is_classification)
        
        # Train model
        model.fit(processed_data.X_train, processed_data.y_train)
        
        # Evaluate model
        training_metrics = self.evaluate_model(
            model, processed_data.X_train, processed_data.y_train, is_classification
        )
        validation_metrics = self.evaluate_model(
            model, processed_data.X_test, processed_data.y_test, is_classification
        )
        
        # Generate model ID
        model_id = self._generate_model_id(algorithm, is_classification)
        
        # Create result
        result = ModelResult(
            model=model,
            model_id=model_id,
            model_type=f"{algorithm}_{'classifier' if is_classification else 'regressor'}",
            training_metrics=training_metrics.to_dict(),
            validation_metrics=validation_metrics.to_dict(),
            feature_names=processed_data.feature_names,
            target_name=processed_data.target_name,
            created_at=datetime.now(),
            training_data_size=len(processed_data.X_train),
            test_data_size=len(processed_data.X_test)
        )
        
        return result
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                      is_classification: bool = True) -> MetricsDict:
        """
        Evaluate model performance using appropriate metrics.
        
        Args:
            model: Trained model to evaluate
            X: Feature data
            y: Target data
            is_classification: Whether this is a classification problem
            
        Returns:
            MetricsDict containing evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X)
        
        metrics = MetricsDict()
        
        if is_classification:
            # Classification metrics
            metrics.accuracy = accuracy_score(y, y_pred)
            
            # Handle multiclass vs binary classification
            average_method = 'weighted' if len(np.unique(y)) > 2 else 'binary'
            
            metrics.precision = precision_score(y, y_pred, average=average_method, zero_division=0)
            metrics.recall = recall_score(y, y_pred, average=average_method, zero_division=0)
            metrics.f1_score = f1_score(y, y_pred, average=average_method, zero_division=0)
            
            # Cross-validation scores
            try:
                cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                metrics.cross_val_scores = cv_scores.tolist()
            except Exception:
                # Skip cross-validation if it fails (e.g., not enough samples)
                pass
                
        else:
            # Regression metrics
            metrics.mse = mean_squared_error(y, y_pred)
            metrics.mae = mean_absolute_error(y, y_pred)
            metrics.r2_score = r2_score(y, y_pred)
            
            # Cross-validation scores
            try:
                cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                metrics.cross_val_scores = cv_scores.tolist()
            except Exception:
                # Skip cross-validation if it fails
                pass
        
        return metrics
    
    def save_model(self, model_result: ModelResult, model_path: str = None) -> bool:
        """
        Save trained model and metadata to disk.
        
        Args:
            model_result: ModelResult containing model and metadata
            model_path: Optional custom path for saving model
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Determine save path
            if model_path is None:
                model_dir = self.config.get('model_save_path', 'models/')
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{model_result.model_id}.joblib")
            
            # Save model using joblib
            joblib.dump(model_result.model, model_path)
            
            # Save metadata
            metadata_path = model_path.replace('.joblib', '_metadata.json')
            metadata = {
                'model_id': model_result.model_id,
                'model_type': model_result.model_type,
                'training_metrics': model_result.training_metrics,
                'validation_metrics': model_result.validation_metrics,
                'feature_names': model_result.feature_names,
                'target_name': model_result.target_name,
                'created_at': model_result.created_at.isoformat(),
                'training_data_size': model_result.training_data_size,
                'test_data_size': model_result.test_data_size,
                'model_file_path': model_path
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Model saved successfully to {model_path}")
            print(f"Metadata saved to {metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path: str) -> Tuple[Any, Dict]:
        """
        Load saved model and metadata from disk.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Tuple of (model, metadata_dict)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model cannot be loaded
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = model_path.replace('.joblib', '_metadata.json')
            metadata = {}
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            return model, metadata
            
        except Exception as e:
            raise ValueError(f"Error loading model from {model_path}: {str(e)}")
    
    def _is_classification_problem(self, y: np.ndarray) -> bool:
        """
        Determine if the target variable indicates a classification problem.
        
        Args:
            y: Target variable array
            
        Returns:
            True if classification, False if regression
        """
        # Check if target is integer type and has limited unique values
        unique_values = len(np.unique(y))
        
        # If target has string/object dtype, it's classification
        if hasattr(y, 'dtype') and y.dtype == 'object':
            return True
        
        # If target is integer and has few unique values, likely classification
        if np.issubdtype(y.dtype, np.integer) and unique_values <= 20:
            return True
        
        # If target is float but has few unique values, might be classification
        if unique_values <= 10:
            return True
        
        # Otherwise, assume regression
        return False
    
    def _get_model(self, algorithm: str, is_classification: bool) -> Any:
        """
        Get appropriate model instance based on algorithm and problem type.
        
        Args:
            algorithm: Algorithm name
            is_classification: Whether this is a classification problem
            
        Returns:
            Instantiated model object
            
        Raises:
            ValueError: If algorithm is not supported
        """
        if algorithm not in self.algorithms:
            available = list(self.algorithms.keys())
            raise ValueError(f"Algorithm '{algorithm}' not supported. Available: {available}")
        
        algo_config = self.algorithms[algorithm]
        model_type = 'classifier' if is_classification else 'regressor'
        
        if model_type not in algo_config:
            raise ValueError(f"Algorithm '{algorithm}' doesn't support {model_type}")
        
        model_class = algo_config[model_type]
        params = algo_config['params'].copy()
        
        # Adjust parameters based on algorithm and problem type
        if algorithm == 'logistic_regression' and not is_classification:
            # LinearRegression doesn't have random_state parameter
            params.pop('random_state', None)
        
        return model_class(**params)
    
    def _generate_model_id(self, algorithm: str, is_classification: bool) -> str:
        """
        Generate unique model identifier.
        
        Args:
            algorithm: Algorithm name
            is_classification: Whether this is a classification problem
            
        Returns:
            Unique model ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        problem_type = "clf" if is_classification else "reg"
        return f"{algorithm}_{problem_type}_{timestamp}"


class TrainingScript:
    """
    Programmatic interface for triggering model training.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize training script with configuration.
        
        Args:
            config: Dictionary containing training parameters
        """
        self.config = config or {}
        self.trainer = ModelTrainer(config)
    
    def train_and_save_model(self, data_path: str, target_column: str = 'target',
                           algorithm: str = 'random_forest', 
                           save_path: str = None) -> Dict[str, Any]:
        """
        Complete training pipeline: train model, evaluate, and save.
        
        Args:
            data_path: Path to training data
            target_column: Name of target column
            algorithm: Algorithm to use
            save_path: Optional custom save path
            
        Returns:
            Dictionary with training results and model info
        """
        try:
            print(f"Starting model training with {algorithm}...")
            print(f"Data: {data_path}, Target: {target_column}")
            
            # Train model
            result = self.trainer.train_model(data_path, target_column, algorithm)
            
            # Print training results
            print(f"\nTraining completed successfully!")
            print(f"Model ID: {result.model_id}")
            print(f"Model Type: {result.model_type}")
            print(f"Training Data Size: {result.training_data_size}")
            print(f"Test Data Size: {result.test_data_size}")
            
            print(f"\nTraining Metrics:")
            for metric, value in result.training_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
            
            print(f"\nValidation Metrics:")
            for metric, value in result.validation_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
            
            # Save model
            save_success = self.trainer.save_model(result, save_path)
            
            if save_success:
                print(f"\nModel saved successfully!")
            else:
                print(f"\nWarning: Model training succeeded but saving failed.")
            
            # Return results
            return {
                'success': True,
                'model_id': result.model_id,
                'model_type': result.model_type,
                'training_metrics': result.training_metrics,
                'validation_metrics': result.validation_metrics,
                'feature_names': result.feature_names,
                'target_name': result.target_name,
                'created_at': result.created_at.isoformat(),
                'saved': save_success
            }
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            print(f"\nError: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg
            }
    
    def retrain_model(self, model_id: str, new_data_path: str) -> Dict[str, Any]:
        """
        Retrain an existing model with new data.
        
        Args:
            model_id: ID of existing model to retrain
            new_data_path: Path to new training data
            
        Returns:
            Dictionary with retraining results
        """
        try:
            # Find existing model
            model_dir = self.config.get('model_save_path', 'models/')
            model_path = os.path.join(model_dir, f"{model_id}.joblib")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model {model_id} not found")
            
            # Load existing model metadata
            _, metadata = self.trainer.load_model(model_path)
            
            # Extract training parameters from metadata
            target_column = metadata.get('target_name', 'target')
            model_type = metadata.get('model_type', 'random_forest_classifier')
            
            # Extract algorithm from model type (e.g., 'random_forest_classifier' -> 'random_forest')
            if '_classifier' in model_type:
                algorithm = model_type.replace('_classifier', '')
            elif '_regressor' in model_type:
                algorithm = model_type.replace('_regressor', '')
            else:
                # Fallback to first part before underscore
                algorithm = model_type.split('_')[0]
            
            print(f"Retraining model {model_id}...")
            print(f"Original algorithm: {algorithm}")
            print(f"New data: {new_data_path}")
            
            # Train new model with same parameters
            result = self.train_and_save_model(
                new_data_path, target_column, algorithm
            )
            
            if result['success']:
                print(f"Retraining completed! New model ID: {result['model_id']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Retraining failed: {str(e)}"
            print(f"\nError: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg
            }


def main():
    """
    Main function for command-line usage of training script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--data', required=True, help='Path to training data')
    parser.add_argument('--target', default='target', help='Target column name')
    parser.add_argument('--algorithm', default='random_forest', 
                       choices=['random_forest', 'logistic_regression', 'svm'],
                       help='Algorithm to use')
    parser.add_argument('--config', help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Run training
    script = TrainingScript(config)
    result = script.train_and_save_model(args.data, args.target, args.algorithm)
    
    if not result['success']:
        exit(1)


if __name__ == '__main__':
    main()