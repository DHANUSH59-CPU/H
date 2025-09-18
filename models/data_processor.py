"""
Data processing module for ML model training and prediction.
Handles data loading, preprocessing, validation, and sample data generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
import os


@dataclass
class ValidationResult:
    """Result of data validation process."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class ProcessedData:
    """Container for processed data."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    target_name: str
    scaler: Optional[StandardScaler] = None
    label_encoder: Optional[LabelEncoder] = None


class DataProcessor:
    """
    Handles data loading, preprocessing, and validation for ML models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataProcessor with optional configuration.
        
        Args:
            config: Dictionary containing processing parameters
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension == '.json':
                return pd.read_json(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {str(e)}")
    
    def validate_data(self, data: pd.DataFrame, target_column: str = None) -> ValidationResult:
        """
        Validate data format and quality.
        
        Args:
            data: DataFrame to validate
            target_column: Name of target column (optional)
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        # Check if DataFrame is empty
        if data.empty:
            errors.append("Dataset is empty")
            return ValidationResult(False, errors, warnings)
        
        # Check for minimum number of rows
        min_rows = self.config.get('min_rows', 10)
        if len(data) < min_rows:
            errors.append(f"Dataset has only {len(data)} rows, minimum required: {min_rows}")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0]
        
        if not missing_columns.empty:
            missing_pct = (missing_columns / len(data) * 100).round(2)
            for col, pct in missing_pct.items():
                if pct > 50:
                    errors.append(f"Column '{col}' has {pct}% missing values")
                elif pct > 20:
                    warnings.append(f"Column '{col}' has {pct}% missing values")
        
        # Validate target column if specified
        if target_column:
            if target_column not in data.columns:
                errors.append(f"Target column '{target_column}' not found in dataset")
            else:
                # Check target column for missing values
                target_missing = data[target_column].isnull().sum()
                if target_missing > 0:
                    errors.append(f"Target column '{target_column}' has {target_missing} missing values")
        
        # Check for duplicate rows
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            warnings.append(f"Dataset contains {duplicates} duplicate rows")
        
        # Check data types
        for col in data.columns:
            if data[col].dtype == 'object':
                unique_values = data[col].nunique()
                if unique_values > len(data) * 0.8:
                    warnings.append(f"Column '{col}' has high cardinality ({unique_values} unique values)")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings)
    
    def preprocess(self, data: pd.DataFrame, target_column: str, 
                  test_size: float = 0.2, random_state: int = 42) -> ProcessedData:
        """
        Preprocess data for machine learning.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            ProcessedData object with train/test splits and preprocessing objects
        """
        # Validate input
        validation_result = self.validate_data(data, target_column)
        if not validation_result.is_valid:
            raise ValueError(f"Data validation failed: {validation_result.errors}")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Encode categorical variables
        X_encoded = self._encode_categorical_features(X)
        
        # Encode target if it's categorical
        y_encoded = self._encode_target(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=test_size, random_state=random_state,
            stratify=y_encoded if self._is_classification_target(y_encoded) else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return ProcessedData(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            feature_names=list(X.columns),
            target_name=target_column,
            scaler=self.scaler,
            label_encoder=self.label_encoder if hasattr(self, '_target_encoded') else None
        )
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        X_clean = X.copy()
        
        for col in X_clean.columns:
            if X_clean[col].isnull().any():
                if X_clean[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    X_clean[col] = X_clean[col].fillna(X_clean[col].median())
                else:
                    # Fill categorical columns with mode
                    mode_value = X_clean[col].mode()
                    if len(mode_value) > 0:
                        X_clean[col] = X_clean[col].fillna(mode_value[0])
        
        return X_clean
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding."""
        X_encoded = X.copy()
        
        # Get categorical columns
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            # Use one-hot encoding for categorical variables
            X_encoded = pd.get_dummies(X_encoded, columns=categorical_cols, drop_first=True)
        
        return X_encoded
    
    def _encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target variable if categorical."""
        if y.dtype == 'object' or y.dtype.name == 'category':
            self._target_encoded = True
            return self.label_encoder.fit_transform(y)
        else:
            self._target_encoded = False
            return y.values
    
    def _is_classification_target(self, y: np.ndarray) -> bool:
        """Determine if target is for classification."""
        return hasattr(self, '_target_encoded') and self._target_encoded
    
    def transform_new_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            data: New data to transform
            
        Returns:
            Transformed data array
        """
        if self.scaler is None:
            raise ValueError("DataProcessor not fitted. Call preprocess() first.")
        
        # Handle missing values
        data_clean = self._handle_missing_values(data)
        
        # Encode categorical features (same as training)
        data_encoded = self._encode_categorical_features(data_clean)
        
        # Scale features
        return self.scaler.transform(data_encoded)


class SampleDataGenerator:
    """
    Generates sample datasets for demo and testing purposes.
    """
    
    @staticmethod
    def generate_classification_data(n_samples: int = 1000, n_features: int = 4, 
                                   n_classes: int = 3, random_state: int = 42) -> pd.DataFrame:
        """
        Generate synthetic classification dataset.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features
            n_classes: Number of target classes
            random_state: Random seed
            
        Returns:
            DataFrame with synthetic classification data
        """
        np.random.seed(random_state)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target with some correlation to features
        weights = np.random.randn(n_features)
        linear_combination = X @ weights
        
        # Create classes based on quantiles
        quantiles = np.quantile(linear_combination, np.linspace(0, 1, n_classes + 1))
        y = np.digitize(linear_combination, quantiles[1:-1])
        
        # Create DataFrame
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        class_names = [f'class_{i}' for i in range(n_classes)]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = [class_names[i] for i in y]
        
        return df
    
    @staticmethod
    def generate_regression_data(n_samples: int = 1000, n_features: int = 4, 
                               noise: float = 0.1, random_state: int = 42) -> pd.DataFrame:
        """
        Generate synthetic regression dataset.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features
            noise: Amount of noise to add
            random_state: Random seed
            
        Returns:
            DataFrame with synthetic regression data
        """
        np.random.seed(random_state)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target with linear relationship plus noise
        weights = np.random.randn(n_features)
        y = X @ weights + np.random.randn(n_samples) * noise
        
        # Create DataFrame
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    @staticmethod
    def generate_iris_like_data(n_samples: int = 150, random_state: int = 42) -> pd.DataFrame:
        """
        Generate iris-like dataset for demo purposes.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed
            
        Returns:
            DataFrame with iris-like data
        """
        np.random.seed(random_state)
        
        # Generate three clusters of data
        n_per_class = n_samples // 3
        
        # Class 0: Small flowers
        class_0 = np.random.multivariate_normal([4.5, 3.0, 1.2, 0.3], 
                                               [[0.1, 0.05, 0.02, 0.01],
                                                [0.05, 0.1, 0.01, 0.005],
                                                [0.02, 0.01, 0.05, 0.02],
                                                [0.01, 0.005, 0.02, 0.01]], n_per_class)
        
        # Class 1: Medium flowers
        class_1 = np.random.multivariate_normal([6.0, 2.8, 4.2, 1.3], 
                                               [[0.2, 0.1, 0.1, 0.05],
                                                [0.1, 0.15, 0.05, 0.02],
                                                [0.1, 0.05, 0.15, 0.08],
                                                [0.05, 0.02, 0.08, 0.05]], n_per_class)
        
        # Class 2: Large flowers
        class_2 = np.random.multivariate_normal([7.2, 3.2, 5.8, 2.1], 
                                               [[0.25, 0.15, 0.12, 0.08],
                                                [0.15, 0.2, 0.08, 0.05],
                                                [0.12, 0.08, 0.2, 0.12],
                                                [0.08, 0.05, 0.12, 0.1]], n_per_class)
        
        # Combine data
        X = np.vstack([class_0, class_1, class_2])
        y = ['setosa'] * n_per_class + ['versicolor'] * n_per_class + ['virginica'] * n_per_class
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        df['species'] = y
        
        return df
    
    @staticmethod
    def save_sample_data(data_type: str = 'classification', file_path: str = 'data/sample_data.csv'):
        """
        Generate and save sample data to file.
        
        Args:
            data_type: Type of data to generate ('classification', 'regression', 'iris')
            file_path: Path to save the data
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if data_type == 'classification':
            df = SampleDataGenerator.generate_classification_data()
        elif data_type == 'regression':
            df = SampleDataGenerator.generate_regression_data()
        elif data_type == 'iris':
            df = SampleDataGenerator.generate_iris_like_data()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        df.to_csv(file_path, index=False)
        print(f"Sample {data_type} data saved to {file_path}")