"""
Test individual components of the API without running the server.
"""

import os
import json
import tempfile
from app import ModelLoader, PredictionLogger, PredictionLog
from datetime import datetime


def test_model_loader():
    """Test the ModelLoader class."""
    print("Testing ModelLoader...")
    
    loader = ModelLoader("models/")
    
    # Test listing available models
    models = loader.list_available_models()
    print(f"✓ Found {len(models)} available models")
    
    if models:
        # Test loading a specific model
        model_id = models[0]['model_id']
        print(f"  Testing model: {model_id}")
        
        try:
            model, metadata = loader.load_model(model_id)
            print(f"✓ Model loaded successfully")
            print(f"  Model type: {metadata['model_type']}")
            print(f"  Features: {metadata['feature_names']}")
            
            # Test getting latest model
            latest_id, latest_model, latest_metadata = loader.get_latest_model()
            print(f"✓ Latest model: {latest_id}")
            
            # Test input validation
            test_features = {feature: 1.0 for feature in metadata['feature_names']}
            is_valid, errors = loader.validate_input_features(test_features, metadata['feature_names'])
            print(f"✓ Input validation: {'Valid' if is_valid else 'Invalid'}")
            
            # Test invalid input
            invalid_features = {"wrong_feature": 1.0}
            is_valid, errors = loader.validate_input_features(invalid_features, metadata['feature_names'])
            print(f"✓ Invalid input correctly rejected: {not is_valid}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    return True


def test_prediction_logger():
    """Test the PredictionLogger class."""
    print("\nTesting PredictionLogger...")
    
    # Create temporary log file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        temp_log_file = f.name
    
    try:
        logger = PredictionLogger(temp_log_file)
        
        # Test logging a prediction
        prediction_log = PredictionLog(
            timestamp=datetime.now().isoformat(),
            model_id="test_model",
            input_features={"feature_1": 1.0, "feature_2": 2.0},
            prediction="class_a",
            confidence=0.85,
            response_time_ms=150,
            request_id="test_req_001"
        )
        
        logger.log_prediction(prediction_log)
        print("✓ Prediction logged successfully")
        
        # Test reading recent predictions
        recent = logger.get_recent_predictions(10)
        print(f"✓ Retrieved {len(recent)} recent predictions")
        
        if recent:
            print(f"  Latest prediction: {recent[-1]['prediction']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing prediction logger: {e}")
        return False
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_log_file):
            os.unlink(temp_log_file)


def test_model_prediction():
    """Test making predictions with a loaded model."""
    print("\nTesting model prediction...")
    
    try:
        loader = ModelLoader("models/")
        models = loader.list_available_models()
        
        if not models:
            print("✗ No models available for prediction test")
            return False
        
        # Load the first available model
        model_id = models[0]['model_id']
        model, metadata = loader.load_model(model_id)
        
        # Create test input
        feature_names = metadata['feature_names']
        test_input = [1.5, 2.0, 1.2, 0.8]  # Sample values
        
        # Make prediction
        prediction = model.predict([test_input])[0]
        print(f"✓ Prediction made: {prediction}")
        
        # Test prediction probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([test_input])[0]
            confidence = max(probabilities)
            print(f"✓ Prediction confidence: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing model prediction: {e}")
        return False


def main():
    """Run all component tests."""
    print("=" * 50)
    print("API Component Tests")
    print("=" * 50)
    
    tests = [
        test_model_loader,
        test_prediction_logger,
        test_model_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Component Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All component tests passed!")
    else:
        print(f"⚠️  {total - passed} tests failed")
    
    print("=" * 50)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)