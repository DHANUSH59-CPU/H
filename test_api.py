"""
Test script for the ML prediction API.
Tests various endpoints and scenarios.
"""

import requests
import json
import time
from typing import Dict, Any


class APITester:
    """Test the ML prediction API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize API tester.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        print("Testing health check endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Health check passed: {data['status']}")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Health check error: {e}")
            return False
    
    def test_model_status(self) -> bool:
        """Test the model status endpoint."""
        print("\nTesting model status endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/model/status")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Model status retrieved")
                print(f"  Total models: {data['total_models']}")
                
                if data['current_model']:
                    current = data['current_model']
                    print(f"  Current model: {current['model_id']}")
                    print(f"  Model type: {current['model_type']}")
                    print(f"  Features: {current['feature_names']}")
                
                return True
            else:
                print(f"✗ Model status failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Model status error: {e}")
            return False
    
    def test_prediction_valid(self) -> bool:
        """Test prediction with valid input."""
        print("\nTesting prediction with valid input...")
        
        # Sample input based on the model metadata we saw
        test_input = {
            "features": {
                "feature_1": 1.5,
                "feature_2": 2.0,
                "feature_3": 1.2,
                "feature_4": 0.8
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_input,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Prediction successful")
                print(f"  Prediction: {data['prediction']}")
                print(f"  Model: {data['model_id']}")
                print(f"  Confidence: {data.get('confidence', 'N/A')}")
                print(f"  Response time: {data['response_time_ms']}ms")
                return True
            else:
                print(f"✗ Prediction failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return False
    
    def test_prediction_invalid_features(self) -> bool:
        """Test prediction with invalid features."""
        print("\nTesting prediction with invalid features...")
        
        # Missing required features
        test_input = {
            "features": {
                "wrong_feature": 1.5,
                "another_wrong": 2.0
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_input,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 400:
                data = response.json()
                print(f"✓ Invalid features correctly rejected")
                print(f"  Error: {data['error']}")
                return True
            else:
                print(f"✗ Invalid features not properly handled: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Invalid features test error: {e}")
            return False
    
    def test_prediction_missing_json(self) -> bool:
        """Test prediction without JSON payload."""
        print("\nTesting prediction without JSON...")
        
        try:
            response = self.session.post(f"{self.base_url}/predict")
            
            if response.status_code == 400:
                data = response.json()
                print(f"✓ Missing JSON correctly rejected")
                print(f"  Error: {data['error']}")
                return True
            else:
                print(f"✗ Missing JSON not properly handled: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Missing JSON test error: {e}")
            return False
    
    def test_prediction_specific_model(self) -> bool:
        """Test prediction with specific model ID."""
        print("\nTesting prediction with specific model ID...")
        
        # First get available models
        try:
            status_response = self.session.get(f"{self.base_url}/model/status")
            if status_response.status_code != 200:
                print("✗ Could not get model status for specific model test")
                return False
            
            status_data = status_response.json()
            if not status_data['available_models']:
                print("✗ No models available for specific model test")
                return False
            
            # Use the first available model
            model_id = status_data['available_models'][0]['model_id']
            
            test_input = {
                "features": {
                    "feature_1": 2.1,
                    "feature_2": 1.8,
                    "feature_3": 1.5,
                    "feature_4": 0.9
                },
                "model_id": model_id
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_input,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Specific model prediction successful")
                print(f"  Used model: {data['model_id']}")
                print(f"  Prediction: {data['prediction']}")
                return True
            else:
                print(f"✗ Specific model prediction failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Specific model test error: {e}")
            return False
    
    def test_metrics_endpoint(self) -> bool:
        """Test the metrics endpoint."""
        print("\nTesting metrics endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Metrics retrieved successfully")
                print(f"  Total predictions: {data['total_predictions']}")
                print(f"  Average response time: {data['average_response_time_ms']}ms")
                print(f"  Model usage: {data['model_usage']}")
                return True
            else:
                print(f"✗ Metrics failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Metrics error: {e}")
            return False
    
    def test_nonexistent_endpoint(self) -> bool:
        """Test accessing a non-existent endpoint."""
        print("\nTesting non-existent endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/nonexistent")
            
            if response.status_code == 404:
                print(f"✓ Non-existent endpoint correctly returns 404")
                return True
            else:
                print(f"✗ Non-existent endpoint returned: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Non-existent endpoint test error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all API tests."""
        print("=" * 60)
        print("ML Prediction API Test Suite")
        print("=" * 60)
        
        tests = [
            self.test_health_check,
            self.test_model_status,
            self.test_prediction_valid,
            self.test_prediction_invalid_features,
            self.test_prediction_missing_json,
            self.test_prediction_specific_model,
            self.test_metrics_endpoint,
            self.test_nonexistent_endpoint
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
            time.sleep(0.5)  # Small delay between tests
        
        print("\n" + "=" * 60)
        print(f"Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print(f"⚠️  {total - passed} tests failed")
        
        print("=" * 60)
        
        return passed == total


def main():
    """Main function to run API tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ML Prediction API')
    parser.add_argument('--url', default='http://localhost:5000', 
                       help='Base URL of the API server')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for server to be ready before testing')
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.wait:
        print("Waiting for server to be ready...")
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{args.url}/health", timeout=2)
                if response.status_code == 200:
                    print("Server is ready!")
                    break
            except:
                pass
            
            time.sleep(1)
            print(f"Attempt {attempt + 1}/{max_attempts}...")
        else:
            print("Server did not become ready in time")
            return False
    
    return tester.run_all_tests()


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)