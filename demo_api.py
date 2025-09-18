"""
Demonstration script for the ML Prediction API.
Shows how to start the server and make predictions.
"""

import subprocess
import time
import requests
import json
import threading
from typing import Optional


class APIDemo:
    """Demonstration of the ML Prediction API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize API demo.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.server_process: Optional[subprocess.Popen] = None
    
    def start_server(self) -> bool:
        """Start the Flask server in a subprocess."""
        print("Starting ML Prediction API server...")
        
        try:
            # Start server in background
            self.server_process = subprocess.Popen(
                ["python", "app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to be ready
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=2)
                    if response.status_code == 200:
                        print("✓ Server is ready!")
                        return True
                except:
                    pass
                
                time.sleep(1)
                print(f"  Waiting... ({attempt + 1}/{max_attempts})")
            
            print("✗ Server failed to start in time")
            return False
            
        except Exception as e:
            print(f"✗ Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop the Flask server."""
        if self.server_process:
            print("Stopping server...")
            self.server_process.terminate()
            self.server_process.wait()
            print("✓ Server stopped")
    
    def demo_health_check(self):
        """Demonstrate health check endpoint."""
        print("\n" + "="*50)
        print("1. Health Check")
        print("="*50)
        
        response = requests.get(f"{self.base_url}/health")
        print(f"GET {self.base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    def demo_model_status(self):
        """Demonstrate model status endpoint."""
        print("\n" + "="*50)
        print("2. Model Status")
        print("="*50)
        
        response = requests.get(f"{self.base_url}/model/status")
        print(f"GET {self.base_url}/model/status")
        print(f"Status: {response.status_code}")
        
        data = response.json()
        print(f"Response:")
        print(f"  Status: {data['status']}")
        print(f"  Total models: {data['total_models']}")
        
        if data['current_model']:
            current = data['current_model']
            print(f"  Current model: {current['model_id']}")
            print(f"  Model type: {current['model_type']}")
            print(f"  Features: {current['feature_names']}")
        
        return data
    
    def demo_prediction(self, model_info: dict):
        """Demonstrate prediction endpoint."""
        print("\n" + "="*50)
        print("3. Making Predictions")
        print("="*50)
        
        if not model_info['current_model']:
            print("No models available for prediction demo")
            return
        
        current_model = model_info['current_model']
        feature_names = current_model['feature_names']
        
        # Create sample input
        sample_input = {
            "features": {feature: round(1.5 + i * 0.3, 2) for i, feature in enumerate(feature_names)}
        }
        
        print(f"Sample input: {json.dumps(sample_input, indent=2)}")
        
        # Make prediction
        response = requests.post(
            f"{self.base_url}/predict",
            json=sample_input,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"\nPOST {self.base_url}/predict")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response:")
            print(f"  Prediction: {data['prediction']}")
            print(f"  Model used: {data['model_id']}")
            print(f"  Confidence: {data.get('confidence', 'N/A')}")
            print(f"  Response time: {data['response_time_ms']}ms")
        else:
            print(f"Error: {response.text}")
    
    def demo_prediction_with_specific_model(self, model_info: dict):
        """Demonstrate prediction with specific model ID."""
        print("\n" + "="*50)
        print("4. Prediction with Specific Model")
        print("="*50)
        
        available_models = model_info['available_models']
        if len(available_models) < 2:
            print("Need at least 2 models for this demo")
            return
        
        # Use the second model
        target_model = available_models[1]
        model_id = target_model['model_id']
        feature_names = target_model['feature_names']
        
        sample_input = {
            "features": {feature: round(2.0 + i * 0.4, 2) for i, feature in enumerate(feature_names)},
            "model_id": model_id
        }
        
        print(f"Using specific model: {model_id}")
        print(f"Input: {json.dumps(sample_input, indent=2)}")
        
        response = requests.post(
            f"{self.base_url}/predict",
            json=sample_input,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"\nPOST {self.base_url}/predict")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response:")
            print(f"  Prediction: {data['prediction']}")
            print(f"  Model used: {data['model_id']}")
            print(f"  Model type: {data['model_type']}")
        else:
            print(f"Error: {response.text}")
    
    def demo_invalid_input(self):
        """Demonstrate error handling with invalid input."""
        print("\n" + "="*50)
        print("5. Error Handling - Invalid Input")
        print("="*50)
        
        # Test with missing features
        invalid_input = {
            "features": {
                "wrong_feature": 1.0,
                "another_wrong": 2.0
            }
        }
        
        print(f"Invalid input: {json.dumps(invalid_input, indent=2)}")
        
        response = requests.post(
            f"{self.base_url}/predict",
            json=invalid_input,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"\nPOST {self.base_url}/predict")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 400:
            data = response.json()
            print(f"Error correctly handled:")
            print(f"  Error: {data['error']}")
            print(f"  Details: {data.get('details', [])}")
        else:
            print(f"Unexpected response: {response.text}")
    
    def demo_metrics(self):
        """Demonstrate metrics endpoint."""
        print("\n" + "="*50)
        print("6. Prediction Metrics")
        print("="*50)
        
        response = requests.get(f"{self.base_url}/metrics")
        print(f"GET {self.base_url}/metrics")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Metrics:")
            print(f"  Total predictions: {data['total_predictions']}")
            print(f"  Average response time: {data['average_response_time_ms']}ms")
            print(f"  Model usage: {data['model_usage']}")
            
            if data['recent_predictions']:
                print(f"  Recent predictions: {len(data['recent_predictions'])}")
        else:
            print(f"Error: {response.text}")
    
    def run_demo(self):
        """Run the complete API demonstration."""
        print("ML Prediction API Demonstration")
        print("="*60)
        
        # Start server
        if not self.start_server():
            return False
        
        try:
            # Run demonstrations
            self.demo_health_check()
            
            model_info = self.demo_model_status()
            
            self.demo_prediction(model_info)
            
            self.demo_prediction_with_specific_model(model_info)
            
            self.demo_invalid_input()
            
            self.demo_metrics()
            
            print("\n" + "="*60)
            print("🎉 API Demonstration Complete!")
            print("="*60)
            
            print("\nAPI Endpoints Summary:")
            print(f"  Health Check:    GET  {self.base_url}/health")
            print(f"  Model Status:    GET  {self.base_url}/model/status")
            print(f"  Make Prediction: POST {self.base_url}/predict")
            print(f"  Get Metrics:     GET  {self.base_url}/metrics")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Demo error: {e}")
            return False
            
        finally:
            self.stop_server()


def main():
    """Main function to run the API demonstration."""
    demo = APIDemo()
    success = demo.run_demo()
    return success


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)