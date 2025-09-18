#!/usr/bin/env python3
"""
Test script for API documentation and testing interface functionality.
Tests the new health check endpoints and API documentation features.
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List


class APIDocumentationTester:
    """Test the API documentation and health check endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """Initialize the tester with base URL."""
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"   {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details
        })
    
    def test_basic_health_check(self) -> bool:
        """Test the basic health check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code != 200:
                self.log_test("Basic Health Check", False, f"Expected 200, got {response.status_code}")
                return False
            
            data = response.json()
            required_fields = ['status', 'timestamp', 'service', 'version']
            
            for field in required_fields:
                if field not in data:
                    self.log_test("Basic Health Check", False, f"Missing field: {field}")
                    return False
            
            if data['status'] != 'healthy':
                self.log_test("Basic Health Check", False, f"Status is {data['status']}, expected 'healthy'")
                return False
            
            self.log_test("Basic Health Check", True, f"Response time: {response.elapsed.total_seconds():.3f}s")
            return True
            
        except Exception as e:
            self.log_test("Basic Health Check", False, f"Exception: {str(e)}")
            return False
    
    def test_detailed_health_check(self) -> bool:
        """Test the detailed health check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health/detailed")
            
            if response.status_code not in [200, 503]:
                self.log_test("Detailed Health Check", False, f"Unexpected status code: {response.status_code}")
                return False
            
            data = response.json()
            required_fields = ['status', 'timestamp', 'service', 'version', 'components']
            
            for field in required_fields:
                if field not in data:
                    self.log_test("Detailed Health Check", False, f"Missing field: {field}")
                    return False
            
            # Check components
            expected_components = ['models', 'monitoring', 'retraining', 'filesystem']
            components = data.get('components', {})
            
            for component in expected_components:
                if component not in components:
                    self.log_test("Detailed Health Check", False, f"Missing component: {component}")
                    return False
                
                if 'status' not in components[component]:
                    self.log_test("Detailed Health Check", False, f"Component {component} missing status")
                    return False
            
            self.log_test("Detailed Health Check", True, f"Status: {data['status']}, Components: {len(components)}")
            return True
            
        except Exception as e:
            self.log_test("Detailed Health Check", False, f"Exception: {str(e)}")
            return False
    
    def test_readiness_check(self) -> bool:
        """Test the readiness check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health/readiness")
            
            if response.status_code not in [200, 503]:
                self.log_test("Readiness Check", False, f"Unexpected status code: {response.status_code}")
                return False
            
            data = response.json()
            required_fields = ['status', 'timestamp']
            
            for field in required_fields:
                if field not in data:
                    self.log_test("Readiness Check", False, f"Missing field: {field}")
                    return False
            
            status = data['status']
            if status not in ['ready', 'not_ready', 'error']:
                self.log_test("Readiness Check", False, f"Invalid status: {status}")
                return False
            
            self.log_test("Readiness Check", True, f"Status: {status}")
            return True
            
        except Exception as e:
            self.log_test("Readiness Check", False, f"Exception: {str(e)}")
            return False
    
    def test_liveness_check(self) -> bool:
        """Test the liveness check endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health/liveness")
            
            if response.status_code != 200:
                self.log_test("Liveness Check", False, f"Expected 200, got {response.status_code}")
                return False
            
            data = response.json()
            required_fields = ['status', 'timestamp']
            
            for field in required_fields:
                if field not in data:
                    self.log_test("Liveness Check", False, f"Missing field: {field}")
                    return False
            
            if data['status'] != 'alive':
                self.log_test("Liveness Check", False, f"Status is {data['status']}, expected 'alive'")
                return False
            
            uptime = data.get('uptime_seconds')
            uptime_info = f", uptime: {uptime}s" if uptime is not None else ""
            
            self.log_test("Liveness Check", True, f"Status: alive{uptime_info}")
            return True
            
        except Exception as e:
            self.log_test("Liveness Check", False, f"Exception: {str(e)}")
            return False
    
    def test_api_documentation_page(self) -> bool:
        """Test that the API documentation page is accessible."""
        try:
            response = self.session.get(f"{self.base_url}/api-docs")
            
            if response.status_code != 200:
                self.log_test("API Documentation Page", False, f"Expected 200, got {response.status_code}")
                return False
            
            content = response.text
            
            # Check for key content
            required_content = [
                "ML Prediction API Documentation",
                "/health",
                "/predict",
                "/model/status",
                "Code Examples"
            ]
            
            for content_check in required_content:
                if content_check not in content:
                    self.log_test("API Documentation Page", False, f"Missing content: {content_check}")
                    return False
            
            self.log_test("API Documentation Page", True, f"Content length: {len(content)} chars")
            return True
            
        except Exception as e:
            self.log_test("API Documentation Page", False, f"Exception: {str(e)}")
            return False
    
    def test_api_testing_interface(self) -> bool:
        """Test that the API testing interface is accessible."""
        try:
            response = self.session.get(f"{self.base_url}/api-test")
            
            if response.status_code != 200:
                self.log_test("API Testing Interface", False, f"Expected 200, got {response.status_code}")
                return False
            
            content = response.text
            
            # Check for key interface elements
            required_elements = [
                "API Documentation & Testing",
                "Health Check",
                "Live Testing",
                "Code Examples",
                "test-endpoint-btn"
            ]
            
            for element in required_elements:
                if element not in content:
                    self.log_test("API Testing Interface", False, f"Missing element: {element}")
                    return False
            
            self.log_test("API Testing Interface", True, "All required elements present")
            return True
            
        except Exception as e:
            self.log_test("API Testing Interface", False, f"Exception: {str(e)}")
            return False
    
    def test_health_check_performance(self) -> bool:
        """Test health check endpoint performance."""
        try:
            times = []
            
            for i in range(5):
                start_time = time.time()
                response = self.session.get(f"{self.base_url}/health")
                end_time = time.time()
                
                if response.status_code != 200:
                    self.log_test("Health Check Performance", False, f"Request {i+1} failed with {response.status_code}")
                    return False
                
                times.append(end_time - start_time)
            
            avg_time = sum(times) * 1000 / len(times)  # Convert to ms
            max_time = max(times) * 1000
            
            # Health checks should be fast (under 100ms average)
            if avg_time > 100:
                self.log_test("Health Check Performance", False, f"Average time {avg_time:.1f}ms exceeds 100ms")
                return False
            
            self.log_test("Health Check Performance", True, f"Avg: {avg_time:.1f}ms, Max: {max_time:.1f}ms")
            return True
            
        except Exception as e:
            self.log_test("Health Check Performance", False, f"Exception: {str(e)}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling in health endpoints."""
        try:
            # Test non-existent endpoint
            response = self.session.get(f"{self.base_url}/health/nonexistent")
            
            if response.status_code != 404:
                self.log_test("Error Handling", False, f"Expected 404 for non-existent endpoint, got {response.status_code}")
                return False
            
            data = response.json()
            if 'error' not in data:
                self.log_test("Error Handling", False, "404 response missing error field")
                return False
            
            self.log_test("Error Handling", True, "Proper 404 handling for non-existent endpoints")
            return True
            
        except Exception as e:
            self.log_test("Error Handling", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all API documentation tests."""
        print("🧪 Running API Documentation Tests...")
        print("=" * 50)
        
        tests = [
            self.test_basic_health_check,
            self.test_detailed_health_check,
            self.test_readiness_check,
            self.test_liveness_check,
            self.test_api_documentation_page,
            self.test_api_testing_interface,
            self.test_health_check_performance,
            self.test_error_handling
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                print(f"✗ FAIL: {test.__name__} - Exception: {str(e)}")
                results.append(False)
        
        # Summary
        print("\n" + "=" * 50)
        passed = sum(results)
        total = len(results)
        
        print(f"📊 Test Summary: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All API documentation tests passed!")
            return True
        else:
            print(f"❌ {total - passed} tests failed")
            return False
    
    def print_test_summary(self):
        """Print detailed test summary."""
        print("\n📋 Detailed Test Results:")
        print("-" * 50)
        
        for result in self.test_results:
            status = "✓" if result['success'] else "✗"
            print(f"{status} {result['test']}")
            if result['details']:
                print(f"   {result['details']}")


def main():
    """Main function to run API documentation tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test API documentation and health endpoints')
    parser.add_argument('--url', default='http://localhost:5000', 
                       help='Base URL of the API (default: http://localhost:5000)')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for server to be available before testing')
    
    args = parser.parse_args()
    
    tester = APIDocumentationTester(args.url)
    
    if args.wait:
        print("⏳ Waiting for server to be available...")
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{args.url}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Server is available!")
                    break
            except requests.exceptions.RequestException:
                pass
            
            if attempt == max_attempts - 1:
                print("❌ Server not available after 30 attempts")
                sys.exit(1)
            
            time.sleep(1)
    
    # Run tests
    success = tester.run_all_tests()
    tester.print_test_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()