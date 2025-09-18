#!/usr/bin/env python3
"""
Demo script for API documentation and testing interface.
Showcases the new health check endpoints and interactive API documentation.
"""

import requests
import json
import time
import webbrowser
from typing import Dict, Any


class APIDocumentationDemo:
    """Demonstrate the API documentation and testing interface features."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """Initialize the demo with base URL."""
        self.base_url = base_url
        self.session = requests.Session()
    
    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "=" * 60)
        print(f"🚀 {title}")
        print("=" * 60)
    
    def print_section(self, title: str):
        """Print a formatted section header."""
        print(f"\n📋 {title}")
        print("-" * 40)
    
    def print_response(self, response: requests.Response, title: str = "Response"):
        """Print formatted response information."""
        print(f"\n{title}:")
        print(f"Status: {response.status_code}")
        print(f"Response Time: {response.elapsed.total_seconds():.3f}s")
        
        try:
            data = response.json()
            print("Body:")
            print(json.dumps(data, indent=2))
        except:
            print("Body (text):")
            print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
    
    def demo_health_endpoints(self):
        """Demonstrate all health check endpoints."""
        self.print_header("Health Check Endpoints Demo")
        
        # Basic health check
        self.print_section("1. Basic Health Check (/health)")
        print("This endpoint provides a simple health status check.")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            self.print_response(response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Detailed health check
        self.print_section("2. Detailed Health Check (/health/detailed)")
        print("This endpoint provides comprehensive system status including component health.")
        
        try:
            response = self.session.get(f"{self.base_url}/health/detailed")
            self.print_response(response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Readiness check
        self.print_section("3. Readiness Check (/health/readiness)")
        print("This endpoint checks if the service is ready to serve requests.")
        
        try:
            response = self.session.get(f"{self.base_url}/health/readiness")
            self.print_response(response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Liveness check
        self.print_section("4. Liveness Check (/health/liveness)")
        print("This endpoint checks if the service is alive and responding.")
        
        try:
            response = self.session.get(f"{self.base_url}/health/liveness")
            self.print_response(response)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def demo_api_documentation(self):
        """Demonstrate the API documentation features."""
        self.print_header("API Documentation Demo")
        
        self.print_section("1. Static API Documentation")
        print("Comprehensive API documentation is available at /api-docs")
        print(f"URL: {self.base_url}/api-docs")
        
        try:
            response = self.session.get(f"{self.base_url}/api-docs")
            print(f"\nDocumentation Status: {response.status_code}")
            print(f"Content Length: {len(response.text)} characters")
            print("✅ Documentation includes:")
            print("   - Complete endpoint reference")
            print("   - Request/response examples")
            print("   - Code samples in Python, JavaScript, and cURL")
            print("   - Error handling documentation")
        except Exception as e:
            print(f"❌ Error accessing documentation: {e}")
        
        self.print_section("2. Interactive API Testing Interface")
        print("Interactive testing interface is available at /api-test")
        print(f"URL: {self.base_url}/api-test")
        
        try:
            response = self.session.get(f"{self.base_url}/api-test")
            print(f"\nTesting Interface Status: {response.status_code}")
            print("✅ Interface includes:")
            print("   - Live endpoint testing")
            print("   - Sample data templates")
            print("   - Real-time health monitoring")
            print("   - Batch testing capabilities")
        except Exception as e:
            print(f"❌ Error accessing testing interface: {e}")
    
    def demo_example_requests(self):
        """Demonstrate example API requests."""
        self.print_header("Example API Requests Demo")
        
        # Test prediction endpoint
        self.print_section("1. Prediction Request Example")
        print("Making a sample prediction request...")
        
        sample_data = {
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
                json=sample_data,
                headers={'Content-Type': 'application/json'}
            )
            print(f"Request Body: {json.dumps(sample_data, indent=2)}")
            self.print_response(response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test model status
        self.print_section("2. Model Status Request Example")
        print("Getting model status information...")
        
        try:
            response = self.session.get(f"{self.base_url}/model/status")
            self.print_response(response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test metrics
        self.print_section("3. Metrics Request Example")
        print("Getting monitoring metrics...")
        
        try:
            response = self.session.get(f"{self.base_url}/metrics?hours=1")
            self.print_response(response)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def demo_error_handling(self):
        """Demonstrate error handling examples."""
        self.print_header("Error Handling Demo")
        
        # Test invalid endpoint
        self.print_section("1. Invalid Endpoint (404 Error)")
        print("Testing error handling for non-existent endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/invalid-endpoint")
            self.print_response(response, "404 Error Response")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test invalid prediction request
        self.print_section("2. Invalid Prediction Request (400 Error)")
        print("Testing error handling for invalid prediction data...")
        
        invalid_data = {
            "invalid_field": "invalid_value"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=invalid_data,
                headers={'Content-Type': 'application/json'}
            )
            print(f"Request Body: {json.dumps(invalid_data, indent=2)}")
            self.print_response(response, "400 Error Response")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def demo_performance_monitoring(self):
        """Demonstrate performance monitoring of health endpoints."""
        self.print_header("Performance Monitoring Demo")
        
        self.print_section("Health Endpoint Performance Test")
        print("Testing response times for health endpoints...")
        
        endpoints = [
            ("/health", "Basic Health"),
            ("/health/detailed", "Detailed Health"),
            ("/health/readiness", "Readiness"),
            ("/health/liveness", "Liveness")
        ]
        
        for endpoint, name in endpoints:
            try:
                times = []
                for i in range(3):
                    start_time = time.time()
                    response = self.session.get(f"{self.base_url}{endpoint}")
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                
                avg_time = sum(times) / len(times)
                print(f"{name:15}: {avg_time:6.1f}ms avg (Status: {response.status_code})")
                
            except Exception as e:
                print(f"{name:15}: Error - {e}")
    
    def open_browser_interfaces(self):
        """Open browser interfaces for interactive demo."""
        self.print_header("Interactive Browser Demo")
        
        print("Opening browser interfaces for hands-on exploration...")
        
        interfaces = [
            ("/api-docs", "API Documentation"),
            ("/api-test", "API Testing Interface"),
            ("/", "Main Dashboard")
        ]
        
        for path, name in interfaces:
            url = f"{self.base_url}{path}"
            print(f"\n🌐 Opening {name}: {url}")
            try:
                # Check if interface is accessible
                response = self.session.get(url)
                if response.status_code == 200:
                    print(f"✅ {name} is accessible")
                    # Uncomment the next line to actually open browsers
                    # webbrowser.open(url)
                else:
                    print(f"❌ {name} returned status {response.status_code}")
            except Exception as e:
                print(f"❌ Error accessing {name}: {e}")
        
        print("\n💡 Tip: You can manually open these URLs in your browser to explore the interfaces.")
    
    def run_complete_demo(self):
        """Run the complete API documentation demo."""
        print("🎯 ML Prediction API Documentation & Testing Demo")
        print("This demo showcases the new API documentation and testing features.")
        
        try:
            # Check if server is available
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                print(f"❌ Server not available at {self.base_url}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server at {self.base_url}: {e}")
            return False
        
        # Run all demo sections
        self.demo_health_endpoints()
        self.demo_api_documentation()
        self.demo_example_requests()
        self.demo_error_handling()
        self.demo_performance_monitoring()
        self.open_browser_interfaces()
        
        # Summary
        self.print_header("Demo Summary")
        print("✅ Demonstrated Features:")
        print("   • Health check endpoints (/health, /health/detailed, /health/readiness, /health/liveness)")
        print("   • Interactive API documentation (/api-docs)")
        print("   • Live API testing interface (/api-test)")
        print("   • Example requests and responses")
        print("   • Error handling examples")
        print("   • Performance monitoring")
        
        print("\n🎉 API Documentation & Testing Demo Complete!")
        print(f"\n🔗 Quick Links:")
        print(f"   📚 API Docs: {self.base_url}/api-docs")
        print(f"   🧪 API Test: {self.base_url}/api-test")
        print(f"   📊 Dashboard: {self.base_url}/")
        
        return True


def main():
    """Main function to run the API documentation demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo API documentation and testing interface')
    parser.add_argument('--url', default='http://localhost:5000', 
                       help='Base URL of the API (default: http://localhost:5000)')
    parser.add_argument('--section', choices=['health', 'docs', 'examples', 'errors', 'performance', 'browser'],
                       help='Run only a specific demo section')
    
    args = parser.parse_args()
    
    demo = APIDocumentationDemo(args.url)
    
    if args.section:
        # Run specific section
        section_map = {
            'health': demo.demo_health_endpoints,
            'docs': demo.demo_api_documentation,
            'examples': demo.demo_example_requests,
            'errors': demo.demo_error_handling,
            'performance': demo.demo_performance_monitoring,
            'browser': demo.open_browser_interfaces
        }
        
        if args.section in section_map:
            section_map[args.section]()
        else:
            print(f"❌ Unknown section: {args.section}")
    else:
        # Run complete demo
        demo.run_complete_demo()


if __name__ == "__main__":
    main()