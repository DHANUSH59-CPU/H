#!/usr/bin/env python3
"""
Demo script for the web dashboard interface.
Shows how to use the dashboard and its features.
"""

import webbrowser
import time
import requests
import json
import threading
from datetime import datetime

def start_demo():
    """Start the dashboard demo."""
    print("🚀 Starting ML Monitoring Dashboard Demo")
    print("=" * 50)
    
    # Check if Flask app is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        if response.status_code == 200:
            print("✅ Flask app is running")
        else:
            print("❌ Flask app returned error:", response.status_code)
            return
    except requests.exceptions.RequestException:
        print("❌ Flask app is not running!")
        print("Please start it with: python app.py")
        return
    
    print("\n📊 Dashboard Features:")
    print("- Real-time monitoring metrics")
    print("- Interactive charts with Chart.js")
    print("- Model management interface")
    print("- API testing tools")
    print("- Live data updates via AJAX")
    
    # Open dashboard in browser
    dashboard_url = "http://localhost:5000"
    print(f"\n🌐 Opening dashboard at: {dashboard_url}")
    
    try:
        webbrowser.open(dashboard_url)
        print("✅ Dashboard opened in browser")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")
        print(f"Please manually open: {dashboard_url}")
    
    # Generate some demo data
    print("\n🎯 Generating demo prediction data...")
    generate_demo_data()
    
    print("\n📱 Dashboard Pages Available:")
    print(f"- Main Dashboard: {dashboard_url}/")
    print(f"- Model Management: {dashboard_url}/models")
    print(f"- API Testing: {dashboard_url}/api-test")
    
    print("\n🔄 The dashboard will auto-refresh every 30 seconds")
    print("💡 Try making predictions to see real-time updates!")
    
    # Show API endpoints
    print("\n🔌 Available API Endpoints:")
    endpoints = [
        ("GET /", "Main dashboard page"),
        ("GET /models", "Model management page"),
        ("GET /api-test", "API testing interface"),
        ("POST /predict", "Make predictions"),
        ("GET /metrics", "Get monitoring metrics"),
        ("GET /model/status", "Get model information"),
        ("GET /health", "Health check")
    ]
    
    for endpoint, description in endpoints:
        print(f"  {endpoint:<20} - {description}")

def generate_demo_data():
    """Generate some demo prediction data."""
    base_url = "http://localhost:5000"
    
    # Sample feature sets for different scenarios
    sample_features = [
        {"feature_1": 1.5, "feature_2": 2.3, "feature_3": 0.8, "feature_4": 1.2},
        {"feature_1": 2.1, "feature_2": 1.8, "feature_3": 1.5, "feature_4": 0.9},
        {"feature_1": 0.8, "feature_2": 3.2, "feature_3": 2.1, "feature_4": 1.7},
        {"feature_1": 1.9, "feature_2": 2.7, "feature_3": 1.3, "feature_4": 1.1},
        {"feature_1": 1.2, "feature_2": 2.0, "feature_3": 1.8, "feature_4": 1.4}
    ]
    
    successful_requests = 0
    
    for i, features in enumerate(sample_features):
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"features": features},
                timeout=5
            )
            
            if response.status_code == 200:
                successful_requests += 1
                data = response.json()
                print(f"  ✅ Prediction {i+1}: {data.get('prediction')} ({data.get('response_time_ms')}ms)")
            elif response.status_code == 503:
                print(f"  ⚠️  No models available - train a model first")
                break
            else:
                print(f"  ❌ Request {i+1} failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Request {i+1} error: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    if successful_requests > 0:
        print(f"✅ Generated {successful_requests} demo predictions")
        print("🔄 Refresh the dashboard to see the new data")
    else:
        print("💡 Train a model first to see prediction data in the dashboard")

def show_dashboard_usage():
    """Show how to use the dashboard features."""
    print("\n📖 Dashboard Usage Guide:")
    print("=" * 30)
    
    print("\n🏠 Main Dashboard:")
    print("- View real-time metrics in status cards")
    print("- Monitor prediction volume with interactive charts")
    print("- Check model performance and response times")
    print("- View recent predictions and active alerts")
    print("- Data refreshes automatically every 30 seconds")
    
    print("\n🤖 Models Page:")
    print("- View all available trained models")
    print("- See current active model information")
    print("- Trigger manual model training")
    print("- Configure automated retraining settings")
    print("- View detailed model information")
    
    print("\n🧪 API Test Page:")
    print("- Test prediction API with custom inputs")
    print("- Load sample data for different model types")
    print("- Run batch tests to check performance")
    print("- View API documentation and examples")
    
    print("\n💡 Tips:")
    print("- Use the period buttons (1H, 6H, 24H) to change chart timeframes")
    print("- Click 'Check Now' to manually trigger drift detection")
    print("- Use the API test page to generate data for the dashboard")
    print("- Alerts can be resolved by clicking the 'Resolve' button")

def monitor_dashboard_activity():
    """Monitor dashboard activity in the background."""
    print("\n🔍 Monitoring dashboard activity...")
    
    base_url = "http://localhost:5000"
    
    while True:
        try:
            # Check metrics
            response = requests.get(f"{base_url}/metrics", timeout=5)
            if response.status_code == 200:
                data = response.json()
                predictions_today = data.get('predictions_today', 0)
                active_alerts = data.get('active_alerts', 0)
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Predictions: {predictions_today}, Alerts: {active_alerts}")
            
            time.sleep(30)  # Check every 30 seconds
            
        except requests.exceptions.RequestException:
            print("Dashboard monitoring stopped (app not running)")
            break
        except KeyboardInterrupt:
            print("\nDashboard monitoring stopped")
            break

if __name__ == "__main__":
    import sys
    
    if "--usage" in sys.argv:
        show_dashboard_usage()
    elif "--monitor" in sys.argv:
        monitor_dashboard_activity()
    else:
        start_demo()
        
        if "--interactive" in sys.argv:
            print("\n" + "=" * 50)
            show_dashboard_usage()
            
            print("\n🔄 Starting background monitoring...")
            print("Press Ctrl+C to stop")
            
            try:
                monitor_dashboard_activity()
            except KeyboardInterrupt:
                print("\n👋 Demo finished!")