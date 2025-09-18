#!/usr/bin/env python3
"""
Demo scenarios for the ML deployment and monitoring system.
This script creates realistic datasets and demonstrates the complete workflow
from data generation to model deployment to monitoring.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import subprocess
import threading

# Add models directory to path
sys.path.append('models')

from data_processor import DataProcessor, SampleDataGenerator
from model_trainer import ModelTrainer


class DemoDataGenerator:
    """Generate realistic demo datasets for different scenarios."""
    
    @staticmethod
    def generate_customer_churn_data(n_samples: int = 1000) -> pd.DataFrame:
        """Generate realistic customer churn dataset."""
        np.random.seed(42)
        
        # Customer demographics
        age = np.random.normal(40, 15, n_samples).clip(18, 80)
        tenure_months = np.random.exponential(24, n_samples).clip(1, 120)
        monthly_charges = np.random.normal(65, 25, n_samples).clip(20, 150)
        
        # Usage patterns
        data_usage_gb = np.random.lognormal(2, 1, n_samples).clip(0.1, 100)
        call_minutes = np.random.gamma(2, 100, n_samples).clip(0, 2000)
        support_tickets = np.random.poisson(2, n_samples).clip(0, 20)
        
        # Contract features
        contract_type = np.random.choice(['monthly', 'annual', 'two_year'], n_samples, p=[0.5, 0.3, 0.2])
        auto_pay = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        
        # Calculate churn probability based on realistic factors
        churn_prob = (
            0.1 +  # Base churn rate
            0.3 * (monthly_charges > 100) +  # High charges increase churn
            0.2 * (support_tickets > 5) +  # Many tickets increase churn
            0.15 * (tenure_months < 6) +  # New customers more likely to churn
            -0.1 * (contract_type == 'two_year') +  # Long contracts reduce churn
            -0.05 * auto_pay  # Auto-pay reduces churn
        ).clip(0, 1)
        
        churn = np.random.binomial(1, churn_prob, n_samples)
        
        return pd.DataFrame({
            'age': age.round(0).astype(int),
            'tenure_months': tenure_months.round(0).astype(int),
            'monthly_charges': monthly_charges.round(2),
            'data_usage_gb': data_usage_gb.round(2),
            'call_minutes': call_minutes.round(0).astype(int),
            'support_tickets': support_tickets,
            'contract_monthly': (contract_type == 'monthly').astype(int),
            'contract_annual': (contract_type == 'annual').astype(int),
            'contract_two_year': (contract_type == 'two_year').astype(int),
            'auto_pay': auto_pay,
            'churn': churn
        })
    
    @staticmethod
    def generate_fraud_detection_data(n_samples: int = 1000) -> pd.DataFrame:
        """Generate realistic fraud detection dataset."""
        np.random.seed(123)
        
        # Transaction features
        amount = np.random.lognormal(3, 2, n_samples).clip(1, 10000)
        hour_of_day = np.random.randint(0, 24, n_samples)
        day_of_week = np.random.randint(0, 7, n_samples)
        
        # Account features
        account_age_days = np.random.exponential(365, n_samples).clip(1, 3650)
        previous_transactions = np.random.poisson(50, n_samples).clip(0, 500)
        avg_transaction_amount = np.random.lognormal(2.5, 1, n_samples).clip(10, 1000)
        
        # Location and device features
        merchant_category = np.random.randint(1, 20, n_samples)
        is_weekend = (day_of_week >= 5).astype(int)
        is_night = ((hour_of_day < 6) | (hour_of_day > 22)).astype(int)
        
        # Calculate fraud probability
        fraud_prob = (
            0.02 +  # Base fraud rate
            0.15 * (amount > 1000) +  # Large amounts more suspicious
            0.1 * is_night +  # Night transactions more suspicious
            0.05 * (account_age_days < 30) +  # New accounts more risky
            0.08 * (amount > 3 * avg_transaction_amount) +  # Unusual amount
            -0.01 * (previous_transactions > 100)  # Established accounts less risky
        ).clip(0, 1)
        
        fraud = np.random.binomial(1, fraud_prob, n_samples)
        
        return pd.DataFrame({
            'amount': amount.round(2),
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'account_age_days': account_age_days.round(0).astype(int),
            'previous_transactions': previous_transactions,
            'avg_transaction_amount': avg_transaction_amount.round(2),
            'merchant_category': merchant_category,
            'is_weekend': is_weekend,
            'is_night': is_night,
            'amount_vs_avg_ratio': (amount / avg_transaction_amount).round(2),
            'fraud': fraud
        })
    
    @staticmethod
    def generate_sales_prediction_data(n_samples: int = 1000) -> pd.DataFrame:
        """Generate realistic sales prediction dataset."""
        np.random.seed(456)
        
        # Time-based features
        month = np.random.randint(1, 13, n_samples)
        is_holiday_season = ((month == 11) | (month == 12)).astype(int)
        is_summer = ((month >= 6) & (month <= 8)).astype(int)
        
        # Marketing features
        marketing_spend = np.random.exponential(5000, n_samples).clip(100, 50000)
        email_campaigns = np.random.poisson(3, n_samples).clip(0, 15)
        social_media_reach = np.random.lognormal(8, 1, n_samples).clip(1000, 1000000)
        
        # Product features
        product_price = np.random.normal(50, 20, n_samples).clip(10, 200)
        competitor_price = product_price * np.random.normal(1.1, 0.2, n_samples).clip(0.8, 1.5)
        inventory_level = np.random.uniform(0, 1000, n_samples)
        
        # Economic features
        economic_index = np.random.normal(100, 10, n_samples).clip(70, 130)
        
        # Calculate sales based on realistic factors
        base_sales = 100
        sales = (
            base_sales +
            50 * is_holiday_season +  # Holiday boost
            20 * is_summer +  # Summer boost
            marketing_spend * 0.01 +  # Marketing effect
            email_campaigns * 5 +  # Email effect
            np.log(social_media_reach) * 2 +  # Social media effect
            -30 * (product_price > competitor_price) +  # Price competitiveness
            inventory_level * 0.1 +  # Inventory availability
            economic_index * 0.5 +  # Economic conditions
            np.random.normal(0, 20, n_samples)  # Random noise
        ).clip(0, None)
        
        return pd.DataFrame({
            'month': month,
            'is_holiday_season': is_holiday_season,
            'is_summer': is_summer,
            'marketing_spend': marketing_spend.round(2),
            'email_campaigns': email_campaigns,
            'social_media_reach': social_media_reach.round(0).astype(int),
            'product_price': product_price.round(2),
            'competitor_price': competitor_price.round(2),
            'price_ratio': (product_price / competitor_price).round(3),
            'inventory_level': inventory_level.round(0).astype(int),
            'economic_index': economic_index.round(1),
            'sales': sales.round(0).astype(int)
        })


class DataDriftSimulator:
    """Simulate data drift for monitoring demonstrations."""
    
    @staticmethod
    def create_drifted_data(original_data: pd.DataFrame, 
                           target_column: str,
                           drift_type: str = 'gradual',
                           drift_strength: float = 0.3) -> pd.DataFrame:
        """
        Create drifted version of data for monitoring demo.
        
        Args:
            original_data: Original dataset
            target_column: Name of target column
            drift_type: Type of drift ('gradual', 'sudden', 'seasonal')
            drift_strength: Strength of drift (0.0 to 1.0)
        """
        drifted_data = original_data.copy()
        
        # Get numeric columns (excluding target)
        numeric_cols = drifted_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        if drift_type == 'gradual':
            # Gradually shift feature distributions
            for col in feature_cols:
                shift = drifted_data[col].std() * drift_strength
                drifted_data[col] = drifted_data[col] + shift
                
        elif drift_type == 'sudden':
            # Sudden shift in half the data
            n_samples = len(drifted_data)
            shift_start = n_samples // 2
            
            for col in feature_cols:
                shift = drifted_data[col].std() * drift_strength * 2
                drifted_data.loc[shift_start:, col] += shift
                
        elif drift_type == 'seasonal':
            # Cyclical drift pattern
            n_samples = len(drifted_data)
            cycle = np.sin(np.linspace(0, 4*np.pi, n_samples))
            
            for col in feature_cols:
                shift = drifted_data[col].std() * drift_strength
                drifted_data[col] += shift * cycle
        
        return drifted_data
    
    @staticmethod
    def generate_drift_scenarios(base_data: pd.DataFrame, 
                                target_column: str) -> Dict[str, pd.DataFrame]:
        """Generate multiple drift scenarios for testing."""
        scenarios = {}
        
        # No drift (baseline)
        scenarios['no_drift'] = base_data.copy()
        
        # Different drift types
        scenarios['gradual_drift'] = DataDriftSimulator.create_drifted_data(
            base_data, target_column, 'gradual', 0.3
        )
        
        scenarios['sudden_drift'] = DataDriftSimulator.create_drifted_data(
            base_data, target_column, 'sudden', 0.4
        )
        
        scenarios['seasonal_drift'] = DataDriftSimulator.create_drifted_data(
            base_data, target_column, 'seasonal', 0.2
        )
        
        scenarios['strong_drift'] = DataDriftSimulator.create_drifted_data(
            base_data, target_column, 'gradual', 0.6
        )
        
        return scenarios


class HackathonDemoScenarios:
    """Complete demo scenarios for hackathon presentation."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.server_process: Optional[subprocess.Popen] = None
    
    def setup_demo_environment(self) -> bool:
        """Set up the complete demo environment."""
        print("🚀 Setting up hackathon demo environment...")
        
        # Generate demo datasets
        print("📊 Generating demo datasets...")
        self._generate_demo_datasets()
        
        # Train initial models
        print("🤖 Training initial models...")
        self._train_demo_models()
        
        # Start API server
        print("🌐 Starting API server...")
        return self._start_api_server()
    
    def _generate_demo_datasets(self):
        """Generate all demo datasets."""
        # Customer churn dataset
        churn_data = DemoDataGenerator.generate_customer_churn_data(1000)
        churn_data.to_csv('data/demo_customer_churn.csv', index=False)
        print("  ✓ Customer churn dataset (1000 samples)")
        
        # Fraud detection dataset
        fraud_data = DemoDataGenerator.generate_fraud_detection_data(1000)
        fraud_data.to_csv('data/demo_fraud_detection.csv', index=False)
        print("  ✓ Fraud detection dataset (1000 samples)")
        
        # Sales prediction dataset
        sales_data = DemoDataGenerator.generate_sales_prediction_data(1000)
        sales_data.to_csv('data/demo_sales_prediction.csv', index=False)
        print("  ✓ Sales prediction dataset (1000 samples)")
        
        # Generate drift scenarios for each dataset
        for name, data, target in [
            ('churn', churn_data, 'churn'),
            ('fraud', fraud_data, 'fraud'),
            ('sales', sales_data, 'sales')
        ]:
            drift_scenarios = DataDriftSimulator.generate_drift_scenarios(data, target)
            for scenario_name, scenario_data in drift_scenarios.items():
                filename = f'data/demo_{name}_{scenario_name}.csv'
                scenario_data.to_csv(filename, index=False)
                print(f"    ✓ {name} {scenario_name} scenario")
    
    def _train_demo_models(self):
        """Train models for demo datasets."""
        trainer = ModelTrainer()
        
        datasets = [
            ('data/demo_customer_churn.csv', 'churn', 'classification'),
            ('data/demo_fraud_detection.csv', 'fraud', 'classification'),
            ('data/demo_sales_prediction.csv', 'sales', 'regression')
        ]
        
        for data_path, target, problem_type in datasets:
            try:
                result = trainer.train_model(data_path, target_column=target)
                if result.success:
                    print(f"  ✓ Trained {problem_type} model: {result.model_id}")
                    print(f"    Accuracy: {result.metrics.get('accuracy', result.metrics.get('r2_score', 'N/A')):.3f}")
                else:
                    print(f"  ✗ Failed to train model for {data_path}: {result.error}")
            except Exception as e:
                print(f"  ✗ Error training model for {data_path}: {e}")
    
    def _start_api_server(self) -> bool:
        """Start the API server."""
        try:
            self.server_process = subprocess.Popen(
                ["python", "app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to be ready
            for attempt in range(30):
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=2)
                    if response.status_code == 200:
                        print("  ✓ API server is ready!")
                        return True
                except:
                    pass
                time.sleep(1)
            
            print("  ✗ API server failed to start")
            return False
            
        except Exception as e:
            print(f"  ✗ Error starting server: {e}")
            return False
    
    def stop_demo_environment(self):
        """Stop the demo environment."""
        if self.server_process:
            print("🛑 Stopping API server...")
            self.server_process.terminate()
            self.server_process.wait()
            print("  ✓ Server stopped")
    
    def run_scenario_1_basic_workflow(self):
        """Scenario 1: Basic ML workflow demonstration."""
        print("\n" + "="*60)
        print("📋 SCENARIO 1: Basic ML Workflow")
        print("="*60)
        
        print("This scenario demonstrates the complete ML workflow:")
        print("1. Data loading and preprocessing")
        print("2. Model training and evaluation")
        print("3. Model deployment via API")
        print("4. Making predictions")
        
        # Check model status
        response = requests.get(f"{self.base_url}/model/status")
        if response.status_code == 200:
            status = response.json()
            print(f"\n✓ {status['total_models']} models available")
            
            if status['current_model']:
                current = status['current_model']
                print(f"✓ Active model: {current['model_id']}")
                print(f"  Type: {current['model_type']}")
                print(f"  Features: {current['feature_names']}")
                
                # Make sample predictions
                print(f"\n🔮 Making sample predictions...")
                self._make_sample_predictions(current, 5)
            else:
                print("❌ No models available")
        
        print("\n✅ Scenario 1 completed!")
    
    def run_scenario_2_monitoring_demo(self):
        """Scenario 2: Monitoring and drift detection."""
        print("\n" + "="*60)
        print("📊 SCENARIO 2: Monitoring & Drift Detection")
        print("="*60)
        
        print("This scenario demonstrates:")
        print("1. Real-time prediction monitoring")
        print("2. Data drift detection")
        print("3. Alert generation")
        print("4. Performance tracking")
        
        # Get current model
        response = requests.get(f"{self.base_url}/model/status")
        if response.status_code != 200:
            print("❌ Cannot get model status")
            return
        
        current_model = response.json()['current_model']
        if not current_model:
            print("❌ No active model")
            return
        
        print(f"\n📈 Generating baseline predictions...")
        self._make_sample_predictions(current_model, 20, "normal")
        
        print(f"\n⚠️ Generating drifted predictions...")
        self._make_drifted_predictions(current_model, 15)
        
        # Check drift
        print(f"\n🔍 Checking for data drift...")
        model_id = current_model['model_id']
        response = requests.get(f"{self.base_url}/metrics/drift/{model_id}?hours=1")
        if response.status_code == 200:
            drift = response.json()
            print(f"  Drift detected: {drift['has_drift']}")
            print(f"  Drift score: {drift['drift_score']:.3f}")
            print(f"  Threshold: {drift['threshold']}")
        
        # Check alerts
        print(f"\n🚨 Checking alerts...")
        response = requests.get(f"{self.base_url}/alerts")
        if response.status_code == 200:
            alerts = response.json()['alerts']
            print(f"  Active alerts: {len(alerts)}")
            for alert in alerts[:3]:  # Show first 3
                print(f"    - {alert['message']} ({alert['severity']})")
        
        print("\n✅ Scenario 2 completed!")
    
    def run_scenario_3_retraining_demo(self):
        """Scenario 3: Automated retraining demonstration."""
        print("\n" + "="*60)
        print("🔄 SCENARIO 3: Automated Retraining")
        print("="*60)
        
        print("This scenario demonstrates:")
        print("1. Performance degradation detection")
        print("2. Automatic retraining trigger")
        print("3. Model versioning")
        print("4. Seamless model deployment")
        
        # Trigger retraining
        print(f"\n🔄 Triggering model retraining...")
        response = requests.post(f"{self.base_url}/model/retrain")
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Retraining initiated: {result['message']}")
            
            # Wait for retraining to complete
            print("  ⏳ Waiting for retraining to complete...")
            time.sleep(10)  # Simulate retraining time
            
            # Check new model status
            response = requests.get(f"{self.base_url}/model/status")
            if response.status_code == 200:
                status = response.json()
                print(f"  ✓ Models available: {status['total_models']}")
                if status['current_model']:
                    print(f"  ✓ New active model: {status['current_model']['model_id']}")
        else:
            print(f"  ❌ Retraining failed: {response.text}")
        
        print("\n✅ Scenario 3 completed!")
    
    def run_complete_hackathon_demo(self):
        """Run the complete hackathon demonstration."""
        print("🎯 HACKATHON ML DEPLOYMENT & MONITORING DEMO")
        print("="*60)
        
        try:
            # Setup
            if not self.setup_demo_environment():
                print("❌ Failed to setup demo environment")
                return False
            
            print("\n🎬 Starting demo scenarios...")
            
            # Run scenarios
            self.run_scenario_1_basic_workflow()
            time.sleep(2)
            
            self.run_scenario_2_monitoring_demo()
            time.sleep(2)
            
            self.run_scenario_3_retraining_demo()
            
            # Summary
            print("\n" + "="*60)
            print("🎉 HACKATHON DEMO COMPLETED!")
            print("="*60)
            
            print("\n📋 Demo Summary:")
            print("✅ Generated realistic datasets for 3 use cases")
            print("✅ Trained and deployed ML models")
            print("✅ Demonstrated real-time monitoring")
            print("✅ Showed data drift detection")
            print("✅ Simulated automated retraining")
            
            print("\n🔗 Available Endpoints:")
            print(f"  API Health:      GET  {self.base_url}/health")
            print(f"  Model Status:    GET  {self.base_url}/model/status")
            print(f"  Make Prediction: POST {self.base_url}/predict")
            print(f"  Get Metrics:     GET  {self.base_url}/metrics")
            print(f"  Check Drift:     GET  {self.base_url}/metrics/drift/<model_id>")
            print(f"  View Alerts:     GET  {self.base_url}/alerts")
            print(f"  Trigger Retrain: POST {self.base_url}/model/retrain")
            
            print(f"\n🌐 Web Dashboard: {self.base_url}/dashboard")
            print(f"📚 API Documentation: {self.base_url}/docs")
            
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️ Demo interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            return False
        finally:
            self.stop_demo_environment()
    
    def _make_sample_predictions(self, model_info: dict, count: int, data_type: str = "normal"):
        """Make sample predictions for demonstration."""
        feature_names = model_info['feature_names']
        
        for i in range(count):
            # Generate realistic feature values
            features = {}
            for feature in feature_names:
                if data_type == "normal":
                    if 'age' in feature:
                        features[feature] = np.random.randint(25, 65)
                    elif 'amount' in feature or 'charges' in feature or 'spend' in feature:
                        features[feature] = round(np.random.uniform(20, 200), 2)
                    elif 'usage' in feature or 'minutes' in feature:
                        features[feature] = round(np.random.uniform(0, 100), 2)
                    else:
                        features[feature] = round(np.random.uniform(0, 10), 2)
                else:
                    # Generate values from original data ranges
                    features[feature] = round(np.random.uniform(-2, 2), 2)
            
            payload = {"features": features}
            response = requests.post(f"{self.base_url}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"    Prediction {i+1}: {result['prediction']} "
                      f"(time: {result['response_time_ms']}ms)")
            
            time.sleep(0.1)
    
    def _make_drifted_predictions(self, model_info: dict, count: int):
        """Make predictions with drifted data."""
        feature_names = model_info['feature_names']
        
        for i in range(count):
            # Generate drifted feature values (shifted distributions)
            features = {}
            for feature in feature_names:
                if 'age' in feature:
                    features[feature] = np.random.randint(70, 90)  # Older customers
                elif 'amount' in feature or 'charges' in feature:
                    features[feature] = round(np.random.uniform(200, 500), 2)  # Higher amounts
                elif 'usage' in feature or 'minutes' in feature:
                    features[feature] = round(np.random.uniform(100, 300), 2)  # Higher usage
                else:
                    features[feature] = round(np.random.uniform(10, 20), 2)  # Shifted values
            
            payload = {"features": features}
            response = requests.post(f"{self.base_url}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"    Drifted {i+1}: {result['prediction']} "
                      f"(time: {result['response_time_ms']}ms)")
            
            time.sleep(0.1)


def main():
    """Main function to run demo scenarios."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Demo Scenarios')
    parser.add_argument('--scenario', choices=['setup', 'basic', 'monitoring', 'retraining', 'complete'],
                       default='complete', help='Which scenario to run')
    parser.add_argument('--generate-data', action='store_true', 
                       help='Only generate demo datasets')
    
    args = parser.parse_args()
    
    demo = HackathonDemoScenarios()
    
    if args.generate_data:
        print("📊 Generating demo datasets only...")
        demo._generate_demo_datasets()
        print("✅ Demo datasets generated!")
        return
    
    if args.scenario == 'setup':
        success = demo.setup_demo_environment()
        if success:
            print("✅ Demo environment setup complete!")
            input("Press Enter to stop the server...")
        demo.stop_demo_environment()
    
    elif args.scenario == 'complete':
        demo.run_complete_hackathon_demo()
    
    else:
        # Setup first
        if not demo.setup_demo_environment():
            print("❌ Failed to setup demo environment")
            return
        
        try:
            if args.scenario == 'basic':
                demo.run_scenario_1_basic_workflow()
            elif args.scenario == 'monitoring':
                demo.run_scenario_2_monitoring_demo()
            elif args.scenario == 'retraining':
                demo.run_scenario_3_retraining_demo()
        finally:
            demo.stop_demo_environment()


if __name__ == '__main__':
    main()