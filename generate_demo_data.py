#!/usr/bin/env python3
"""
Generate comprehensive demo datasets for different ML use cases.
This script creates realistic datasets that can be used for hackathon demonstrations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_ecommerce_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate e-commerce customer behavior dataset."""
    np.random.seed(789)
    
    # Customer features
    customer_age = np.random.normal(35, 12, n_samples).clip(18, 70)
    days_since_signup = np.random.exponential(200, n_samples).clip(1, 1000)
    
    # Behavioral features
    page_views = np.random.poisson(15, n_samples).clip(1, 100)
    session_duration = np.random.lognormal(3, 1, n_samples).clip(60, 7200)  # seconds
    cart_additions = np.random.poisson(3, n_samples).clip(0, 20)
    
    # Purchase history
    previous_purchases = np.random.poisson(5, n_samples).clip(0, 50)
    avg_order_value = np.random.lognormal(4, 0.8, n_samples).clip(20, 500)
    
    # Session features
    is_mobile = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    is_weekend = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    hour_of_day = np.random.randint(0, 24, n_samples)
    
    # Marketing features
    email_subscriber = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    discount_used = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    
    # Calculate purchase probability
    purchase_prob = (
        0.1 +  # Base conversion rate
        0.2 * (cart_additions > 0) +  # Items in cart
        0.15 * (previous_purchases > 10) +  # Loyal customers
        0.1 * email_subscriber +  # Email subscribers
        0.05 * discount_used +  # Discount effect
        0.08 * (session_duration > 300) +  # Engaged users
        -0.05 * is_mobile +  # Mobile penalty
        0.03 * (hour_of_day >= 10) * (hour_of_day <= 20)  # Peak hours
    ).clip(0, 1)
    
    purchase = np.random.binomial(1, purchase_prob, n_samples)
    
    return pd.DataFrame({
        'customer_age': customer_age.round(0).astype(int),
        'days_since_signup': days_since_signup.round(0).astype(int),
        'page_views': page_views,
        'session_duration_minutes': (session_duration / 60).round(1),
        'cart_additions': cart_additions,
        'previous_purchases': previous_purchases,
        'avg_order_value': avg_order_value.round(2),
        'is_mobile': is_mobile,
        'is_weekend': is_weekend,
        'hour_of_day': hour_of_day,
        'email_subscriber': email_subscriber,
        'discount_used': discount_used,
        'purchase': purchase
    })


def generate_predictive_maintenance_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate predictive maintenance dataset for industrial equipment."""
    np.random.seed(321)
    
    # Equipment features
    equipment_age_months = np.random.exponential(36, n_samples).clip(1, 120)
    operating_hours = np.random.uniform(0, 24, n_samples)
    load_factor = np.random.beta(2, 2, n_samples)  # 0 to 1
    
    # Sensor readings
    temperature = np.random.normal(75, 15, n_samples).clip(40, 120)
    vibration = np.random.lognormal(2, 0.5, n_samples).clip(0.1, 10)
    pressure = np.random.normal(100, 20, n_samples).clip(50, 200)
    
    # Maintenance history
    days_since_maintenance = np.random.exponential(30, n_samples).clip(0, 180)
    maintenance_count = np.random.poisson(equipment_age_months / 12, n_samples).clip(0, 20)
    
    # Environmental factors
    ambient_temp = np.random.normal(20, 10, n_samples).clip(-10, 40)
    humidity = np.random.uniform(30, 90, n_samples)
    
    # Calculate failure probability
    failure_prob = (
        0.02 +  # Base failure rate
        0.3 * (equipment_age_months > 60) +  # Old equipment
        0.2 * (temperature > 100) +  # Overheating
        0.15 * (vibration > 5) +  # High vibration
        0.1 * (days_since_maintenance > 90) +  # Overdue maintenance
        0.05 * (load_factor > 0.8) +  # High load
        0.08 * (pressure > 150)  # High pressure
    ).clip(0, 1)
    
    failure = np.random.binomial(1, failure_prob, n_samples)
    
    return pd.DataFrame({
        'equipment_age_months': equipment_age_months.round(0).astype(int),
        'operating_hours': operating_hours.round(1),
        'load_factor': load_factor.round(3),
        'temperature': temperature.round(1),
        'vibration': vibration.round(2),
        'pressure': pressure.round(1),
        'days_since_maintenance': days_since_maintenance.round(0).astype(int),
        'maintenance_count': maintenance_count,
        'ambient_temp': ambient_temp.round(1),
        'humidity': humidity.round(1),
        'failure': failure
    })


def generate_credit_scoring_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate credit scoring dataset."""
    np.random.seed(654)
    
    # Personal information
    age = np.random.normal(40, 15, n_samples).clip(18, 80)
    income = np.random.lognormal(10.5, 0.8, n_samples).clip(20000, 200000)
    employment_years = np.random.exponential(8, n_samples).clip(0, 40)
    
    # Credit history
    credit_history_years = np.random.exponential(10, n_samples).clip(0, 50)
    existing_loans = np.random.poisson(2, n_samples).clip(0, 10)
    credit_utilization = np.random.beta(2, 3, n_samples)  # Skewed towards lower values
    
    # Financial ratios
    debt_to_income = np.random.beta(2, 5, n_samples) * 0.8  # Max 80%
    savings_ratio = np.random.beta(1.5, 3, n_samples) * 0.3  # Max 30%
    
    # Loan details
    loan_amount = np.random.lognormal(9.5, 1, n_samples).clip(5000, 100000)
    loan_term_months = np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    
    # Other factors
    homeowner = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    education_level = np.random.randint(1, 5, n_samples)  # 1-4 scale
    
    # Calculate default probability (inverse of credit score)
    default_prob = (
        0.05 +  # Base default rate
        0.2 * (debt_to_income > 0.5) +  # High debt ratio
        0.15 * (credit_utilization > 0.8) +  # High credit utilization
        0.1 * (employment_years < 2) +  # Job instability
        0.08 * (existing_loans > 5) +  # Too many loans
        -0.05 * homeowner +  # Homeowners less risky
        -0.03 * (education_level - 1) +  # Education effect
        -0.1 * (savings_ratio > 0.1)  # Savings buffer
    ).clip(0, 1)
    
    default = np.random.binomial(1, default_prob, n_samples)
    
    return pd.DataFrame({
        'age': age.round(0).astype(int),
        'annual_income': income.round(0).astype(int),
        'employment_years': employment_years.round(1),
        'credit_history_years': credit_history_years.round(1),
        'existing_loans': existing_loans,
        'credit_utilization': credit_utilization.round(3),
        'debt_to_income_ratio': debt_to_income.round(3),
        'savings_ratio': savings_ratio.round(3),
        'loan_amount': loan_amount.round(0).astype(int),
        'loan_term_months': loan_term_months,
        'homeowner': homeowner,
        'education_level': education_level,
        'default': default
    })


def generate_time_series_data(n_samples: int = 365) -> pd.DataFrame:
    """Generate time series data for demand forecasting."""
    np.random.seed(987)
    
    # Create date range
    start_date = datetime.now() - timedelta(days=n_samples)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Time-based features
    day_of_week = [d.weekday() for d in dates]
    month = [d.month for d in dates]
    is_weekend = [1 if d.weekday() >= 5 else 0 for d in dates]
    is_holiday = [1 if d.month == 12 and d.day >= 20 else 0 for d in dates]  # Simplified holidays
    
    # Seasonal patterns
    seasonal_trend = np.sin(2 * np.pi * np.arange(n_samples) / 365.25) * 20
    weekly_pattern = np.sin(2 * np.pi * np.arange(n_samples) / 7) * 10
    
    # External factors
    temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25) + np.random.normal(0, 5, n_samples)
    marketing_spend = np.random.exponential(1000, n_samples).clip(100, 10000)
    competitor_price = np.random.normal(50, 5, n_samples).clip(30, 80)
    
    # Base demand with trends and patterns
    base_demand = 100
    trend = 0.1 * np.arange(n_samples)  # Slight upward trend
    demand = (
        base_demand + 
        trend +
        seasonal_trend +
        weekly_pattern +
        30 * np.array(is_holiday) +  # Holiday boost
        -10 * np.array(is_weekend) +  # Weekend drop
        0.5 * (temperature - 20) +  # Temperature effect
        marketing_spend * 0.01 +  # Marketing effect
        -0.5 * (competitor_price - 50) +  # Competition effect
        np.random.normal(0, 15, n_samples)  # Random noise
    ).clip(0, None)
    
    return pd.DataFrame({
        'date': dates,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'temperature': temperature.round(1),
        'marketing_spend': marketing_spend.round(2),
        'competitor_price': competitor_price.round(2),
        'demand': demand.round(0).astype(int)
    })


def main():
    """Generate all demo datasets."""
    print("🎯 Generating comprehensive demo datasets...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate datasets
    datasets = [
        ('demo_ecommerce', generate_ecommerce_data, 1000, 'E-commerce conversion prediction'),
        ('demo_maintenance', generate_predictive_maintenance_data, 1000, 'Predictive maintenance'),
        ('demo_credit_scoring', generate_credit_scoring_data, 1000, 'Credit scoring'),
        ('demo_demand_forecast', generate_time_series_data, 365, 'Demand forecasting')
    ]
    
    for name, generator, n_samples, description in datasets:
        print(f"\n📊 Generating {description} dataset...")
        
        try:
            data = generator(n_samples)
            filename = f'data/{name}.csv'
            data.to_csv(filename, index=False)
            
            print(f"  ✓ Saved {filename}")
            print(f"  ✓ Shape: {data.shape}")
            print(f"  ✓ Columns: {list(data.columns)}")
            
            # Show basic statistics
            if data.select_dtypes(include=[np.number]).shape[1] > 0:
                target_col = data.columns[-1]  # Assume last column is target
                if target_col in data.select_dtypes(include=[np.number]).columns:
                    if data[target_col].nunique() <= 10:  # Classification
                        print(f"  ✓ Target distribution: {dict(data[target_col].value_counts())}")
                    else:  # Regression
                        print(f"  ✓ Target range: {data[target_col].min():.1f} - {data[target_col].max():.1f}")
            
        except Exception as e:
            print(f"  ❌ Error generating {name}: {e}")
    
    print(f"\n🎉 Demo dataset generation completed!")
    print(f"\nGenerated datasets:")
    for name, _, _, description in datasets:
        print(f"  - data/{name}.csv: {description}")
    
    print(f"\n💡 Usage:")
    print(f"  - Use these datasets with train_model.py to create demo models")
    print(f"  - Run demo_scenarios.py for complete hackathon demonstrations")
    print(f"  - Each dataset includes realistic features and target variables")


if __name__ == '__main__':
    main()