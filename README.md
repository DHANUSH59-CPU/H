# Automated ML Deployment & Monitoring System

A hackathon-friendly ML system that demonstrates automated model training, deployment, and monitoring.

## Project Structure

```
├── models/          # Trained ML models storage
├── data/           # Training and test datasets
├── templates/      # HTML templates for web dashboard
├── static/         # CSS, JavaScript, and static assets
├── config.py       # Configuration file with parameters and thresholds
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. The system is configured through `config.py` where you can adjust:
   - Model parameters and algorithms
   - Monitoring thresholds
   - API endpoints and settings
   - Dashboard configuration

## Configuration

Key configuration sections in `config.py`:
- `MODEL_CONFIG`: ML model parameters
- `MONITORING_THRESHOLDS`: Performance and drift thresholds
- `API_CONFIG`: Flask server settings
- `DASHBOARD_CONFIG`: Web interface settings

## Requirements Addressed

This setup addresses requirements:
- **5.3**: API documentation and standard HTTP/JSON format support
- **5.4**: Clear error messages and 2-second response time configuration