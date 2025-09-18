"""
Automated retraining system for ML model deployment and monitoring.
Handles background monitoring, automatic retraining triggers, model versioning, and deployment.
"""

import os
import json
import time
import threading
import logging
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np

from config import MONITORING_THRESHOLDS, MODEL_CONFIG, DATABASE_CONFIG
from .model_trainer import ModelTrainer, TrainingScript
from .monitoring import MetricsCollector, PerformanceMetrics, Alert
from .data_processor import DataProcessor


@dataclass
class RetrainingTrigger:
    """Represents a trigger condition for model retraining."""
    trigger_id: str
    trigger_type: str  # 'accuracy_drop', 'drift_detected', 'scheduled', 'manual'
    model_id: str
    threshold_value: float
    current_value: float
    triggered_at: datetime
    data: Dict[str, Any]


@dataclass
class ModelVersion:
    """Represents a version of a model with metadata."""
    version_id: str
    model_id: str
    parent_version: Optional[str]
    version_number: int
    created_at: datetime
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    model_file_path: str
    metadata_file_path: str
    is_active: bool
    deployment_status: str  # 'pending', 'deployed', 'retired'
    notes: str


class ModelVersionManager:
    """
    Manages model versions and tracks model evolution.
    """
    
    def __init__(self, model_dir: str = "models/", versions_file: str = "models/model_versions.json"):
        """
        Initialize model version manager.
        
        Args:
            model_dir: Directory containing model files
            versions_file: JSON file to store version metadata
        """
        self.model_dir = model_dir
        self.versions_file = versions_file
        self.versions = {}  # model_id -> list of versions
        self.active_versions = {}  # model_id -> active version_id
        self.lock = threading.Lock()
        
        os.makedirs(model_dir, exist_ok=True)
        self._load_versions()
    
    def _load_versions(self):
        """Load version metadata from file."""
        if os.path.exists(self.versions_file):
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert loaded data back to ModelVersion objects
                for model_id, versions_data in data.get('versions', {}).items():
                    self.versions[model_id] = []
                    for version_data in versions_data:
                        version = ModelVersion(
                            version_id=version_data['version_id'],
                            model_id=version_data['model_id'],
                            parent_version=version_data.get('parent_version'),
                            version_number=version_data['version_number'],
                            created_at=datetime.fromisoformat(version_data['created_at']),
                            training_metrics=version_data['training_metrics'],
                            validation_metrics=version_data['validation_metrics'],
                            model_file_path=version_data['model_file_path'],
                            metadata_file_path=version_data['metadata_file_path'],
                            is_active=version_data['is_active'],
                            deployment_status=version_data['deployment_status'],
                            notes=version_data.get('notes', '')
                        )
                        self.versions[model_id].append(version)
                
                self.active_versions = data.get('active_versions', {})
                
            except Exception as e:
                logging.error(f"Error loading model versions: {e}")
                self.versions = {}
                self.active_versions = {}
    
    def _save_versions(self):
        """Save version metadata to file."""
        try:
            # Convert ModelVersion objects to serializable format
            data = {
                'versions': {},
                'active_versions': self.active_versions,
                'last_updated': datetime.now().isoformat()
            }
            
            for model_id, versions_list in self.versions.items():
                data['versions'][model_id] = []
                for version in versions_list:
                    version_data = {
                        'version_id': version.version_id,
                        'model_id': version.model_id,
                        'parent_version': version.parent_version,
                        'version_number': version.version_number,
                        'created_at': version.created_at.isoformat(),
                        'training_metrics': version.training_metrics,
                        'validation_metrics': version.validation_metrics,
                        'model_file_path': version.model_file_path,
                        'metadata_file_path': version.metadata_file_path,
                        'is_active': version.is_active,
                        'deployment_status': version.deployment_status,
                        'notes': version.notes
                    }
                    data['versions'][model_id].append(version_data)
            
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving model versions: {e}")
    
    def create_version(self, model_id: str, model_file_path: str, metadata_file_path: str,
                      training_metrics: Dict[str, float], validation_metrics: Dict[str, float],
                      parent_version: str = None, notes: str = "") -> ModelVersion:
        """
        Create a new model version.
        
        Args:
            model_id: Base model identifier
            model_file_path: Path to model file
            metadata_file_path: Path to metadata file
            training_metrics: Training performance metrics
            validation_metrics: Validation performance metrics
            parent_version: ID of parent version (for retraining)
            notes: Optional notes about this version
            
        Returns:
            ModelVersion object
        """
        with self.lock:
            # Determine version number
            if model_id not in self.versions:
                self.versions[model_id] = []
                version_number = 1
            else:
                version_number = max(v.version_number for v in self.versions[model_id]) + 1
            
            # Generate version ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_id = f"{model_id}_v{version_number}_{timestamp}"
            
            # Create version object
            version = ModelVersion(
                version_id=version_id,
                model_id=model_id,
                parent_version=parent_version,
                version_number=version_number,
                created_at=datetime.now(),
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                model_file_path=model_file_path,
                metadata_file_path=metadata_file_path,
                is_active=False,  # Will be activated separately
                deployment_status='pending',
                notes=notes
            )
            
            # Add to versions list
            self.versions[model_id].append(version)
            
            # Save to file
            self._save_versions()
            
            logging.info(f"Created model version {version_id} for model {model_id}")
            return version
    
    def activate_version(self, version_id: str) -> bool:
        """
        Activate a specific model version (deploy it).
        
        Args:
            version_id: Version ID to activate
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Find the version
            target_version = None
            for model_id, versions_list in self.versions.items():
                for version in versions_list:
                    if version.version_id == version_id:
                        target_version = version
                        break
                if target_version:
                    break
            
            if not target_version:
                logging.error(f"Version {version_id} not found")
                return False
            
            # Deactivate current active version for this model
            current_active = self.active_versions.get(target_version.model_id)
            if current_active:
                for version in self.versions[target_version.model_id]:
                    if version.version_id == current_active:
                        version.is_active = False
                        version.deployment_status = 'retired'
                        break
            
            # Activate new version
            target_version.is_active = True
            target_version.deployment_status = 'deployed'
            self.active_versions[target_version.model_id] = version_id
            
            # Copy model files to standard locations for API to use
            try:
                standard_model_path = os.path.join(self.model_dir, f"{target_version.model_id}.joblib")
                standard_metadata_path = os.path.join(self.model_dir, f"{target_version.model_id}_metadata.json")
                
                shutil.copy2(target_version.model_file_path, standard_model_path)
                shutil.copy2(target_version.metadata_file_path, standard_metadata_path)
                
                logging.info(f"Deployed version {version_id} as active model {target_version.model_id}")
                
            except Exception as e:
                logging.error(f"Error deploying version {version_id}: {e}")
                target_version.deployment_status = 'pending'
                return False
            
            # Save changes
            self._save_versions()
            return True
    
    def get_active_version(self, model_id: str) -> Optional[ModelVersion]:
        """
        Get the currently active version for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelVersion object if found, None otherwise
        """
        active_version_id = self.active_versions.get(model_id)
        if not active_version_id:
            return None
        
        for version in self.versions.get(model_id, []):
            if version.version_id == active_version_id:
                return version
        
        return None
    
    def get_version_history(self, model_id: str) -> List[ModelVersion]:
        """
        Get version history for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of ModelVersion objects sorted by version number
        """
        versions = self.versions.get(model_id, [])
        return sorted(versions, key=lambda v: v.version_number, reverse=True)
    
    def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            version_id_1: First version ID
            version_id_2: Second version ID
            
        Returns:
            Dictionary containing comparison results
        """
        # Find versions
        version_1 = None
        version_2 = None
        
        for versions_list in self.versions.values():
            for version in versions_list:
                if version.version_id == version_id_1:
                    version_1 = version
                elif version.version_id == version_id_2:
                    version_2 = version
        
        if not version_1 or not version_2:
            return {'error': 'One or both versions not found'}
        
        # Compare metrics
        comparison = {
            'version_1': {
                'version_id': version_1.version_id,
                'version_number': version_1.version_number,
                'created_at': version_1.created_at.isoformat(),
                'validation_metrics': version_1.validation_metrics
            },
            'version_2': {
                'version_id': version_2.version_id,
                'version_number': version_2.version_number,
                'created_at': version_2.created_at.isoformat(),
                'validation_metrics': version_2.validation_metrics
            },
            'improvements': {}
        }
        
        # Calculate improvements
        for metric in version_1.validation_metrics:
            if metric in version_2.validation_metrics:
                val_1 = version_1.validation_metrics[metric]
                val_2 = version_2.validation_metrics[metric]
                
                if val_1 and val_2:
                    improvement = val_2 - val_1
                    improvement_pct = (improvement / val_1) * 100 if val_1 != 0 else 0
                    
                    comparison['improvements'][metric] = {
                        'absolute': improvement,
                        'percentage': improvement_pct,
                        'better': improvement > 0
                    }
        
        return comparison


class AutomatedRetrainingSystem:
    """
    Main automated retraining system that monitors performance and triggers retraining.
    """
    
    def __init__(self, metrics_collector: MetricsCollector = None, 
                 model_dir: str = "models/", config: Dict = None):
        """
        Initialize automated retraining system.
        
        Args:
            metrics_collector: MetricsCollector instance for monitoring
            model_dir: Directory containing models
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model_dir = model_dir
        self.metrics_collector = metrics_collector
        self.version_manager = ModelVersionManager(model_dir)
        self.trainer = ModelTrainer(config)
        self.training_script = TrainingScript(config)
        
        # Retraining state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.retraining_in_progress = set()  # Track models currently being retrained
        self.trigger_history = deque(maxlen=1000)  # Keep history of triggers
        
        # Configuration
        self.check_interval = self.config.get('retraining_check_interval', 300)  # 5 minutes
        self.accuracy_drop_threshold = MONITORING_THRESHOLDS.get('retraining_accuracy_drop', 0.05)
        self.min_predictions_for_retraining = self.config.get('min_predictions_for_retraining', 100)
        self.retraining_data_window_hours = self.config.get('retraining_data_window_hours', 24)
        
        # Setup logging
        self.logger = logging.getLogger('retraining_system')
        self.logger.setLevel(logging.INFO)
    
    def start_monitoring(self):
        """Start background monitoring for retraining triggers."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Automated retraining monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Automated retraining monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        while self.monitoring_active:
            try:
                self._check_retraining_triggers()
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _check_retraining_triggers(self):
        """Check all models for retraining triggers."""
        if not self.metrics_collector:
            return
        
        # Get all models with recent activity
        recent_metrics = self.metrics_collector.storage.get_recent_metrics(hours=self.retraining_data_window_hours)
        
        # Group metrics by model
        model_metrics = defaultdict(list)
        for metric in recent_metrics:
            model_metrics[metric['model_id']].append(metric)
        
        # Check each model for triggers
        for model_id, metrics_list in model_metrics.items():
            if model_id in self.retraining_in_progress:
                continue  # Skip models already being retrained
            
            self._check_model_triggers(model_id, metrics_list)
    
    def _check_model_triggers(self, model_id: str, metrics_list: List[Dict]):
        """
        Check a specific model for retraining triggers.
        
        Args:
            model_id: Model identifier
            metrics_list: List of recent metrics for the model
        """
        if not metrics_list:
            return
        
        # Get current active version
        active_version = self.version_manager.get_active_version(model_id)
        if not active_version:
            self.logger.warning(f"No active version found for model {model_id}")
            return
        
        # Check accuracy drop trigger
        self._check_accuracy_drop_trigger(model_id, metrics_list, active_version)
        
        # Check drift trigger
        self._check_drift_trigger(model_id)
        
        # Check scheduled retraining (if configured)
        self._check_scheduled_trigger(model_id, active_version)
    
    def _check_accuracy_drop_trigger(self, model_id: str, metrics_list: List[Dict], 
                                   active_version: ModelVersion):
        """Check for accuracy drop trigger."""
        # Get baseline accuracy from active version
        baseline_accuracy = active_version.validation_metrics.get('accuracy')
        if not baseline_accuracy:
            return
        
        # Calculate recent average accuracy
        recent_accuracies = [m['accuracy'] for m in metrics_list if m.get('accuracy') is not None]
        if len(recent_accuracies) < self.min_predictions_for_retraining:
            return
        
        current_accuracy = np.mean(recent_accuracies)
        accuracy_drop = baseline_accuracy - current_accuracy
        
        if accuracy_drop > self.accuracy_drop_threshold:
            trigger = RetrainingTrigger(
                trigger_id=f"accuracy_drop_{model_id}_{int(datetime.now().timestamp())}",
                trigger_type='accuracy_drop',
                model_id=model_id,
                threshold_value=self.accuracy_drop_threshold,
                current_value=accuracy_drop,
                triggered_at=datetime.now(),
                data={
                    'baseline_accuracy': baseline_accuracy,
                    'current_accuracy': current_accuracy,
                    'accuracy_drop': accuracy_drop,
                    'sample_size': len(recent_accuracies)
                }
            )
            
            self._handle_retraining_trigger(trigger)
    
    def _check_drift_trigger(self, model_id: str):
        """Check for data drift trigger."""
        if not self.metrics_collector:
            return
        
        # Check recent drift detection results
        drift_result = self.metrics_collector.check_drift(model_id, hours=1)
        
        if drift_result and drift_result.has_drift:
            trigger = RetrainingTrigger(
                trigger_id=f"drift_{model_id}_{int(datetime.now().timestamp())}",
                trigger_type='drift_detected',
                model_id=model_id,
                threshold_value=drift_result.threshold,
                current_value=drift_result.drift_score,
                triggered_at=datetime.now(),
                data={
                    'drift_score': drift_result.drift_score,
                    'feature_drifts': drift_result.feature_drifts,
                    'method': drift_result.method
                }
            )
            
            self._handle_retraining_trigger(trigger)
    
    def _check_scheduled_trigger(self, model_id: str, active_version: ModelVersion):
        """Check for scheduled retraining trigger."""
        # Check if model is old enough for scheduled retraining
        scheduled_interval_hours = self.config.get('scheduled_retraining_hours')
        if not scheduled_interval_hours:
            return
        
        model_age = datetime.now() - active_version.created_at
        if model_age.total_seconds() / 3600 > scheduled_interval_hours:
            trigger = RetrainingTrigger(
                trigger_id=f"scheduled_{model_id}_{int(datetime.now().timestamp())}",
                trigger_type='scheduled',
                model_id=model_id,
                threshold_value=scheduled_interval_hours,
                current_value=model_age.total_seconds() / 3600,
                triggered_at=datetime.now(),
                data={
                    'model_age_hours': model_age.total_seconds() / 3600,
                    'scheduled_interval': scheduled_interval_hours
                }
            )
            
            self._handle_retraining_trigger(trigger)
    
    def _handle_retraining_trigger(self, trigger: RetrainingTrigger):
        """
        Handle a retraining trigger by starting retraining process.
        
        Args:
            trigger: RetrainingTrigger object
        """
        # Add to trigger history
        self.trigger_history.append(trigger)
        
        # Log the trigger
        self.logger.info(f"Retraining trigger activated: {trigger.trigger_type} for model {trigger.model_id}")
        
        # Create alert
        if self.metrics_collector:
            alert = Alert(
                alert_id=f"retraining_{trigger.trigger_id}",
                alert_type='retraining_triggered',
                severity='medium',
                message=f"Automatic retraining triggered for model {trigger.model_id} due to {trigger.trigger_type}",
                model_id=trigger.model_id,
                timestamp=trigger.triggered_at,
                data=trigger.data
            )
            self.metrics_collector.alert_manager.send_alert(alert)
        
        # Start retraining in background thread
        retraining_thread = threading.Thread(
            target=self._execute_retraining,
            args=(trigger,),
            daemon=True
        )
        retraining_thread.start()
    
    def _execute_retraining(self, trigger: RetrainingTrigger):
        """
        Execute the actual retraining process.
        
        Args:
            trigger: RetrainingTrigger that initiated this retraining
        """
        model_id = trigger.model_id
        
        try:
            # Mark model as being retrained
            self.retraining_in_progress.add(model_id)
            
            self.logger.info(f"Starting retraining for model {model_id}")
            
            # Get current active version for reference
            active_version = self.version_manager.get_active_version(model_id)
            if not active_version:
                raise ValueError(f"No active version found for model {model_id}")
            
            # Prepare training data (use recent prediction data)
            training_data_path = self._prepare_retraining_data(model_id)
            if not training_data_path:
                raise ValueError("Failed to prepare training data")
            
            # Load original model metadata to get training parameters
            with open(active_version.metadata_file_path, 'r') as f:
                original_metadata = json.load(f)
            
            target_column = original_metadata.get('target_name', 'target')
            model_type = original_metadata.get('model_type', 'random_forest_classifier')
            
            # Extract algorithm from model type
            if '_classifier' in model_type:
                algorithm = model_type.replace('_classifier', '')
            elif '_regressor' in model_type:
                algorithm = model_type.replace('_regressor', '')
            else:
                algorithm = model_type.split('_')[0]
            
            # Train new model
            result = self.training_script.train_and_save_model(
                training_data_path, target_column, algorithm
            )
            
            if not result['success']:
                raise ValueError(f"Training failed: {result.get('error', 'Unknown error')}")
            
            # Create new version
            new_model_path = os.path.join(self.model_dir, f"{result['model_id']}.joblib")
            new_metadata_path = os.path.join(self.model_dir, f"{result['model_id']}_metadata.json")
            
            new_version = self.version_manager.create_version(
                model_id=model_id,
                model_file_path=new_model_path,
                metadata_file_path=new_metadata_path,
                training_metrics=result['training_metrics'],
                validation_metrics=result['validation_metrics'],
                parent_version=active_version.version_id,
                notes=f"Automatic retraining triggered by {trigger.trigger_type}"
            )
            
            # Evaluate if new model is better
            should_deploy = self._evaluate_new_version(active_version, new_version)
            
            if should_deploy:
                # Deploy new version
                success = self.version_manager.activate_version(new_version.version_id)
                
                if success:
                    self.logger.info(f"Successfully deployed new version {new_version.version_id} for model {model_id}")
                    
                    # Send success alert
                    if self.metrics_collector:
                        alert = Alert(
                            alert_id=f"retraining_success_{new_version.version_id}",
                            alert_type='retraining_completed',
                            severity='low',
                            message=f"Model {model_id} successfully retrained and deployed (version {new_version.version_number})",
                            model_id=model_id,
                            timestamp=datetime.now(),
                            data={
                                'new_version_id': new_version.version_id,
                                'trigger_type': trigger.trigger_type,
                                'validation_metrics': new_version.validation_metrics
                            }
                        )
                        self.metrics_collector.alert_manager.send_alert(alert)
                else:
                    raise ValueError("Failed to deploy new model version")
            else:
                self.logger.info(f"New version {new_version.version_id} did not improve performance, keeping current version")
                
                # Send alert about no improvement
                if self.metrics_collector:
                    alert = Alert(
                        alert_id=f"retraining_no_improvement_{new_version.version_id}",
                        alert_type='retraining_no_improvement',
                        severity='low',
                        message=f"Model {model_id} retraining completed but new version did not improve performance",
                        model_id=model_id,
                        timestamp=datetime.now(),
                        data={
                            'new_version_id': new_version.version_id,
                            'trigger_type': trigger.trigger_type,
                            'kept_active': active_version.version_id
                        }
                    )
                    self.metrics_collector.alert_manager.send_alert(alert)
            
            # Clean up training data
            if os.path.exists(training_data_path):
                os.remove(training_data_path)
            
        except Exception as e:
            self.logger.error(f"Retraining failed for model {model_id}: {e}")
            
            # Send failure alert
            if self.metrics_collector:
                alert = Alert(
                    alert_id=f"retraining_failed_{model_id}_{int(datetime.now().timestamp())}",
                    alert_type='retraining_failed',
                    severity='high',
                    message=f"Automatic retraining failed for model {model_id}: {str(e)}",
                    model_id=model_id,
                    timestamp=datetime.now(),
                    data={
                        'error': str(e),
                        'trigger_type': trigger.trigger_type
                    }
                )
                self.metrics_collector.alert_manager.send_alert(alert)
        
        finally:
            # Remove from retraining set
            self.retraining_in_progress.discard(model_id)
    
    def _prepare_retraining_data(self, model_id: str) -> Optional[str]:
        """
        Prepare training data for retraining using recent predictions.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Path to prepared training data file, or None if failed
        """
        try:
            # Get recent predictions for this model
            recent_predictions = self.metrics_collector.storage.get_recent_predictions(
                model_id=model_id, 
                limit=self.config.get('retraining_data_limit', 10000)
            )
            
            if len(recent_predictions) < self.min_predictions_for_retraining:
                self.logger.warning(f"Insufficient data for retraining model {model_id}: {len(recent_predictions)} predictions")
                return None
            
            # Convert predictions to training data format
            # Note: This is a simplified approach - in practice, you'd need actual labels
            # For demo purposes, we'll use the original training data with some noise
            
            # Get original training data path from model metadata
            active_version = self.version_manager.get_active_version(model_id)
            if not active_version:
                return None
            
            # For demo purposes, use original training data
            # In a real system, you'd collect labeled data from production
            original_data_path = self.config.get('original_training_data', 'data/sample_classification.csv')
            
            if not os.path.exists(original_data_path):
                self.logger.error(f"Original training data not found: {original_data_path}")
                return None
            
            # Create retraining data path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            retraining_data_path = os.path.join(self.model_dir, f"retraining_data_{model_id}_{timestamp}.csv")
            
            # Copy original data for retraining (in practice, this would be new labeled data)
            shutil.copy2(original_data_path, retraining_data_path)
            
            self.logger.info(f"Prepared retraining data: {retraining_data_path}")
            return retraining_data_path
            
        except Exception as e:
            self.logger.error(f"Error preparing retraining data for model {model_id}: {e}")
            return None
    
    def _evaluate_new_version(self, current_version: ModelVersion, new_version: ModelVersion) -> bool:
        """
        Evaluate if new version is better than current version.
        
        Args:
            current_version: Currently active ModelVersion
            new_version: Newly trained ModelVersion
            
        Returns:
            True if new version should be deployed, False otherwise
        """
        # Compare validation metrics
        current_metrics = current_version.validation_metrics
        new_metrics = new_version.validation_metrics
        
        # Primary metric for comparison (accuracy for classification, r2_score for regression)
        primary_metric = 'accuracy' if 'accuracy' in new_metrics else 'r2_score'
        
        if primary_metric not in current_metrics or primary_metric not in new_metrics:
            self.logger.warning(f"Cannot compare versions: missing {primary_metric}")
            return False
        
        current_score = current_metrics[primary_metric]
        new_score = new_metrics[primary_metric]
        
        # Require minimum improvement threshold
        min_improvement = self.config.get('min_improvement_threshold', 0.01)
        improvement = new_score - current_score
        
        self.logger.info(f"Version comparison - Current: {current_score:.4f}, New: {new_score:.4f}, Improvement: {improvement:.4f}")
        
        return improvement > min_improvement
    
    def trigger_manual_retraining(self, model_id: str, notes: str = "") -> Dict[str, Any]:
        """
        Manually trigger retraining for a specific model.
        
        Args:
            model_id: Model identifier
            notes: Optional notes about why retraining was triggered
            
        Returns:
            Dictionary with trigger result
        """
        if model_id in self.retraining_in_progress:
            return {
                'success': False,
                'error': f'Model {model_id} is already being retrained'
            }
        
        trigger = RetrainingTrigger(
            trigger_id=f"manual_{model_id}_{int(datetime.now().timestamp())}",
            trigger_type='manual',
            model_id=model_id,
            threshold_value=0.0,
            current_value=1.0,
            triggered_at=datetime.now(),
            data={'notes': notes}
        )
        
        self._handle_retraining_trigger(trigger)
        
        return {
            'success': True,
            'trigger_id': trigger.trigger_id,
            'message': f'Manual retraining triggered for model {model_id}'
        }
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """
        Get current status of retraining system.
        
        Returns:
            Dictionary with system status
        """
        return {
            'monitoring_active': self.monitoring_active,
            'models_being_retrained': list(self.retraining_in_progress),
            'recent_triggers': [
                {
                    'trigger_id': t.trigger_id,
                    'trigger_type': t.trigger_type,
                    'model_id': t.model_id,
                    'triggered_at': t.triggered_at.isoformat(),
                    'threshold_value': t.threshold_value,
                    'current_value': t.current_value
                }
                for t in list(self.trigger_history)[-10:]  # Last 10 triggers
            ],
            'configuration': {
                'check_interval': self.check_interval,
                'accuracy_drop_threshold': self.accuracy_drop_threshold,
                'min_predictions_for_retraining': self.min_predictions_for_retraining,
                'retraining_data_window_hours': self.retraining_data_window_hours
            }
        }


# Global instance for use in Flask app
_retraining_system = None

def get_retraining_system(metrics_collector=None, config=None) -> AutomatedRetrainingSystem:
    """Get or create global retraining system instance."""
    global _retraining_system
    
    if _retraining_system is None:
        _retraining_system = AutomatedRetrainingSystem(
            metrics_collector=metrics_collector,
            config=config
        )
    
    return _retraining_system