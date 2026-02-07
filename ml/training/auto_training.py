"""
ML Auto-Training System

Production-ready automatic model training system that:
1. Monitors data collection metrics
2. Detects when retraining is needed
3. Triggers training automatically based on configurable thresholds
4. Integrates with Celery for background training

Usage in production:
    - Add 'ml.training.tasks.auto_training_check' to Celery beat schedule
    - Configure ML_AUTO_TRAINING = True in settings
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json

from django.conf import settings
from django.utils import timezone

logger = logging.getLogger("bunoraa.ml.auto_training")


class TrainingTrigger(Enum):
    """Reasons for triggering training."""
    SCHEDULED = "scheduled"
    DATA_THRESHOLD = "data_threshold"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DROP = "performance_drop"
    MANUAL = "manual"
    NEW_MODEL = "new_model"


@dataclass
class TrainingConfig:
    """Configuration for auto-training."""
    
    # Enable/disable auto-training
    enabled: bool = True
    
    # Minimum data requirements
    min_interactions: int = 1000
    min_users: int = 100
    min_products: int = 50
    
    # New data thresholds (trigger when X new items since last training)
    new_interactions_threshold: int = 5000
    new_users_threshold: int = 500
    new_products_threshold: int = 100
    
    # Time-based training
    max_days_between_training: int = 7
    training_hour: int = 2  # 2 AM local time
    
    # Performance thresholds
    min_performance_score: float = 0.6
    max_performance_drop: float = 0.1
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_threshold: float = 0.15
    
    # Resource management
    max_concurrent_trainings: int = 2
    training_timeout_minutes: int = 120
    
    # Models to auto-train (empty = all)
    models_to_train: List[str] = field(default_factory=list)
    
    # Exclude models from auto-training
    models_to_exclude: List[str] = field(default_factory=list)
    
    @classmethod
    def from_settings(cls) -> 'TrainingConfig':
        """Load config from Django settings."""
        config_dict = getattr(settings, 'ML_AUTO_TRAINING_CONFIG', {})
        
        # Check production mode
        is_production = getattr(settings, 'PRODUCTION', False)
        
        return cls(
            enabled=is_production and config_dict.get('enabled', True),
            min_interactions=config_dict.get('min_interactions', 1000),
            min_users=config_dict.get('min_users', 100),
            min_products=config_dict.get('min_products', 50),
            new_interactions_threshold=config_dict.get('new_interactions_threshold', 5000),
            new_users_threshold=config_dict.get('new_users_threshold', 500),
            new_products_threshold=config_dict.get('new_products_threshold', 100),
            max_days_between_training=config_dict.get('max_days_between_training', 7),
            training_hour=config_dict.get('training_hour', 2),
            min_performance_score=config_dict.get('min_performance_score', 0.6),
            max_performance_drop=config_dict.get('max_performance_drop', 0.1),
            enable_drift_detection=config_dict.get('enable_drift_detection', True),
            drift_threshold=config_dict.get('drift_threshold', 0.15),
            max_concurrent_trainings=config_dict.get('max_concurrent_trainings', 2),
            training_timeout_minutes=config_dict.get('training_timeout_minutes', 120),
            models_to_train=config_dict.get('models_to_train', []),
            models_to_exclude=config_dict.get('models_to_exclude', []),
        )


@dataclass
class TrainingDecision:
    """Decision about whether to train a model."""
    model_name: str
    should_train: bool
    trigger: Optional[TrainingTrigger] = None
    priority: int = 0
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class AutoTrainingManager:
    """
    Manages automatic model training in production.
    
    Monitors data collection and model performance to determine
    when retraining is needed and triggers training accordingly.
    """
    
    ALL_MODELS = [
        'ncf', 'deepfm', 'two_tower', 'sequence',
        'embeddings', 'demand', 'fraud', 'churn', 'search', 'image'
    ]
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig.from_settings()
        self._redis = None
        self._registry = None
        self._metrics_tracker = None
    
    @property
    def redis(self):
        """Lazy load Redis connection."""
        if self._redis is None:
            import redis
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            self._redis = redis.from_url(redis_url)
        return self._redis
    
    @property
    def registry(self):
        """Lazy load model registry."""
        if self._registry is None:
            from ..core.registry import ModelRegistry
            self._registry = ModelRegistry()
        return self._registry
    
    @property
    def metrics_tracker(self):
        """Lazy load metrics tracker."""
        if self._metrics_tracker is None:
            from ..core.metrics import MetricsTracker
            self._metrics_tracker = MetricsTracker()
        return self._metrics_tracker
    
    def check_training_needed(self) -> List[TrainingDecision]:
        """
        Check all models and determine which need training.
        
        Returns:
            List of TrainingDecision objects for models that need training.
        """
        if not self.config.enabled:
            logger.info("Auto-training is disabled")
            return []
        
        # Check if we're in the right time window
        if not self._is_training_window():
            logger.debug("Not in training window")
            return []
        
        decisions = []
        models_to_check = self._get_models_to_check()
        
        for model_name in models_to_check:
            try:
                decision = self._check_model(model_name)
                if decision.should_train:
                    decisions.append(decision)
            except Exception as e:
                logger.error(f"Error checking model {model_name}: {e}")
        
        # Sort by priority (higher = more urgent)
        decisions.sort(key=lambda d: d.priority, reverse=True)
        
        return decisions
    
    def _get_models_to_check(self) -> List[str]:
        """Get list of models to check for training."""
        if self.config.models_to_train:
            models = self.config.models_to_train
        else:
            models = self.ALL_MODELS.copy()
        
        # Remove excluded models
        models = [m for m in models if m not in self.config.models_to_exclude]
        
        return models
    
    def _is_training_window(self) -> bool:
        """Check if current time is suitable for training."""
        now = timezone.localtime()
        
        # Check if we're within 1 hour of the configured training time
        target_hour = self.config.training_hour
        current_hour = now.hour
        
        return abs(current_hour - target_hour) <= 1
    
    def _check_model(self, model_name: str) -> TrainingDecision:
        """
        Check if a specific model needs training.
        
        Checks in order:
        1. New model (never trained)
        2. Data drift detected
        3. Performance drop
        4. Scheduled retraining
        5. Data threshold exceeded
        """
        decision = TrainingDecision(
            model_name=model_name,
            should_train=False,
        )
        
        # Get model info
        model_info = self.registry.get_model_info(model_name)
        
        # Check 1: New model (never trained)
        if not model_info or not model_info.get('last_trained'):
            if self._has_minimum_data():
                decision.should_train = True
                decision.trigger = TrainingTrigger.NEW_MODEL
                decision.priority = 100
                decision.reason = "Model has never been trained"
                return decision
        
        last_trained = model_info.get('last_trained')
        if isinstance(last_trained, str):
            last_trained = datetime.fromisoformat(last_trained)
        
        # Check 2: Drift detection
        if self.config.enable_drift_detection:
            drift_score = self._check_drift(model_name)
            decision.metrics['drift_score'] = drift_score
            
            if drift_score > self.config.drift_threshold:
                decision.should_train = True
                decision.trigger = TrainingTrigger.DRIFT_DETECTED
                decision.priority = 90
                decision.reason = f"Data drift detected: {drift_score:.3f}"
                return decision
        
        # Check 3: Performance drop
        current_score = model_info.get('metrics', {}).get('score', 0)
        baseline_score = model_info.get('baseline_score', current_score)
        
        if baseline_score > 0:
            performance_drop = baseline_score - current_score
            decision.metrics['performance_drop'] = performance_drop
            
            if performance_drop > self.config.max_performance_drop:
                decision.should_train = True
                decision.trigger = TrainingTrigger.PERFORMANCE_DROP
                decision.priority = 85
                decision.reason = f"Performance dropped by {performance_drop:.3f}"
                return decision
            
            if current_score < self.config.min_performance_score:
                decision.should_train = True
                decision.trigger = TrainingTrigger.PERFORMANCE_DROP
                decision.priority = 80
                decision.reason = f"Performance below threshold: {current_score:.3f}"
                return decision
        
        # Check 4: Scheduled retraining
        if last_trained:
            days_since_training = (timezone.now() - last_trained).days
            decision.metrics['days_since_training'] = days_since_training
            
            if days_since_training >= self.config.max_days_between_training:
                decision.should_train = True
                decision.trigger = TrainingTrigger.SCHEDULED
                decision.priority = 50
                decision.reason = f"{days_since_training} days since last training"
                return decision
        
        # Check 5: Data threshold
        new_data_stats = self._get_new_data_stats(model_name, last_trained)
        decision.metrics.update(new_data_stats)
        
        if (new_data_stats.get('new_interactions', 0) >= self.config.new_interactions_threshold or
            new_data_stats.get('new_users', 0) >= self.config.new_users_threshold or
            new_data_stats.get('new_products', 0) >= self.config.new_products_threshold):
            decision.should_train = True
            decision.trigger = TrainingTrigger.DATA_THRESHOLD
            decision.priority = 40
            decision.reason = f"New data threshold exceeded: {new_data_stats}"
            return decision
        
        return decision
    
    def _has_minimum_data(self) -> bool:
        """Check if we have minimum data for training."""
        try:
            from apps.catalog.models import Product
            from apps.accounts.models import UserProfile
            
            product_count = Product.objects.count()
            user_count = UserProfile.objects.count()
            
            # Check Redis for interactions
            interaction_count = self.redis.llen('ml:interactions')
            
            return (
                product_count >= self.config.min_products and
                user_count >= self.config.min_users and
                interaction_count >= self.config.min_interactions
            )
        except Exception as e:
            logger.error(f"Error checking minimum data: {e}")
            return False
    
    def _check_drift(self, model_name: str) -> float:
        """
        Check for data drift for a model.
        
        Returns:
            Drift score (0-1, higher = more drift)
        """
        try:
            return self.metrics_tracker.calculate_drift(model_name)
        except Exception as e:
            logger.error(f"Error calculating drift for {model_name}: {e}")
            return 0.0
    
    def _get_new_data_stats(
        self,
        model_name: str,
        since: Optional[datetime]
    ) -> Dict[str, int]:
        """Get statistics about new data since last training."""
        if not since:
            since = timezone.now() - timedelta(days=30)
        
        stats = {
            'new_interactions': 0,
            'new_users': 0,
            'new_products': 0,
        }
        
        try:
            from apps.catalog.models import Product
            from apps.accounts.models import UserProfile
            
            # Count new products
            stats['new_products'] = Product.objects.filter(
                created_at__gte=since
            ).count()
            
            # Count new users
            stats['new_users'] = UserProfile.objects.filter(
                created_at__gte=since
            ).count()
            
            # Count new interactions from Redis
            stats['new_interactions'] = self.redis.llen('ml:interactions')
            
        except Exception as e:
            logger.error(f"Error getting new data stats: {e}")
        
        return stats
    
    def trigger_training(
        self,
        decisions: List[TrainingDecision],
        async_mode: bool = True
    ) -> List[str]:
        """
        Trigger training for models that need it.
        
        Args:
            decisions: List of training decisions
            async_mode: Whether to run training asynchronously
        
        Returns:
            List of task IDs or model names that were triggered
        """
        if not decisions:
            return []
        
        # Check concurrent training limit
        currently_training = self._get_currently_training()
        available_slots = self.config.max_concurrent_trainings - len(currently_training)
        
        if available_slots <= 0:
            logger.warning(f"Training limit reached: {len(currently_training)} already running")
            return []
        
        triggered = []
        
        for decision in decisions[:available_slots]:
            try:
                if async_mode:
                    task_id = self._trigger_async_training(decision)
                    triggered.append(task_id)
                else:
                    self._trigger_sync_training(decision)
                    triggered.append(decision.model_name)
                
                # Log training trigger
                self._log_training_trigger(decision)
                
            except Exception as e:
                logger.error(f"Error triggering training for {decision.model_name}: {e}")
        
        return triggered
    
    def _get_currently_training(self) -> List[str]:
        """Get list of models currently being trained."""
        try:
            training_set = self.redis.smembers('ml:training_in_progress')
            return [m.decode() for m in training_set]
        except Exception:
            return []
    
    def _trigger_async_training(self, decision: TrainingDecision) -> str:
        """Trigger asynchronous training via Celery."""
        from .tasks import train_recommendation_model
        
        task = train_recommendation_model.delay(
            model_type=decision.model_name,
            config={},
        )
        
        # Mark as training in progress
        self.redis.sadd('ml:training_in_progress', decision.model_name)
        self.redis.expire('ml:training_in_progress', 3600 * 4)  # 4 hour expiry
        
        return task.id
    
    def _trigger_sync_training(self, decision: TrainingDecision):
        """Trigger synchronous training."""
        from .tasks import train_recommendation_model
        
        # Mark as training
        self.redis.sadd('ml:training_in_progress', decision.model_name)
        
        try:
            # Execute the training task synchronously
            train_recommendation_model(decision.model_name, {})
        finally:
            # Remove from training set
            self.redis.srem('ml:training_in_progress', decision.model_name)
    
    def _log_training_trigger(self, decision: TrainingDecision):
        """Log training trigger for auditing."""
        log_entry = {
            'timestamp': timezone.now().isoformat(),
            'model': decision.model_name,
            'trigger': decision.trigger.value if decision.trigger else 'unknown',
            'reason': decision.reason,
            'priority': decision.priority,
            'metrics': decision.metrics,
        }
        
        # Store in Redis for history
        self.redis.lpush('ml:training_log', json.dumps(log_entry))
        self.redis.ltrim('ml:training_log', 0, 999)  # Keep last 1000 entries
        
        logger.info(
            f"Training triggered for {decision.model_name}: "
            f"{decision.trigger.value if decision.trigger else 'unknown'} - {decision.reason}"
        )
    
    def get_training_history(self, limit: int = 100) -> List[Dict]:
        """Get recent training history."""
        entries = self.redis.lrange('ml:training_log', 0, limit - 1)
        return [json.loads(e) for e in entries]
    
    def run_auto_training_check(self) -> Dict[str, Any]:
        """
        Main entry point for auto-training check.
        
        This should be called periodically (e.g., via Celery beat).
        
        Returns:
            Summary of actions taken.
        """
        if not self.config.enabled:
            return {'status': 'disabled'}
        
        # Check production mode
        is_production = getattr(settings, 'PRODUCTION', False)
        if not is_production:
            logger.info("Auto-training skipped: not in production mode")
            return {'status': 'skipped', 'reason': 'not in production'}
        
        logger.info("Running auto-training check...")
        
        try:
            # Check which models need training
            decisions = self.check_training_needed()
            
            if not decisions:
                logger.info("No models need training")
                return {
                    'status': 'ok',
                    'models_checked': len(self._get_models_to_check()),
                    'models_to_train': 0,
                }
            
            # Trigger training
            triggered = self.trigger_training(decisions, async_mode=True)
            
            return {
                'status': 'training_triggered',
                'models_checked': len(self._get_models_to_check()),
                'models_to_train': len(decisions),
                'triggered': triggered,
                'decisions': [asdict(d) for d in decisions],
            }
            
        except Exception as e:
            logger.exception("Auto-training check failed")
            return {
                'status': 'error',
                'error': str(e),
            }


def get_celery_beat_schedule() -> dict:
    """
    Get Celery beat schedule for auto-training.
    
    Add this to your CELERY_BEAT_SCHEDULE in settings:
    
    from ml.training.auto_training import get_celery_beat_schedule
    CELERY_BEAT_SCHEDULE.update(get_celery_beat_schedule())
    """
    return {
        'ml-auto-training-check': {
            'task': 'ml.auto_training_check',
            'schedule': 3600,  # Every hour
        },
        'ml-process-training-queues': {
            'task': 'ml.process_training_queues',
            'schedule': 300,  # Every 5 minutes
        },
    }
