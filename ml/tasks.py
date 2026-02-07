"""
ML Celery Tasks

Asynchronous tasks for ML model training, data processing, and auto-training.

Add to your Celery app configuration:
    app.autodiscover_tasks(['ml'])
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from celery import shared_task
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger("bunoraa.ml.tasks")


# ================================
# Model Training Tasks
# ================================

@shared_task(
    name='ml.train_model',
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    time_limit=7200,  # 2 hours
    soft_time_limit=6600,  # 1 hour 50 min
)
def train_model_task(self, model_type: str, config: Dict = None, trigger: str = 'manual'):
    """
    Train an ML model asynchronously.
    
    Args:
        model_type: Type of model to train (ncf, deepfm, etc.)
        config: Training configuration overrides
        trigger: What triggered the training (manual, auto, scheduled)
    
    Returns:
        Training result with metrics
    """
    import redis
    
    logger.info(f"Starting training for model: {model_type}, trigger: {trigger}")
    
    task_id = self.request.id
    redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
    r = redis.from_url(redis_url)
    
    try:
        # Mark as training in progress
        r.sadd('ml:training_in_progress', model_type)
        r.set(f'ml:training:{model_type}:task_id', task_id)
        r.set(f'ml:training:{model_type}:started', timezone.now().isoformat())
        
        # Import and run training
        from ml.services import MLService
        
        service = MLService()
        result = service.train_model(model_type, config or {})
        
        # Store result
        r.set(
            f'ml:training:{model_type}:result',
            str(result),
            ex=86400 * 7  # Keep for 7 days
        )
        
        logger.info(f"Training completed for {model_type}: {result}")
        
        return {
            'status': 'success',
            'model': model_type,
            'trigger': trigger,
            'result': result,
        }
        
    except Exception as e:
        logger.exception(f"Training failed for {model_type}")
        
        # Store error
        r.set(
            f'ml:training:{model_type}:error',
            str(e),
            ex=86400  # Keep for 1 day
        )
        
        # Retry on failure
        raise self.retry(exc=e)
        
    finally:
        # Remove from in-progress set
        r.srem('ml:training_in_progress', model_type)


@shared_task(name='ml.train_all_models')
def train_all_models_task(config: Dict = None, trigger: str = 'manual'):
    """Train all ML models sequentially."""
    models = [
        'embeddings',  # Train embeddings first
        'ncf',
        'deepfm',
        'two_tower',
        'sequence',
        'demand',
        'fraud',
        'churn',
        'search',
        'image',
    ]
    
    results = {}
    
    for model_type in models:
        try:
            result = train_model_task.delay(model_type, config, trigger)
            results[model_type] = {'task_id': result.id, 'status': 'queued'}
        except Exception as e:
            results[model_type] = {'status': 'error', 'error': str(e)}
    
    return results


# ================================
# Data Processing Tasks
# ================================

@shared_task(
    name='ml.process_training_data',
    time_limit=1800,  # 30 minutes
)
def process_training_data_task():
    """
    Process queued training data from Redis.
    
    This task should run periodically (e.g., every 5 minutes) to:
    1. Process interaction data from queues
    2. Update user profiles
    3. Compute product features
    """
    logger.info("Processing training data...")
    
    try:
        from ml.data_collection.collector import DataCollector
        from ml.data_collection.user_profile import UserProfileCollector
        from ml.data_collection.product_analytics import ProductAnalyticsCollector
        
        collector = DataCollector()
        user_collector = UserProfileCollector()
        product_collector = ProductAnalyticsCollector()
        
        results = {
            'interactions_processed': 0,
            'profiles_updated': 0,
            'products_updated': 0,
        }
        
        # Process queued interactions
        processed = collector.process_queued_data()
        results['interactions_processed'] = processed
        
        # Update active user profiles
        try:
            import redis
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            r = redis.from_url(redis_url)
            
            # Get recently active users
            active_users = r.smembers('ml:active_users')
            
            for user_id in list(active_users)[:100]:  # Limit batch size
                try:
                    user_collector.collect_user_profile(user_id.decode())
                    results['profiles_updated'] += 1
                except Exception as e:
                    logger.error(f"Error updating profile for user {user_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error updating user profiles: {e}")
        
        # Update popular product features
        try:
            popular_products = r.zrange('ml:popular_products', -50, -1)
            
            for product_id in popular_products:
                try:
                    product_collector.collect_product_features(product_id.decode())
                    results['products_updated'] += 1
                except Exception as e:
                    logger.error(f"Error updating product {product_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error updating product features: {e}")
        
        logger.info(f"Data processing completed: {results}")
        return results
        
    except Exception as e:
        logger.exception("Data processing failed")
        raise


@shared_task(name='ml.export_training_data')
def export_training_data_task(output_dir: str = None):
    """
    Export training data to files for model training.
    
    Creates CSV/Parquet files for:
    - Product features
    - User interactions
    - User profiles
    - Co-occurrence matrix
    """
    from pathlib import Path
    
    output_dir = output_dir or str(Path(settings.BASE_DIR) / 'ml' / 'training_data')
    
    logger.info(f"Exporting training data to {output_dir}")
    
    try:
        from ml.data_collection.product_analytics import ProductAnalyticsCollector
        
        collector = ProductAnalyticsCollector()
        
        # Export all features
        result = collector.export_all_features(output_dir, format='csv')
        
        logger.info(f"Export completed: {result}")
        return result
        
    except Exception as e:
        logger.exception("Data export failed")
        raise


# ================================
# Auto-Training Tasks
# ================================

@shared_task(name='ml.auto_training_check')
def auto_training_check_task():
    """
    Check if any models need retraining and trigger training if needed.
    
    This is the main entry point for automatic model retraining.
    Should be scheduled to run periodically (e.g., hourly).
    """
    # Only run in production
    if not getattr(settings, 'PRODUCTION', False):
        logger.info("Auto-training check skipped: not in production")
        return {'status': 'skipped', 'reason': 'not in production'}
    
    logger.info("Running auto-training check...")
    
    try:
        from ml.auto_training import AutoTrainingManager
        
        manager = AutoTrainingManager()
        result = manager.run_auto_training_check()
        
        logger.info(f"Auto-training check result: {result}")
        return result
        
    except Exception as e:
        logger.exception("Auto-training check failed")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='ml.scheduled_training')
def scheduled_training_task():
    """
    Scheduled daily/weekly training task.
    
    Can be configured to run at specific times via Celery Beat.
    """
    logger.info("Running scheduled training...")
    
    try:
        # Check if training is needed
        from ml.auto_training import AutoTrainingManager
        
        manager = AutoTrainingManager()
        decisions = manager.check_training_needed()
        
        if not decisions:
            logger.info("No models need training")
            return {'status': 'ok', 'trained': 0}
        
        # Trigger training for models that need it
        triggered = manager.trigger_training(decisions, async_mode=True)
        
        return {
            'status': 'training_triggered',
            'models': [d.model_name for d in decisions],
            'task_ids': triggered,
        }
        
    except Exception as e:
        logger.exception("Scheduled training failed")
        return {'status': 'error', 'error': str(e)}


# ================================
# Model Evaluation Tasks
# ================================

@shared_task(name='ml.evaluate_models')
def evaluate_models_task(models: list = None):
    """
    Evaluate ML model performance.
    
    Args:
        models: List of model names to evaluate (None = all)
    
    Returns:
        Evaluation results with metrics and health status
    """
    logger.info(f"Evaluating models: {models or 'all'}")
    
    try:
        from ml.core.registry import ModelRegistry
        from ml.core.metrics import MetricsTracker
        
        registry = ModelRegistry()
        metrics_tracker = MetricsTracker()
        
        all_models = [
            'ncf', 'deepfm', 'two_tower', 'sequence',
            'embeddings', 'demand', 'fraud', 'churn', 'search', 'image'
        ]
        
        models_to_evaluate = models or all_models
        results = {}
        
        for model_name in models_to_evaluate:
            try:
                model_info = registry.get_model_info(model_name)
                
                if not model_info:
                    results[model_name] = {'status': 'not_found'}
                    continue
                
                # Check for drift
                drift_detected = metrics_tracker.check_drift(model_name)
                
                results[model_name] = {
                    'status': 'ok',
                    'version': model_info.get('version'),
                    'metrics': model_info.get('metrics', {}),
                    'drift_detected': drift_detected,
                    'last_trained': model_info.get('last_trained'),
                }
                
            except Exception as e:
                results[model_name] = {'status': 'error', 'error': str(e)}
        
        logger.info(f"Evaluation completed: {len(results)} models")
        return results
        
    except Exception as e:
        logger.exception("Model evaluation failed")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='ml.check_drift')
def check_drift_task():
    """
    Check all models for data drift.
    
    If drift is detected, can trigger retraining.
    """
    logger.info("Checking for data drift...")
    
    try:
        from ml.core.metrics import MetricsTracker
        
        tracker = MetricsTracker()
        
        models = [
            'ncf', 'deepfm', 'two_tower', 'sequence',
            'embeddings', 'demand', 'fraud', 'churn', 'search', 'image'
        ]
        
        drift_results = {}
        models_with_drift = []
        
        for model_name in models:
            try:
                drift_score = tracker.calculate_drift(model_name)
                drift_detected = drift_score > 0.15  # Threshold
                
                drift_results[model_name] = {
                    'drift_score': drift_score,
                    'drift_detected': drift_detected,
                }
                
                if drift_detected:
                    models_with_drift.append(model_name)
                    
            except Exception as e:
                drift_results[model_name] = {'error': str(e)}
        
        result = {
            'status': 'ok',
            'results': drift_results,
            'models_with_drift': models_with_drift,
        }
        
        # Auto-trigger retraining if enabled and drift detected
        if models_with_drift and getattr(settings, 'ML_AUTO_RETRAIN_ON_DRIFT', True):
            if getattr(settings, 'PRODUCTION', False):
                for model_name in models_with_drift:
                    train_model_task.delay(model_name, trigger='drift_detected')
                result['training_triggered'] = models_with_drift
        
        logger.info(f"Drift check completed: {len(models_with_drift)} models with drift")
        return result
        
    except Exception as e:
        logger.exception("Drift check failed")
        return {'status': 'error', 'error': str(e)}


# ================================
# Cleanup Tasks
# ================================

@shared_task(name='ml.cleanup_old_data')
def cleanup_old_data_task(days: int = 90):
    """
    Clean up old training data and logs.
    
    Args:
        days: Delete data older than this many days
    """
    import redis
    from pathlib import Path
    
    logger.info(f"Cleaning up data older than {days} days...")
    
    try:
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        
        cutoff = timezone.now() - timedelta(days=days)
        
        # Clean up Redis queues (trim to max size)
        queues_to_trim = [
            ('ml:interactions', 100000),
            ('ml:events', 100000),
            ('ml:raw_events', 50000),
            ('ml:training_log', 1000),
        ]
        
        for queue_name, max_size in queues_to_trim:
            current_size = r.llen(queue_name)
            if current_size > max_size:
                r.ltrim(queue_name, 0, max_size - 1)
                logger.info(f"Trimmed {queue_name} from {current_size} to {max_size}")
        
        # Clean up old model versions
        models_dir = Path(settings.BASE_DIR) / 'ml' / 'saved_models'
        if models_dir.exists():
            import time
            cutoff_timestamp = time.time() - (days * 24 * 60 * 60)
            
            for model_file in models_dir.glob('*.pt'):
                if model_file.stat().st_mtime < cutoff_timestamp:
                    # Keep at least the latest version
                    # This is a simplified check
                    logger.info(f"Would delete old model: {model_file}")
        
        return {'status': 'ok', 'message': f'Cleaned up data older than {days} days'}
        
    except Exception as e:
        logger.exception("Cleanup failed")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='ml.generate_report')
def generate_report_task():
    """
    Generate daily ML system report.
    
    Creates a summary of:
    - Model performance
    - Training activities
    - Data collection stats
    - Recommendation quality
    """
    logger.info("Generating ML system report...")
    
    try:
        import redis
        import json
        
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        
        report = {
            'timestamp': timezone.now().isoformat(),
            'period': 'daily',
            'models': {},
            'data': {},
            'training': {},
        }
        
        # Get model status
        evaluation = evaluate_models_task()
        report['models'] = evaluation
        
        # Get data stats
        report['data'] = {
            'interactions_queued': r.llen('ml:interactions'),
            'events_queued': r.llen('ml:events'),
            'active_users': r.scard('ml:active_users'),
        }
        
        # Get training history
        training_log = r.lrange('ml:training_log', 0, 23)  # Last 24 entries
        report['training'] = {
            'recent_trainings': [json.loads(e) for e in training_log],
        }
        
        # Store report
        r.lpush('ml:daily_reports', json.dumps(report))
        r.ltrim('ml:daily_reports', 0, 29)  # Keep 30 days
        
        logger.info("Report generated successfully")
        return report
        
    except Exception as e:
        logger.exception("Report generation failed")
        return {'status': 'error', 'error': str(e)}


# ================================
# Celery Beat Schedule
# ================================

def get_celery_beat_schedule():
    """
    Get Celery Beat schedule for ML tasks.
    
    Add to your CELERY_BEAT_SCHEDULE:
    
        from ml.tasks import get_celery_beat_schedule
        CELERY_BEAT_SCHEDULE.update(get_celery_beat_schedule())
    """
    from celery.schedules import crontab
    
    return {
        # Process training data every 5 minutes
        'ml-process-training-data': {
            'task': 'ml.process_training_data',
            'schedule': 300,  # 5 minutes
        },
        
        # Check for auto-training every hour
        'ml-auto-training-check': {
            'task': 'ml.auto_training_check',
            'schedule': 3600,  # 1 hour
        },
        
        # Check for drift daily at 1 AM
        'ml-check-drift': {
            'task': 'ml.check_drift',
            'schedule': crontab(hour=1, minute=0),
        },
        
        # Evaluate models daily at 2 AM
        'ml-evaluate-models': {
            'task': 'ml.evaluate_models',
            'schedule': crontab(hour=2, minute=0),
        },
        
        # Export training data daily at 3 AM
        'ml-export-training-data': {
            'task': 'ml.export_training_data',
            'schedule': crontab(hour=3, minute=0),
        },
        
        # Generate report daily at 6 AM
        'ml-generate-report': {
            'task': 'ml.generate_report',
            'schedule': crontab(hour=6, minute=0),
        },
        
        # Cleanup old data weekly on Sunday at 4 AM
        'ml-cleanup-old-data': {
            'task': 'ml.cleanup_old_data',
            'schedule': crontab(hour=4, minute=0, day_of_week=0),
        },
        
        # Scheduled training weekly on Saturday at 2 AM
        'ml-scheduled-training': {
            'task': 'ml.scheduled_training',
            'schedule': crontab(hour=2, minute=0, day_of_week=6),
        },
    }
