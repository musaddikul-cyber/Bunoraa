"""
Celery Tasks for ML Model Training

Background tasks for automated model training and inference.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from django.utils import timezone

try:
    from celery import shared_task
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    # Mock decorator
    def shared_task(func=None, **kwargs):
        def wrapper(f):
            return f
        return wrapper(func) if func else wrapper

logger = logging.getLogger("bunoraa.ml.tasks")


# ==============================================================================
# RECOMMENDATION MODEL TASKS
# ==============================================================================

@shared_task(bind=True, max_retries=3, default_retry_delay=300)
def train_recommendation_model(self, model_type: str = "ncf", config: Optional[Dict] = None):
    """
    Train recommendation model.
    
    Args:
        model_type: Type of model (ncf, deepfm, two_tower, sequence)
        config: Optional configuration overrides
    """
    try:
        from django.db import connection
        from ..models.recommender import (
            NeuralCollaborativeFiltering,
            DeepFM,
            TwoTowerRecommender,
            SequenceRecommender
        )
        from ..core.registry import ModelRegistry
        from .trainer import ModelTrainer, TrainerConfig
        from .data_loader import DataConfig, DatasetBuilder
        
        logger.info(f"Starting recommendation model training: {model_type}")
        
        # Get model class
        model_classes = {
            "ncf": NeuralCollaborativeFiltering,
            "deepfm": DeepFM,
            "two_tower": TwoTowerRecommender,
            "sequence": SequenceRecommender,
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_classes[model_type]
        
        # Load data
        interactions = _load_interaction_data()
        
        if not interactions:
            logger.warning("No interaction data available for training")
            return {"status": "skipped", "reason": "no_data"}
        
        # Build datasets
        data_config = DataConfig()
        builder = DatasetBuilder(data_config)
        train_loader, val_loader, test_loader = builder.build_recommendation_dataset(interactions)
        
        # Initialize model
        model_config = _get_model_config(model_type, config)
        model = model_class(**model_config)
        
        # Train
        trainer_config = TrainerConfig(
            epochs=50,
            learning_rate=0.001,
            early_stopping=True,
            patience=5
        )
        
        trainer = ModelTrainer(model.model, trainer_config)
        history = trainer.train(train_loader, val_loader)
        
        # Evaluate
        test_loss, metrics = trainer._validate(test_loader)
        
        # Save to registry
        registry = ModelRegistry()
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        registry.register(
            model_name=f"recommendation_{model_type}",
            version=version,
            model=model,
            metrics={"test_loss": test_loss, **metrics}
        )
        
        logger.info(f"Recommendation model training completed: {model_type} v{version}")
        
        return {
            "status": "success",
            "model_type": model_type,
            "version": version,
            "test_loss": test_loss,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Recommendation model training failed: {e}")
        raise self.retry(exc=e)


@shared_task(bind=True, max_retries=3)
def train_embedding_models(self):
    """Train product and user embedding models."""
    try:
        from ..models.embeddings import ProductEmbeddingModel, UserEmbeddingModel
        from ..core.registry import ModelRegistry
        
        logger.info("Starting embedding model training")
        
        # Train product embeddings
        product_model = ProductEmbeddingModel()
        product_data = _load_product_features()
        
        if product_data:
            product_model.train(product_data)
            
            registry = ModelRegistry()
            registry.register(
                model_name="product_embeddings",
                version=datetime.now().strftime("%Y%m%d_%H%M%S"),
                model=product_model
            )
        
        # Train user embeddings
        user_model = UserEmbeddingModel()
        user_data = _load_user_sequences()
        
        if user_data:
            user_model.train(user_data)
            
            registry = ModelRegistry()
            registry.register(
                model_name="user_embeddings",
                version=datetime.now().strftime("%Y%m%d_%H%M%S"),
                model=user_model
            )
        
        logger.info("Embedding model training completed")
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Embedding model training failed: {e}")
        raise self.retry(exc=e)


# ==============================================================================
# FORECASTING TASKS
# ==============================================================================

@shared_task(bind=True, max_retries=3)
def train_demand_forecaster(self, product_ids: Optional[List[str]] = None):
    """Train demand forecasting model."""
    try:
        from ..models.forecasting import DemandForecaster
        from ..core.registry import ModelRegistry
        
        logger.info("Starting demand forecaster training")
        
        # Load time series data
        time_series_data = _load_sales_timeseries(product_ids)
        
        if not time_series_data:
            logger.warning("No sales data available for training")
            return {"status": "skipped", "reason": "no_data"}
        
        # Train model
        model = DemandForecaster()
        model.train(time_series_data)
        
        # Register
        registry = ModelRegistry()
        registry.register(
            model_name="demand_forecaster",
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            model=model
        )
        
        logger.info("Demand forecaster training completed")
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Demand forecaster training failed: {e}")
        raise self.retry(exc=e)


@shared_task(bind=True, max_retries=3)
def train_price_optimizer(self):
    """Train price optimization model."""
    try:
        from ..models.forecasting import PriceOptimizer
        from ..core.registry import ModelRegistry
        
        logger.info("Starting price optimizer training")
        
        # Load pricing data
        pricing_data = _load_pricing_data()
        
        if not pricing_data:
            logger.warning("No pricing data available for training")
            return {"status": "skipped", "reason": "no_data"}
        
        # Train model
        model = PriceOptimizer()
        model.train(pricing_data)
        
        # Register
        registry = ModelRegistry()
        registry.register(
            model_name="price_optimizer",
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            model=model
        )
        
        logger.info("Price optimizer training completed")
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Price optimizer training failed: {e}")
        raise self.retry(exc=e)


# ==============================================================================
# FRAUD & RISK TASKS
# ==============================================================================

@shared_task(bind=True, max_retries=3)
def train_fraud_detector(self):
    """Train fraud detection model."""
    try:
        from ..models.fraud import FraudDetector
        from ..core.registry import ModelRegistry
        
        logger.info("Starting fraud detector training")
        
        # Load fraud data
        fraud_data = _load_fraud_data()
        
        if not fraud_data:
            logger.warning("No fraud data available for training")
            return {"status": "skipped", "reason": "no_data"}
        
        # Train model
        model = FraudDetector()
        model.train(fraud_data)
        
        # Register
        registry = ModelRegistry()
        registry.register(
            model_name="fraud_detector",
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            model=model
        )
        
        logger.info("Fraud detector training completed")
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Fraud detector training failed: {e}")
        raise self.retry(exc=e)


@shared_task(bind=True, max_retries=3)
def train_churn_predictor(self):
    """Train churn prediction model."""
    try:
        from ..models.churn import ChurnPredictor
        from ..core.registry import ModelRegistry
        
        logger.info("Starting churn predictor training")
        
        # Load customer data
        customer_data = _load_customer_data()
        
        if not customer_data:
            logger.warning("No customer data available for training")
            return {"status": "skipped", "reason": "no_data"}
        
        # Train model
        model = ChurnPredictor()
        model.train(customer_data)
        
        # Register
        registry = ModelRegistry()
        registry.register(
            model_name="churn_predictor",
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            model=model
        )
        
        logger.info("Churn predictor training completed")
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Churn predictor training failed: {e}")
        raise self.retry(exc=e)


# ==============================================================================
# SEARCH & VISION TASKS
# ==============================================================================

@shared_task(bind=True, max_retries=3)
def train_search_model(self):
    """Train semantic search model."""
    try:
        from ..models.search import SemanticSearchModel
        from ..core.registry import ModelRegistry
        
        logger.info("Starting search model training")
        
        # Load search data
        search_data = _load_search_data()
        
        if not search_data:
            logger.warning("No search data available for training")
            return {"status": "skipped", "reason": "no_data"}
        
        # Train model
        model = SemanticSearchModel()
        model.train(search_data)
        
        # Register
        registry = ModelRegistry()
        registry.register(
            model_name="semantic_search",
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            model=model
        )
        
        logger.info("Search model training completed")
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Search model training failed: {e}")
        raise self.retry(exc=e)


@shared_task(bind=True, max_retries=3)
def train_image_classifier(self):
    """Train product image classifier."""
    try:
        from ..models.vision import ProductImageClassifier
        from ..core.registry import ModelRegistry
        
        logger.info("Starting image classifier training")
        
        # Load image data
        image_data = _load_image_data()
        
        if not image_data:
            logger.warning("No image data available for training")
            return {"status": "skipped", "reason": "no_data"}
        
        # Train model
        model = ProductImageClassifier()
        model.train(image_data)
        
        # Register
        registry = ModelRegistry()
        registry.register(
            model_name="product_classifier",
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            model=model
        )
        
        logger.info("Image classifier training completed")
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Image classifier training failed: {e}")
        raise self.retry(exc=e)


# ==============================================================================
# BATCH INFERENCE TASKS
# ==============================================================================

@shared_task(bind=True)
def generate_recommendations_batch(self, user_ids: Optional[List[str]] = None):
    """Generate recommendations for all or specific users."""
    try:
        from ..core.registry import ModelRegistry
        from ..core.inference import InferenceEngine
        
        logger.info("Starting batch recommendation generation")
        
        # Load model
        registry = ModelRegistry()
        model = registry.get_latest("recommendation_ncf")
        
        if not model:
            logger.warning("No recommendation model available")
            return {"status": "skipped", "reason": "no_model"}
        
        # Load users
        if user_ids is None:
            user_ids = _get_active_user_ids()
        
        # Generate recommendations
        engine = InferenceEngine(model)
        recommendations = {}
        
        for user_id in user_ids:
            try:
                recs = engine.predict({"user_id": user_id})
                recommendations[user_id] = recs
            except Exception as e:
                logger.warning(f"Failed to generate recommendations for user {user_id}: {e}")
        
        # Save to cache/database
        _save_recommendations(recommendations)
        
        logger.info(f"Generated recommendations for {len(recommendations)} users")
        
        return {
            "status": "success",
            "users_processed": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Batch recommendation generation failed: {e}")
        raise


@shared_task(bind=True)
def update_product_embeddings(self):
    """Update product embeddings for all products."""
    try:
        from ..core.registry import ModelRegistry
        
        logger.info("Updating product embeddings")
        
        # Load model
        registry = ModelRegistry()
        model = registry.get_latest("product_embeddings")
        
        if not model:
            logger.warning("No product embedding model available")
            return {"status": "skipped", "reason": "no_model"}
        
        # Load products
        products = _load_all_products()
        
        # Generate embeddings
        embeddings = model.encode_products(products)
        
        # Save to feature store
        _save_product_embeddings(embeddings)
        
        logger.info(f"Updated embeddings for {len(products)} products")
        
        return {
            "status": "success",
            "products_processed": len(products)
        }
        
    except Exception as e:
        logger.error(f"Product embedding update failed: {e}")
        raise


# ==============================================================================
# SCHEDULED TASKS
# ==============================================================================

@shared_task
def scheduled_model_training():
    """
    Scheduled task to train all models.
    Run daily or weekly.
    """
    logger.info("Starting scheduled model training")
    
    # Train all models
    tasks = [
        train_recommendation_model.delay("ncf"),
        train_embedding_models.delay(),
        train_demand_forecaster.delay(),
        train_churn_predictor.delay(),
    ]
    
    return {"status": "scheduled", "tasks": len(tasks)}


@shared_task
def scheduled_batch_inference():
    """
    Scheduled task to run batch inference.
    Run hourly or daily.
    """
    logger.info("Starting scheduled batch inference")
    
    tasks = [
        generate_recommendations_batch.delay(),
        update_product_embeddings.delay(),
    ]
    
    return {"status": "scheduled", "tasks": len(tasks)}


@shared_task
def model_health_check():
    """Check health of deployed models."""
    try:
        from ..core.registry import ModelRegistry
        from ..core.metrics import MetricsTracker
        
        logger.info("Running model health check")
        
        registry = ModelRegistry()
        tracker = MetricsTracker()
        
        health_status = {}
        
        # Check each model
        model_names = [
            "recommendation_ncf",
            "product_embeddings",
            "user_embeddings",
            "demand_forecaster",
            "fraud_detector",
            "churn_predictor",
        ]
        
        for model_name in model_names:
            model = registry.get_latest(model_name)
            
            if model:
                # Check for drift
                drift = tracker.check_drift(model_name)
                
                health_status[model_name] = {
                    "available": True,
                    "drift_detected": drift.get("drift_detected", False),
                    "last_updated": model.metadata.get("created_at"),
                }
            else:
                health_status[model_name] = {
                    "available": False,
                    "drift_detected": False,
                }
        
        logger.info(f"Health check completed: {health_status}")
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise


# ==============================================================================
# DATA LOADING HELPERS
# ==============================================================================

def _load_interaction_data() -> List[Dict[str, Any]]:
    """Load user-product interaction data from database."""
    try:
        from apps.recommendations.models import Interaction
        
        interactions = Interaction.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=90)
        ).values("user_id", "product_id", "interaction_type", "created_at")
        
        return [
            {
                "user_id": i["user_id"],
                "product_id": i["product_id"],
                "label": 1 if i["interaction_type"] in ["purchase", "add_to_cart"] else 0,
                "timestamp": i["created_at"].timestamp(),
            }
            for i in interactions
        ]
    except Exception as e:
        logger.warning(f"Failed to load interaction data: {e}")
        return []


def _load_product_features() -> List[Dict[str, Any]]:
    """Load product features from database."""
    try:
        from apps.catalog.models import Product
        
        products = Product.objects.filter(is_active=True).values()
        return list(products)
    except Exception as e:
        logger.warning(f"Failed to load product features: {e}")
        return []


def _load_user_sequences() -> Dict[int, List[int]]:
    """Load user interaction sequences."""
    try:
        from apps.recommendations.models import Interaction
        
        interactions = Interaction.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=90)
        ).order_by("user_id", "created_at").values("user_id", "product_id")
        
        sequences = {}
        for i in interactions:
            user_id = i["user_id"]
            if user_id not in sequences:
                sequences[user_id] = []
            sequences[user_id].append(i["product_id"])
        
        return sequences
    except Exception as e:
        logger.warning(f"Failed to load user sequences: {e}")
        return {}


def _load_sales_timeseries(product_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load sales time series data."""
    try:
        from apps.orders.models import OrderItem
        from django.db.models import Sum
        from django.db.models.functions import TruncDate
        
        query = OrderItem.objects.all()
        if product_ids:
            query = query.filter(product_id__in=product_ids)
        
        daily_sales = query.annotate(
            date=TruncDate("order__created_at")
        ).values("date").annotate(
            total=Sum("quantity")
        ).order_by("date")
        
        return {
            "timestamps": [s["date"] for s in daily_sales],
            "values": [s["total"] for s in daily_sales],
        }
    except Exception as e:
        logger.warning(f"Failed to load sales timeseries: {e}")
        return {}


def _load_pricing_data() -> List[Dict[str, Any]]:
    """Load pricing and sales data."""
    try:
        from apps.orders.models import OrderItem
        
        order_items = OrderItem.objects.select_related("product").values(
            "product_id", "price", "quantity", "order__created_at"
        )[:100000]
        
        return list(order_items)
    except Exception as e:
        logger.warning(f"Failed to load pricing data: {e}")
        return []


def _load_fraud_data() -> List[Dict[str, Any]]:
    """Load transaction data with fraud labels."""
    try:
        from apps.orders.models import Order
        
        orders = Order.objects.values(
            "id", "user_id", "total_amount", "payment_method",
            "created_at", "is_fraudulent"
        )[:100000]
        
        return list(orders)
    except Exception as e:
        logger.warning(f"Failed to load fraud data: {e}")
        return []


def _load_customer_data() -> List[Dict[str, Any]]:
    """Load customer data for churn prediction."""
    try:
        from apps.accounts.models import User
        from apps.orders.models import Order
        from django.db.models import Count, Sum, Avg
        
        customers = User.objects.annotate(
            order_count=Count("orders"),
            total_spent=Sum("orders__total_amount"),
            avg_order_value=Avg("orders__total_amount"),
        ).values()
        
        return list(customers)
    except Exception as e:
        logger.warning(f"Failed to load customer data: {e}")
        return []


def _load_search_data() -> List[Dict[str, Any]]:
    """Load search query data."""
    try:
        from apps.analytics.models import SearchLog
        
        searches = SearchLog.objects.values(
            "query", "results_clicked", "user_id", "created_at"
        )[:100000]
        
        return list(searches)
    except Exception as e:
        logger.warning(f"Failed to load search data: {e}")
        return []


def _load_image_data() -> List[Dict[str, Any]]:
    """Load product image data."""
    try:
        from apps.catalog.models import ProductImage
        
        images = ProductImage.objects.select_related("product").values(
            "id", "image", "product_id", "product__category_id"
        )
        
        return list(images)
    except Exception as e:
        logger.warning(f"Failed to load image data: {e}")
        return []


def _get_active_user_ids() -> List[str]:
    """Get list of active user IDs."""
    try:
        from apps.accounts.models import User
        
        users = User.objects.filter(
            is_active=True,
            last_login__gte=timezone.now() - timedelta(days=30)
        ).values_list("id", flat=True)
        
        return list(users)
    except Exception as e:
        logger.warning(f"Failed to get active users: {e}")
        return []


def _load_all_products() -> List[Dict[str, Any]]:
    """Load all products."""
    try:
        from apps.catalog.models import Product
        
        products = Product.objects.filter(is_active=True).values()
        return list(products)
    except Exception as e:
        logger.warning(f"Failed to load products: {e}")
        return []


def _save_recommendations(recommendations: Dict[str, Any]):
    """Save recommendations to cache/database."""
    try:
        from django.core.cache import cache
        
        for user_id, recs in recommendations.items():
            cache_key = f"recommendations:{user_id}"
            cache.set(cache_key, recs, timeout=3600 * 24)  # 24 hours
    except Exception as e:
        logger.warning(f"Failed to save recommendations: {e}")


def _save_product_embeddings(embeddings: Dict[str, Any]):
    """Save product embeddings to feature store."""
    try:
        from django.core.cache import cache
        
        for product_id, embedding in embeddings.items():
            cache_key = f"embedding:product:{product_id}"
            cache.set(cache_key, embedding.tolist(), timeout=3600 * 24 * 7)  # 7 days
    except Exception as e:
        logger.warning(f"Failed to save embeddings: {e}")


def _get_model_config(model_type: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Get model configuration."""
    default_configs = {
        "ncf": {
            "num_users": 100000,
            "num_products": 50000,
            "embedding_dim": 64,
        },
        "deepfm": {
            "num_users": 100000,
            "num_products": 50000,
            "embedding_dim": 64,
            "num_features": 100,
        },
        "two_tower": {
            "user_features_dim": 64,
            "product_features_dim": 128,
            "embedding_dim": 128,
        },
        "sequence": {
            "num_products": 50000,
            "embedding_dim": 64,
            "max_sequence_length": 50,
        },
    }
    
    model_config = default_configs.get(model_type, {})
    if config:
        model_config.update(config)
    
    return model_config


# ==============================================================================
# AUTO-TRAINING TASKS
# ==============================================================================

@shared_task(name='ml.training.tasks.auto_training_check')
def auto_training_check():
    """
    Periodic task to check if any models need retraining.
    
    This task should be added to Celery beat schedule to run hourly:
    
    CELERY_BEAT_SCHEDULE = {
        'ml-auto-training-check': {
            'task': 'ml.training.tasks.auto_training_check',
            'schedule': 3600,  # Every hour
        },
    }
    """
    try:
        from .auto_training import AutoTrainingManager
        
        manager = AutoTrainingManager()
        result = manager.run_auto_training_check()
        
        logger.info(f"Auto-training check result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Auto-training check failed: {e}")
        return {"status": "error", "error": str(e)}


@shared_task(name='ml.training.tasks.process_training_queues')
def process_training_queues():
    """
    Process any pending training jobs in the queue.
    
    This handles training jobs that were queued but not yet processed.
    """
    try:
        import redis
        from django.conf import settings
        import json
        
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        
        # Check for pending training jobs
        pending_key = 'ml:training_queue'
        jobs_processed = 0
        
        while True:
            job_data = r.lpop(pending_key)
            if not job_data:
                break
            
            try:
                job = json.loads(job_data)
                model_type = job.get('model_type', 'ncf')
                config = job.get('config', {})
                
                # Trigger training
                train_recommendation_model.delay(model_type=model_type, config=config)
                jobs_processed += 1
                
            except json.JSONDecodeError:
                logger.warning(f"Invalid job data in queue: {job_data}")
                continue
        
        logger.info(f"Processed {jobs_processed} training jobs from queue")
        return {"status": "success", "jobs_processed": jobs_processed}
        
    except Exception as e:
        logger.error(f"Failed to process training queues: {e}")
        return {"status": "error", "error": str(e)}


@shared_task(name='ml.training.tasks.cleanup_old_data')
def cleanup_old_data():
    """
    Clean up old training data and model versions.
    
    This task should run weekly to prevent data accumulation.
    """
    try:
        import redis
        from django.conf import settings
        from datetime import timedelta
        
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        
        # Clean old interactions (older than 90 days)
        cutoff = timezone.now() - timedelta(days=90)
        cutoff_ts = cutoff.timestamp()
        
        # Clean interaction logs
        r.zremrangebyscore('ml:interactions_sorted', '-inf', cutoff_ts)
        
        # Trim training log
        r.ltrim('ml:training_log', 0, 999)
        
        # Clean old model versions (keep last 5)
        from ..core.registry import ModelRegistry
        registry = ModelRegistry()
        
        model_names = [
            'ncf', 'deepfm', 'two_tower', 'sequence',
            'embeddings', 'demand', 'fraud', 'churn'
        ]
        
        for model_name in model_names:
            try:
                registry.cleanup_old_versions(model_name, keep=5)
            except Exception as e:
                logger.warning(f"Failed to cleanup versions for {model_name}: {e}")
        
        logger.info("Old data cleanup completed")
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        return {"status": "error", "error": str(e)}


def get_celery_beat_schedule():
    """
    Get Celery beat schedule for ML training tasks.
    
    Usage in Django settings:
    
        from ml.training.tasks import get_celery_beat_schedule
        CELERY_BEAT_SCHEDULE.update(get_celery_beat_schedule())
    """
    from celery.schedules import crontab
    
    return {
        'ml-auto-training-check': {
            'task': 'ml.training.tasks.auto_training_check',
            'schedule': 3600,  # Every hour
        },
        'ml-process-training-queues': {
            'task': 'ml.training.tasks.process_training_queues',
            'schedule': 300,  # Every 5 minutes
        },
        'ml-model-health-check': {
            'task': 'ml.training.tasks.model_health_check',
            'schedule': 3600,  # Every hour
        },
        'ml-scheduled-training': {
            'task': 'ml.training.tasks.scheduled_model_training',
            'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
        },
        'ml-data-cleanup': {
            'task': 'ml.training.tasks.cleanup_old_data',
            'schedule': crontab(day_of_week=0, hour=3, minute=0),  # Sunday 3 AM
        },
    }
