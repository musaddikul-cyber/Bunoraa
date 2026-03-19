"""
Celery Configuration for Bunoraa
Production-ready task queue with scheduled backups and maintenance.
"""
import os
import logging
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from celery import Celery
from celery.schedules import crontab
from django.conf import settings

# Set default Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')


def _normalize_rediss_url(url: str) -> str:
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.scheme != 'rediss':
        return url
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if 'ssl_cert_reqs' not in params:
        params['ssl_cert_reqs'] = os.environ.get('CELERY_REDIS_SSL_CERT_REQS', 'required')
        return urlunparse(parsed._replace(query=urlencode(params)))
    return url


_settings_module = os.environ.get('DJANGO_SETTINGS_MODULE', '')
if _settings_module.endswith('.local'):
    # Local eager mode should not depend on external Redis availability.
    os.environ['CELERY_BROKER_URL'] = 'memory://'
    os.environ['CELERY_RESULT_BACKEND'] = 'cache+memory://'
else:
    for _env_key in ('CELERY_BROKER_URL', 'CELERY_RESULT_BACKEND', 'BROKER_URL', 'RESULT_BACKEND'):
        _env_value = os.environ.get(_env_key)
        if _env_value:
            os.environ[_env_key] = _normalize_rediss_url(_env_value)

app = Celery('bunoraa')
logger = logging.getLogger('bunoraa.celery')

# Load config from Django settings
app.config_from_object('django.conf:settings', namespace='CELERY')
_broker_url = getattr(settings, 'CELERY_BROKER_URL', app.conf.broker_url)
_result_backend = getattr(settings, 'CELERY_RESULT_BACKEND', app.conf.result_backend)
app.conf.update(
    broker_url=_normalize_rediss_url(_broker_url) if _broker_url else _broker_url,
    result_backend=_normalize_rediss_url(_result_backend) if _result_backend else _result_backend,
    task_always_eager=getattr(settings, 'CELERY_TASK_ALWAYS_EAGER', app.conf.task_always_eager),
    task_eager_propagates=getattr(settings, 'CELERY_TASK_EAGER_PROPAGATES', app.conf.task_eager_propagates),
    task_ignore_result=getattr(settings, 'CELERY_TASK_IGNORE_RESULT', app.conf.task_ignore_result),
    task_store_eager_result=getattr(settings, 'CELERY_TASK_STORE_EAGER_RESULT', app.conf.task_store_eager_result),
)

# Auto-discover tasks from all installed apps
app.autodiscover_tasks()

# =============================================================================
# CELERY BEAT SCHEDULE - Automated Tasks
# All times are in Bangladesh timezone (Asia/Dhaka)
# OPTIMIZED FOR RENDER FREE TIER: Disabled memory-intensive tasks
# =============================================================================
app.conf.beat_schedule = {
    # ==========================================================================
    # BACKUP TASKS - DISABLED on Render free tier (memory intensive)
    # ==========================================================================
    
    # Daily database backup at 3 AM Bangladesh time
    # 'daily-database-backup': {
    #     'task': 'core.tasks.backup_database_to_r2',
    #     'schedule': crontab(hour=3, minute=0),
    #     'options': {'queue': 'backups'},
    # },
    
    # Weekly full media backup on Sunday at 4 AM
    # 'weekly-media-backup': {
    #     'task': 'core.tasks.backup_media_to_r2',
    #     'schedule': crontab(hour=4, minute=0, day_of_week=0),
    #     'options': {'queue': 'backups'},
    # },
    
    # Daily incremental media sync at 2 AM
    # 'daily-media-sync': {
    #     'task': 'core.tasks.sync_media_incremental',
    #     'schedule': crontab(hour=2, minute=0),
    #     'options': {'queue': 'backups'},
    # },
    
    # ==========================================================================
    # DATA MAINTENANCE TASKS
    # ==========================================================================
    
    # Update exchange rates every 6 hours
    'update-exchange-rates': {
        'task': 'core.tasks.update_exchange_rates',
        'schedule': crontab(hour='*/6', minute=0),
    },
    
    # Aggregate daily analytics at 12:30 AM
    'aggregate-daily-analytics': {
        'task': 'core.tasks.aggregate_daily_analytics',
        'schedule': crontab(hour=0, minute=30),
    },
    
    # Update user behavior profiles every 4 hours - DISABLED (moved to batch processing)
    # 'update-behavior-profiles': {
    #     'task': 'core.tasks.update_user_behavior_profiles',
    #     'schedule': crontab(hour='*/4', minute=15),
    # },
    
    # Clean expired sessions daily at 2 AM
    'cleanup-expired-sessions': {
        'task': 'core.tasks.cleanup_expired_sessions',
        'schedule': crontab(hour=2, minute=0),
    },
    
    # ==========================================================================
    # PERFORMANCE TASKS - DISABLED (cache warming too frequent)
    # ==========================================================================
    
    # Warm cache every 30 minutes - DISABLED on Render free tier
    # 'warm-cache': {
    #     'task': 'core.tasks.warm_cache',
    #     'schedule': crontab(minute='*/30'),
    # },
    
    # ==========================================================================
    # MARKETING & ENGAGEMENT TASKS
    # ==========================================================================
    
    # Send abandoned cart reminders (every 2 hours during business hours 9AM-9PM)
    # 'abandoned-cart-reminders': {
    #     'task': 'apps.cart.tasks.send_abandoned_cart_reminders',  # Deprecated - moved to apps.commerce
    #     'schedule': crontab(hour='9-21/2', minute=0),
    # },
    
    # ==========================================================================
    # REPORTING TASKS
    # ==========================================================================
    
    # Generate daily reports at 6 AM
    'generate-daily-reports': {
        'task': 'apps.analytics.tasks.generate_daily_report',
        'schedule': crontab(hour=6, minute=0),
    },
    
    # Generate weekly reports on Monday at 7 AM
    'generate-weekly-reports': {
        'task': 'apps.analytics.tasks.generate_weekly_report',
        'schedule': crontab(hour=7, minute=0, day_of_week=1),
    },
    
    # ==========================================================================
    # INVENTORY TASKS
    # ==========================================================================
    
    # Check low stock alerts every hour
    'check-low-stock': {
        'task': 'catalog.check_low_stock',
        'schedule': crontab(minute=0),
    },
    
    # ==========================================================================
    # ML MODEL TASKS
    # ==========================================================================
    
    # Update ML recommendations daily at 5 AM
    'update-ml-recommendations': {
        'task': 'core.tasks.update_ml_models',
        'schedule': crontab(hour=5, minute=0),
    },
    
    # Train recommendation models weekly on Sunday at 2 AM
    'train-recommendation-models': {
        'task': 'ml.training.tasks.train_recommendation_model',
        'schedule': crontab(hour=2, minute=0, day_of_week=0),
        'args': ('ncf',),
        'options': {'queue': 'ml_training'},
    },
    
    # Train embedding models weekly on Sunday at 3 AM
    'train-embedding-models': {
        'task': 'ml.training.tasks.train_embedding_models',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),
        'options': {'queue': 'ml_training'},
    },
    
    # Train demand forecaster weekly on Monday at 1 AM
    'train-demand-forecaster': {
        'task': 'ml.training.tasks.train_demand_forecaster',
        'schedule': crontab(hour=1, minute=0, day_of_week=1),
        'options': {'queue': 'ml_training'},
    },
    
    # Train fraud detector daily at 4 AM
    'train-fraud-detector': {
        'task': 'ml.training.tasks.train_fraud_detector',
        'schedule': crontab(hour=4, minute=0),
        'options': {'queue': 'ml_training'},
    },
    
    # Train churn predictor weekly on Tuesday at 1 AM - DISABLED
    # 'train-churn-predictor': {
    #     'task': 'ml.training.tasks.train_churn_predictor',
    #     'schedule': crontab(hour=1, minute=0, day_of_week=2),
    #     'options': {'queue': 'ml_training'},
    # },
    
    # Train search model weekly on Wednesday at 1 AM - DISABLED
    # 'train-search-model': {
    #     'task': 'ml.training.tasks.train_search_model',
    #     'schedule': crontab(hour=1, minute=0, day_of_week=3),
    #     'options': {'queue': 'ml_training'},
    # },
    
    # ML model health check hourly - DISABLED
    # 'ml-health-check': {
    #     'task': 'ml.training.tasks.model_health_check',
    #     'schedule': crontab(minute=0),
    # },
    
    # Batch inference for recommendations every 6 hours - DISABLED
    # 'batch-recommendations': {
    #     'task': 'ml.training.tasks.batch_generate_recommendations',
    #     'schedule': crontab(hour='*/6', minute=30),
    #     'options': {'queue': 'ml_inference'},
    # },
    
    # ==========================================================================
    # CLEANUP TASKS
    # ==========================================================================
    
    # Clean old analytics data monthly (keep 1 year)
    'cleanup-old-analytics': {
        'task': 'apps.analytics.tasks.cleanup_old_data',
        'schedule': crontab(hour=4, minute=0, day_of_month=1),
        'kwargs': {'days': 365},
    },

    # Cleanup expired product AI autofill evidence (default retention: 365 days)
    'cleanup-product-autofill-evidence': {
        'task': 'catalog.cleanup_product_autofill_evidence',
        'schedule': crontab(hour=4, minute=45, day_of_month=1),
    },
    
    # Clean old user interactions monthly (keep 2 years)
    'cleanup-old-interactions': {
        'task': 'apps.accounts.tasks.cleanup_old_interactions',
        'schedule': crontab(hour=4, minute=30, day_of_month=1),
        'kwargs': {'days': 730},
    },
    
    # Clean expired tokens weekly
    'cleanup-expired-tokens': {
        'task': 'apps.accounts.tasks.cleanup_expired_tokens',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),
    },

    # Cleanup expired data exports daily at 3:30 AM
    'cleanup-expired-exports': {
        'task': 'apps.accounts.tasks.cleanup_expired_exports',
        'schedule': crontab(hour=3, minute=30),
    },

    # Cleanup old auth sessions daily at 3:15 AM
    'cleanup-old-auth-sessions': {
        'task': 'apps.accounts.tasks.cleanup_old_auth_sessions',
        'schedule': crontab(hour=3, minute=15),
        'kwargs': {'days': getattr(settings, 'AUTH_SESSION_RETENTION_DAYS', 90)},
    },

    # Process account deletion requests daily at 4 AM
    'process-account-deletions': {
        'task': 'apps.accounts.tasks.process_account_deletions',
        'schedule': crontab(hour=4, minute=0),
    },
    
    # ==========================================================================
    # LIVE CHAT TASKS
    # ==========================================================================
    
    # Update chat analytics every hour - REDUCED frequency
    'update-chat-analytics': {
        'task': 'apps.chat.tasks.update_daily_analytics',
        'schedule': crontab(minute=5, hour='*/3'),  # Every 3 hours instead of every hour
    },
    
    # Cleanup stale typing indicators - DISABLED (runs too frequently)
    # 'cleanup-typing-indicators': {
    #     'task': 'apps.chat.tasks.cleanup_old_typing_indicators',
    #     'schedule': crontab(minute='*/5'),
    # },
    
    # Auto-resolve inactive conversations daily at 1 AM
    'auto-resolve-inactive-chats': {
        'task': 'apps.chat.tasks.auto_resolve_inactive_conversations',
        'schedule': crontab(hour=1, minute=0),
        'kwargs': {'hours': 24},
    },
    
    # Notify waiting customers - REDUCED frequency for memory
    # 'notify-waiting-customers': {
    #     'task': 'apps.chat.tasks.notify_waiting_customers',
    #     'schedule': crontab(minute='*/10', hour='9-21'),
    # },
    
    # Update agent online status - DISABLED (runs too frequently)
    # 'update-agent-status': {
    #     'task': 'apps.chat.tasks.update_agent_online_status',
    #     'schedule': crontab(minute='*/5'),
    # },
    
    # Sync agent metrics hourly - DISABLED
    # 'sync-agent-metrics': {
    #     'task': 'apps.chat.tasks.sync_agent_metrics',
    #     'schedule': crontab(minute=30),  # 30 minutes past every hour
    # },
    
    # ==========================================================================
    # NOTIFICATION DIGEST TASKS
    # ==========================================================================
    
    # Send daily digest at 9 AM
    'send-daily-digest': {
        'task': 'apps.notifications.tasks.process_daily_digest',
        'schedule': crontab(hour=9, minute=0),
    },

    # Send hourly digest at the top of every hour
    'send-hourly-digest': {
        'task': 'apps.notifications.tasks.process_hourly_digest',
        'schedule': crontab(minute=0),
    },
    
    # Send weekly digest on Monday at 9 AM
    'send-weekly-digest': {
        'task': 'apps.notifications.tasks.process_weekly_digest',
        'schedule': crontab(hour=9, minute=0, day_of_week=1),
    },
}

# -----------------------------------------------------------------------------
# ML schedule controls
# -----------------------------------------------------------------------------
_legacy_ml_keys = (
    'update-ml-recommendations',
    'train-recommendation-models',
    'train-embedding-models',
    'train-demand-forecaster',
    'train-fraud-detector',
)
for _ml_key in _legacy_ml_keys:
    app.conf.beat_schedule.pop(_ml_key, None)

if getattr(settings, 'ML_ENABLED', False) and getattr(settings, 'ML_CELERY_BEAT_ENABLED', False):
    _ml_update_every = max(1, int(getattr(settings, 'ML_MODEL_UPDATE_INTERVAL', 24) or 24))
    app.conf.beat_schedule['update-ml-recommendations'] = {
        'task': 'core.tasks.update_ml_models',
        'schedule': crontab(hour=f'*/{_ml_update_every}', minute=0),
    }

    if getattr(settings, 'ML_AUTO_TRAINING', False):
        try:
            from ml.tasks import get_celery_beat_schedule as get_ml_celery_beat_schedule
            app.conf.beat_schedule.update(get_ml_celery_beat_schedule())
        except Exception as exc:
            logger.warning("Failed to load ML auto-training beat schedule: %s", exc)

# =============================================================================
# TASK ROUTING
# =============================================================================
app.conf.task_routes = {
    # Backup tasks to dedicated queue
    'core.tasks.backup_*': {'queue': 'backups'},
    'core.tasks.sync_*': {'queue': 'backups'},
    
    # Payment tasks to high-priority queue
    'apps.payments.tasks.*': {'queue': 'payments'},
    
    # Notification tasks
    'apps.notifications.tasks.*': {'queue': 'notifications'},
    
    # Chat tasks
    'apps.chat.tasks.*': {'queue': 'chat'},
    'apps.chat.tasks.generate_ai_response': {'queue': 'chat_ai'},
    
    # Analytics tasks (can be slower)
    'apps.analytics.tasks.*': {'queue': 'analytics'},
    
    # ML tasks (resource intensive)
    'core.tasks.update_ml_*': {'queue': 'ml'},
    'ml.training.tasks.*': {'queue': 'ml'},
    'catalog.run_product_autofill_job': {'queue': 'catalog_ai'},
    'catalog.cleanup_product_autofill_evidence': {'queue': 'catalog_ai'},
}

# =============================================================================
# Task settings - MEMORY OPTIMIZED
app.conf.broker_connection_retry_on_startup = getattr(
    settings, 'CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP', True
)
app.conf.broker_connection_max_retries = getattr(
    settings, 'CELERY_BROKER_CONNECTION_MAX_RETRIES', 100
)

app.conf.task_time_limit = getattr(settings, 'CELERY_TASK_TIME_LIMIT', 600)
app.conf.task_soft_time_limit = getattr(settings, 'CELERY_TASK_SOFT_TIME_LIMIT', 540)
app.conf.task_acks_late = getattr(settings, 'CELERY_TASK_ACKS_LATE', True)
app.conf.task_reject_on_worker_lost = getattr(settings, 'CELERY_TASK_REJECT_ON_WORKER_LOST', True)
app.conf.task_acks_on_failure_or_timeout = getattr(
    settings, 'CELERY_TASK_ACKS_ON_FAILURE_OR_TIMEOUT', True
)

# Result backend settings
app.conf.result_expires = getattr(settings, 'CELERY_RESULT_EXPIRES', 3600)

# Retry settings
app.conf.task_default_retry_delay = getattr(settings, 'CELERY_TASK_DEFAULT_RETRY_DELAY', 60)
app.conf.task_max_retries = getattr(settings, 'CELERY_TASK_MAX_RETRIES', 3)

# Worker settings - MEMORY OPTIMIZED for 512MB limit
app.conf.worker_prefetch_multiplier = getattr(settings, 'CELERY_WORKER_PREFETCH_MULTIPLIER', 1)
app.conf.worker_max_tasks_per_child = getattr(settings, 'CELERY_WORKER_MAX_TASKS_PER_CHILD', 500)


@app.task(bind=True)
def debug_task(self):
    """Debug task for testing."""
    print(f'Request: {self.request!r}')
