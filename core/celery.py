"""
Celery Configuration for Bunoraa
Production-ready task queue with scheduled backups and maintenance.
"""
import os
from celery import Celery
from celery.schedules import crontab
from django.conf import settings

# Set default Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

app = Celery('bunoraa')

# Load config from Django settings
app.config_from_object('django.conf:settings', namespace='CELERY')

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
        'task': 'apps.products.tasks.check_low_stock_alerts',
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
        'task': 'ml_models.training.tasks.train_recommendation_model',
        'schedule': crontab(hour=2, minute=0, day_of_week=0),
        'args': ('ncf',),
        'options': {'queue': 'ml_training'},
    },
    
    # Train embedding models weekly on Sunday at 3 AM
    'train-embedding-models': {
        'task': 'ml_models.training.tasks.train_embedding_models',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),
        'options': {'queue': 'ml_training'},
    },
    
    # Train demand forecaster weekly on Monday at 1 AM
    'train-demand-forecaster': {
        'task': 'ml_models.training.tasks.train_demand_forecaster',
        'schedule': crontab(hour=1, minute=0, day_of_week=1),
        'options': {'queue': 'ml_training'},
    },
    
    # Train fraud detector daily at 4 AM
    'train-fraud-detector': {
        'task': 'ml_models.training.tasks.train_fraud_detector',
        'schedule': crontab(hour=4, minute=0),
        'options': {'queue': 'ml_training'},
    },
    
    # Train churn predictor weekly on Tuesday at 1 AM - DISABLED
    # 'train-churn-predictor': {
    #     'task': 'ml_models.training.tasks.train_churn_predictor',
    #     'schedule': crontab(hour=1, minute=0, day_of_week=2),
    #     'options': {'queue': 'ml_training'},
    # },
    
    # Train search model weekly on Wednesday at 1 AM - DISABLED
    # 'train-search-model': {
    #     'task': 'ml_models.training.tasks.train_search_model',
    #     'schedule': crontab(hour=1, minute=0, day_of_week=3),
    #     'options': {'queue': 'ml_training'},
    # },
    
    # ML model health check hourly - DISABLED
    # 'ml-health-check': {
    #     'task': 'ml_models.training.tasks.model_health_check',
    #     'schedule': crontab(minute=0),
    # },
    
    # Batch inference for recommendations every 6 hours - DISABLED
    # 'batch-recommendations': {
    #     'task': 'ml_models.training.tasks.batch_generate_recommendations',
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
    
    # Send weekly digest on Monday at 9 AM
    'send-weekly-digest': {
        'task': 'apps.notifications.tasks.process_weekly_digest',
        'schedule': crontab(hour=9, minute=0, day_of_week=1),
    },
}

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
    'apps.products.tasks.train_*': {'queue': 'ml'},
}

# =============================================================================
# Task settings - MEMORY OPTIMIZED
app.conf.task_time_limit = 600  # 10 minutes hard limit
app.conf.task_soft_time_limit = 540  # 9 minutes soft limit
app.conf.task_acks_late = True  # Acknowledge after task completes
app.conf.task_reject_on_worker_lost = True  # Reject if worker dies

# Result backend settings
app.conf.result_expires = 3600  # Results expire after 1 hour

# Retry settings
app.conf.task_default_retry_delay = 60  # 1 minute
app.conf.task_max_retries = 3

# Worker settings - MEMORY OPTIMIZED for 512MB limit
app.conf.worker_prefetch_multiplier = 1  # Load one task at a time (4)
app.conf.worker_max_tasks_per_child = 500  # Restart worker after 500 (1000) tasks to free memory


@app.task(bind=True)
def debug_task(self):
    """Debug task for testing."""
    print(f'Request: {self.request!r}')
