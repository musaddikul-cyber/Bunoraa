"""
ML Models Configuration

Production-ready configuration for the ML system.
Add these settings to your Django settings file.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================================
# ML System Settings
# ================================

# Enable/disable ML features
ML_ENABLED = True

# Production mode - enables auto-training
# Set this to True in production settings
# PRODUCTION = True

# ================================
# Auto-Training Configuration
# ================================

# Enable automatic model retraining
ML_AUTO_TRAINING = True

# Auto-retrain when drift is detected
ML_AUTO_RETRAIN_ON_DRIFT = True

# Detailed auto-training configuration
ML_AUTO_TRAINING_CONFIG = {
    # Enable/disable auto-training
    'enabled': True,
    
    # Minimum data requirements before training
    'min_interactions': 1000,
    'min_users': 100,
    'min_products': 50,
    
    # Trigger training when new data exceeds these thresholds
    'new_interactions_threshold': 5000,
    'new_users_threshold': 500,
    'new_products_threshold': 100,
    
    # Time-based training
    'max_days_between_training': 7,
    'training_hour': 2,  # 2 AM local time
    
    # Performance thresholds
    'min_performance_score': 0.6,
    'max_performance_drop': 0.1,
    
    # Drift detection
    'enable_drift_detection': True,
    'drift_threshold': 0.15,
    
    # Resource management
    'max_concurrent_trainings': 2,
    'training_timeout_minutes': 120,
    
    # Specific models to include/exclude
    'models_to_train': [],  # Empty = all models
    'models_to_exclude': [],  # Models to skip
}

# ================================
# Model Training Configuration
# ================================

ML_TRAINING_CONFIG = {
    # Default training parameters
    'default': {
        'epochs': 50,
        'batch_size': 256,
        'learning_rate': 0.001,
        'early_stopping_patience': 5,
        'validation_split': 0.1,
    },
    
    # Model-specific overrides
    'ncf': {
        'epochs': 100,
        'batch_size': 512,
        'embedding_dim': 64,
        'layers': [128, 64, 32],
    },
    'deepfm': {
        'epochs': 80,
        'batch_size': 512,
        'embedding_dim': 32,
    },
    'two_tower': {
        'epochs': 60,
        'batch_size': 1024,
        'embedding_dim': 128,
    },
    'sequence': {
        'epochs': 50,
        'batch_size': 128,
        'sequence_length': 20,
        'hidden_dim': 256,
    },
    'demand': {
        'epochs': 100,
        'batch_size': 64,
        'hidden_dim': 128,
    },
    'fraud': {
        'epochs': 100,
        'batch_size': 128,
        'class_weight': 'balanced',
    },
    'churn': {
        'epochs': 80,
        'batch_size': 256,
    },
}

# ================================
# Redis Configuration
# ================================

# Redis URL for ML data storage
# REDIS_URL = 'redis://localhost:6379/0'

# ML-specific Redis settings
ML_REDIS_CONFIG = {
    # Maximum queue sizes
    'max_interactions_queue': 100000,
    'max_events_queue': 100000,
    
    # Cache TTLs
    'user_profile_cache_ttl': 3600,  # 1 hour
    'product_features_cache_ttl': 3600,
    'recommendations_cache_ttl': 300,  # 5 minutes
    
    # Session tracking
    'session_ttl': 1800,  # 30 minutes
}

# ================================
# Data Collection Configuration
# ================================

ML_DATA_COLLECTION = {
    # Enable/disable data collection
    'enabled': True,
    
    # Batch processing
    'batch_size': 100,
    'flush_interval': 300,  # 5 minutes
    
    # Data retention
    'retention_days': 90,
    
    # Privacy settings
    'anonymize_ip': False,  # Set to True for GDPR compliance
    'collect_location': True,
    'collect_device_info': True,
}

# ================================
# Tracking Middleware Configuration
# ================================

ML_TRACKING = {
    # Enable page view tracking
    'track_page_views': True,
    
    # Enable product view tracking
    'track_product_views': True,
    
    # Paths to exclude from tracking
    'exclude_paths': [
        '/admin/',
        '/api/ml/',
        '/static/',
        '/media/',
        '/__debug__/',
    ],
    
    # Bot user agents to exclude
    'exclude_bots': True,
}

# ================================
# Model Storage Configuration
# ================================

ML_MODEL_STORAGE = {
    # Directory for saved models
    'model_dir': os.path.join(BASE_DIR, 'ml', 'saved_models'),
    
    # Directory for training data
    'training_data_dir': os.path.join(BASE_DIR, 'ml', 'training_data'),
    
    # Keep N versions of each model
    'max_model_versions': 5,
    
    # Model file format
    'model_format': 'pt',  # PyTorch format
}

# ================================
# Celery Beat Schedule Integration
# ================================

def get_celery_beat_schedule():
    """
    Get Celery beat schedule for ML tasks.
    
    Add this to your CELERY_BEAT_SCHEDULE in settings:
    
        from ml.core.settings import get_celery_beat_schedule
        CELERY_BEAT_SCHEDULE.update(get_celery_beat_schedule())
    """
    return {
        # Auto-training check - runs every hour
        'ml-auto-training-check': {
            'task': 'ml.training.tasks.auto_training_check',
            'schedule': 3600,  # Every hour
        },
        # Process training queues - runs every 5 minutes
        'ml-process-training-queues': {
            'task': 'ml.training.tasks.process_training_queues',
            'schedule': 300,  # Every 5 minutes
        },
        # Model health check - runs every hour
        'ml-model-health-check': {
            'task': 'ml.training.tasks.model_health_check',
            'schedule': 3600,  # Every hour
        },
        # Scheduled model training - runs daily at 2 AM
        'ml-scheduled-training': {
            'task': 'ml.training.tasks.scheduled_model_training',
            'schedule': {
                'hour': 2,
                'minute': 0,
            },
        },
        # Data cleanup - runs weekly
        'ml-data-cleanup': {
            'task': 'ml.training.tasks.cleanup_old_data',
            'schedule': 604800,  # Weekly
        },
    }

# ================================
# Django Settings Integration
# ================================

# Add to MIDDLEWARE in settings.py:
#
# MIDDLEWARE = [
#     ...
#     'ml.middleware.MLTrackingMiddleware',
#     'ml.middleware.MLProductTrackingMiddleware',
#     ...
# ]

# Add to urls.py:
#
# urlpatterns = [
#     ...
#     path('api/ml/', include('ml.api.urls')),
#     ...
# ]

# Add to INSTALLED_APPS:
#
# INSTALLED_APPS = [
#     ...
#     'ml',
#     ...
# ]

# ================================
# Logging Configuration
# ================================

ML_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'ml_verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'ml_file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'ml.log'),
            'maxBytes': 10 * 1024 * 1024,  # 10 MB
            'backupCount': 5,
            'formatter': 'ml_verbose',
        },
        'ml_console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'ml_verbose',
        },
    },
    'loggers': {
        'bunoraa.ml': {
            'handlers': ['ml_file', 'ml_console'],
            'level': 'INFO',
            'propagate': True,
        },
        'bunoraa.ml.training': {
            'handlers': ['ml_file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'bunoraa.ml.data_collection': {
            'handlers': ['ml_file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}


def get_default_settings():
    """
    Get all default ML settings as a dictionary.
    
    Use this to easily merge ML settings into Django settings:
    
        from ml.core.settings import get_default_settings
        ml_settings = get_default_settings()
        globals().update(ml_settings)
    """
    return {
        'ML_ENABLED': ML_ENABLED,
        'ML_AUTO_TRAINING': ML_AUTO_TRAINING,
        'ML_AUTO_RETRAIN_ON_DRIFT': ML_AUTO_RETRAIN_ON_DRIFT,
        'ML_AUTO_TRAINING_CONFIG': ML_AUTO_TRAINING_CONFIG,
        'ML_TRAINING_CONFIG': ML_TRAINING_CONFIG,
        'ML_REDIS_CONFIG': ML_REDIS_CONFIG,
        'ML_DATA_COLLECTION': ML_DATA_COLLECTION,
        'ML_TRACKING': ML_TRACKING,
        'ML_MODEL_STORAGE': ML_MODEL_STORAGE,
    }
