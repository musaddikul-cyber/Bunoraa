"""
Django Caching Configuration - Multi-Redis Setup
CELERY_REDIS_URL (1GB)  -> Default cache, API cache, channels, Celery
ML_REDIS_URL (200MB)    -> ML cache, heavy computation, analytics  
REDIS_URL (25MB)        -> Sessions, real-time features, templates
"""
import os
from urllib.parse import urlparse

# =============================================================================
# MULTI-REDIS URL CONFIGURATION
# Derived from .env variables
# =============================================================================

# Celery Redis (1GB Aiven) - General caching, Celery, Channels
CELERY_REDIS_URL = os.environ.get('CELERY_REDIS_URL', 'redis://localhost:6379/0')

# ML Redis (200MB Upstash) - ML workloads, heavy computation
ML_REDIS_URL = os.environ.get('ML_REDIS_URL', CELERY_REDIS_URL)

# Render Redis (25MB) - Sessions, real-time, templates
REDIS_URL = os.environ.get('REDIS_URL', CELERY_REDIS_URL)

# Cache timeouts (in seconds)
CACHE_TIMEOUTS = {
    'default': 300,           # 5 minutes
    'short': 60,              # 1 minute
    'medium': 900,            # 15 minutes
    'long': 3600,             # 1 hour
    'very_long': 86400,       # 24 hours
    'session': 86400 * 7,     # 7 days
}

# =============================================================================
# CACHE CONFIGURATION - MULTI-REDIS SETUP
# =============================================================================

CACHES = {
    # Default cache: Celery Redis (1GB Aiven) - General purpose
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': CELERY_REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
            'SERIALIZER': 'django_redis.serializers.json.JSONSerializer',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,  # Reduced for Redis cloud limits
                'retry_on_timeout': True,
            },
            'SOCKET_CONNECT_TIMEOUT': 5,
            'SOCKET_TIMEOUT': 5,
        },
        'KEY_PREFIX': 'bunoraa',
        'VERSION': 1,
        'TIMEOUT': CACHE_TIMEOUTS['default'],
    },
    
    # Template fragments: Render Redis (25MB) - Fast, simple caching
    'template_fragments': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 20,  # Small for Render
            },
        },
        'KEY_PREFIX': 'bunoraa:template',
        'TIMEOUT': CACHE_TIMEOUTS['medium'],
    },
    
    # API cache: Celery Redis (1GB Aiven) - API response caching
    'api_cache': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': CELERY_REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 30,
            },
        },
        'KEY_PREFIX': 'bunoraa:api',
        'TIMEOUT': CACHE_TIMEOUTS['short'],
    },
    
    # Computation/ML cache: ML Redis (1GB) - Heavy ML and analytics workloads
    'computation': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': ML_REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 100,  # Higher for ML workloads
                'retry_on_timeout': True,
            },
            'SOCKET_CONNECT_TIMEOUT': 10,
            'SOCKET_TIMEOUT': 30,  # Longer for ML operations
        },
        'KEY_PREFIX': 'bunoraa:compute',
        'TIMEOUT': CACHE_TIMEOUTS['long'],
    },
    
    # ML model cache: ML Redis (1GB) - Large model storage
    'ml_models': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': ML_REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
            },
            'SOCKET_TIMEOUT': 60,  # Long timeout for model loading
        },
        'KEY_PREFIX': 'bunoraa:ml',
        'TIMEOUT': 86400 * 30,  # 30 days for models
    },
    
    # Real-time/Render cache: Render Redis (25MB) - Quick lookups, sessions
    'realtime': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 20,
            },
        },
        'KEY_PREFIX': 'bunoraa:rt',
        'TIMEOUT': CACHE_TIMEOUTS['short'],
    },
}

# =============================================================================
# SESSION CONFIGURATION
# =============================================================================

# Use Simple Redis (25MB) for session storage - efficient for simple key-value
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'realtime'
SESSION_COOKIE_AGE = 86400 * 7  # 7 days
SESSION_COOKIE_SECURE = True  # Only send over HTTPS
SESSION_COOKIE_HTTPONLY = True  # Prevent JavaScript access
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_SAVE_EVERY_REQUEST = True

# =============================================================================
# CHANNEL LAYERS (WebSockets) - Use Celery Redis for pub/sub
# =============================================================================

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [CELERY_REDIS_URL],
            'capacity': 1500,
            'expiry': 10,
        },
    },
}

# =============================================================================
# CACHE MIDDLEWARE CONFIGURATION
# =============================================================================

# Per-site cache middleware settings
CACHE_MIDDLEWARE_ALIAS = 'default'
CACHE_MIDDLEWARE_SECONDS = 300  # 5 minutes default
CACHE_MIDDLEWARE_KEY_PREFIX = 'bunoraa_page'
CACHE_MIDDLEWARE_ANONYMOUS_ONLY = True

# =============================================================================
# DJANGO REDIS SETTINGS
# =============================================================================

# Ignore Redis connection errors (fail gracefully)
DJANGO_REDIS_IGNORE_EXCEPTIONS = True
DJANGO_REDIS_LOG_IGNORED_EXCEPTIONS = True
DJANGO_REDIS_LOGGER = 'django.cache'

# =============================================================================
# SELECT RELATED / PREFETCH RELATED OPTIMIZATION
# =============================================================================

# Default select_related fields for common models to reduce N+1 queries
SELECT_RELATED_DEFAULTS = {
    'catalog.Product': ['primary_category', 'brand', 'unit'],
    'catalog.ProductVariant': ['product', 'attribute_values'],
    'preorders.Order': ['user', 'shipping_address', 'billing_address'],
    'preorders.OrderItem': ['product', 'variant'],
    'accounts.User': ['profile'],
}

# Default prefetch_related fields
PREFETCH_RELATED_DEFAULTS = {
    'catalog.Product': ['images', 'categories', 'tags', 'attribute_values'],
    'catalog.Category': ['children'],
}

# =============================================================================
# DATABASE QUERY OPTIMIZATION
# =============================================================================

# Database connection pooling
DATABASE_CONNECTION_POOL_SIZE = int(os.environ.get('DB_POOL_SIZE', 10))
DATABASE_CONNECTION_MAX_OVERFLOW = int(os.environ.get('DB_MAX_OVERFLOW', 20))
DATABASE_CONNECTION_POOL_TIMEOUT = int(os.environ.get('DB_POOL_TIMEOUT', 30))
DATABASE_CONNECTION_POOL_RECYCLE = int(os.environ.get('DB_POOL_RECYCLE', 3600))

# Query timeout (prevents long-running queries)
DATABASE_STATEMENT_TIMEOUT = int(os.environ.get('DB_STATEMENT_TIMEOUT', 30000))  # 30 seconds

# =============================================================================
# STATIC FILES & MEDIA OPTIMIZATION
# =============================================================================

# Static files storage with caching
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.ManifestStaticFilesStorage'

# Cache control headers for static files
STATICFILES_CACHE_CONTROL = 'public, max-age=31536000, immutable'

# =============================================================================
# THROTTLING & RATE LIMITING
# =============================================================================

# Default throttle rates
DEFAULT_THROTTLE_RATES = {
    'anon': '100/minute',
    'user': '1000/minute',
    'burst': '10/second',
}

# API endpoint specific throttling
API_THROTTLE_RATES = {
    'catalog': '200/minute',
    'cart': '50/minute',
    'checkout': '20/minute',
    'auth': '10/minute',
    'admin': '500/minute',
}
