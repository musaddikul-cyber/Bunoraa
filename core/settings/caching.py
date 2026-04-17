"""
Django Caching Configuration - Production-Ready
Includes Redis caching, session caching, and cache middleware setup.
"""
import os
from urllib.parse import urlparse

# =============================================================================
# REDIS CACHE CONFIGURATION
# =============================================================================

# Parse Redis URL from environment
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# Cache timeouts (in seconds)
CACHE_TIMEOUTS = {
    'default': 300,           # 5 minutes
    'short': 60,              # 1 minute
    'medium': 900,            # 15 minutes
    'long': 3600,             # 1 hour
    'very_long': 86400,       # 24 hours
    'session': 86400 * 7,     # 7 days
}

# Primary cache configuration using Redis
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
            'SERIALIZER': 'django_redis.serializers.json.JSONSerializer',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 100,
                'retry_on_timeout': True,
            },
            'SOCKET_CONNECT_TIMEOUT': 5,
            'SOCKET_TIMEOUT': 5,
        },
        'KEY_PREFIX': 'bunoraa',
        'VERSION': 1,
        'TIMEOUT': CACHE_TIMEOUTS['default'],
    },
    # Cache for template fragments
    'template_fragments': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
        },
        'KEY_PREFIX': 'bunoraa:template',
        'TIMEOUT': CACHE_TIMEOUTS['medium'],
    },
    # Cache for API responses
    'api_cache': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'KEY_PREFIX': 'bunoraa:api',
        'TIMEOUT': CACHE_TIMEOUTS['short'],
    },
    # Cache for heavy computations (ML, analytics)
    'computation': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'KEY_PREFIX': 'bunoraa:compute',
        'TIMEOUT': CACHE_TIMEOUTS['long'],
    },
}

# =============================================================================
# SESSION CONFIGURATION
# =============================================================================

# Use Redis for session storage
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'
SESSION_COOKIE_AGE = 86400 * 7  # 7 days
SESSION_COOKIE_SECURE = True  # Only send over HTTPS
SESSION_COOKIE_HTTPONLY = True  # Prevent JavaScript access
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_SAVE_EVERY_REQUEST = True

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
