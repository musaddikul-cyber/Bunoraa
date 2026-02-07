"""
S3/Cloudflare settings for local development or testing
"""
import os
from .base import *

# Parse DEBUG as boolean
DEBUG = os.environ.get('DEBUG', 'True').lower() in ('1', 'true', 'yes')

# Use S3/Cloudflare for media files
USE_S3 = True

# MEDIA_URL will be set by base.py S3 logic
# Do not set LOCAL_MEDIA_URL or MEDIA_ROOT here

# Optionally override ALLOWED_HOSTS for local testing
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Optionally set email backend for local
EMAIL_BACKEND = os.environ.get('EMAIL_BACKEND', 'django.core.mail.backends.console.EmailBackend')

# Database fallback for local development: if DEBUG is True, use SQLite so you
# don't need a remote DATABASE_URL during local testing.
if DEBUG:
    import dj_database_url
    DATABASES = {
        'default': dj_database_url.config(
            default=os.environ.get('DATABASE_URL'),
            conn_max_age=600,
            ssl_require=True,
        )
    }
    # Cache
    CACHES = {
        'default': {
            'BACKEND': 'django_redis.cache.RedisCache',
            'LOCATION': os.environ.get('REDIS_URL'),
            'OPTIONS': {
                'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            }
        }
    }

    # DATABASES = {
    #     'default': {
    #         'ENGINE': 'django.db.backends.sqlite3',
    #         'NAME': BASE_DIR / 'db.sqlite3',
    #     }
    # }
    # # Use local in-memory cache during development
    # CACHES = {
    #     'default': {
    #         'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
    #         'LOCATION': 'unique-snowflake',
    #     }
    # }
else:
    # In non-debug, expect DATABASE_URL to be provided (production)
    pass

# Security settings: in DEBUG (development), do not set secure-only cookies so CSRF cookie
# will be sent over plain HTTP. In production (DEBUG=False), enable stricter security.
if DEBUG:
    SESSION_COOKIE_SECURE = False
    CSRF_COOKIE_SECURE = False
    SECURE_SSL_REDIRECT = False
    # Disable HSTS in development
    SECURE_HSTS_SECONDS = 0
    SECURE_HSTS_INCLUDE_SUBDOMAINS = False
    SECURE_HSTS_PRELOAD = False
else:
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'True').lower() in ('1', 'true', 'yes')
    CSRF_COOKIE_SECURE = os.environ.get('CSRF_COOKIE_SECURE', 'True').lower() in ('1', 'true', 'yes')
    SECURE_SSL_REDIRECT = os.environ.get('SECURE_SSL_REDIRECT', 'True').lower() in ('1', 'true', 'yes')
    SECURE_HSTS_SECONDS = int(os.environ.get('SECURE_HSTS_SECONDS', 31536000))
    SECURE_HSTS_INCLUDE_SUBDOMAINS = os.environ.get('SECURE_HSTS_INCLUDE_SUBDOMAINS', 'True').lower() in ('1', 'true', 'yes')
    SECURE_HSTS_PRELOAD = os.environ.get('SECURE_HSTS_PRELOAD', 'True').lower() in ('1', 'true', 'yes')

# CORS allow all for development
CORS_ALLOW_ALL_ORIGINS = True

# =============================================================================
# DEBUG TOOLBAR (for development with S3)
# =============================================================================
if DEBUG:
    try:
        import debug_toolbar
        INSTALLED_APPS += ['debug_toolbar']
        # Insert after GZipMiddleware
        try:
            gzip_index = MIDDLEWARE.index('django.middleware.gzip.GZipMiddleware')
            MIDDLEWARE.insert(gzip_index + 1, 'debug_toolbar.middleware.DebugToolbarMiddleware')
        except ValueError:
            MIDDLEWARE.insert(0, 'debug_toolbar.middleware.DebugToolbarMiddleware')
        
        INTERNAL_IPS = ['127.0.0.1', 'localhost', '::1']
        DEBUG_TOOLBAR_CONFIG = {
            # is_ajax() was removed in Django 4.x, use X-Requested-With header check instead
            'SHOW_TOOLBAR_CALLBACK': lambda request: DEBUG and request.META.get('HTTP_X_REQUESTED_WITH') != 'XMLHttpRequest' and 'text/html' in request.META.get('HTTP_ACCEPT', ''),
            'RESULTS_CACHE_SIZE': 100,
            'IS_RUNNING_TESTS': False,
            # Disable toolbar from intercepting responses that can cause ASGI issues
            'RENDER_PANELS': True,
        }
    except ImportError:
        pass

# =============================================================================
# LOGGING - Reasonable verbosity for S3 development
# =============================================================================
LOGGING['handlers']['console']['level'] = 'DEBUG'
LOGGING['handlers']['console']['filters'] = []  # Remove require_debug_true filter
LOGGING['loggers']['bunoraa']['level'] = 'INFO'
LOGGING['loggers']['bunoraa.i18n'] = {'level': 'DEBUG', 'handlers': ['console'], 'propagate': False}  # Debug currency issues
LOGGING['loggers']['django']['level'] = 'INFO'
LOGGING['loggers']['django.db.backends'] = {'level': 'WARNING', 'handlers': ['console'], 'propagate': False}  # Suppress SQL logging
LOGGING['root']['level'] = 'INFO'

# django.request errors go to console
LOGGING['loggers'].setdefault('django.request', {'handlers': ['console'], 'level': 'ERROR', 'propagate': False})

