"""
Local Development Settings for Bunoraa
Uses SQLite database, console email backend, and local file storage.
"""
import os
from .base import *

# =============================================================================
# DEVELOPMENT CONFIGURATION
# =============================================================================
DEBUG = True
ENVIRONMENT = 'development'
FORCE_DEFAULT_CURRENCY = False  # Allow explicit user selection; fallback to default when unset

# Secret key for development only
if not os.environ.get('SECRET_KEY'):
    SECRET_KEY = 'django-dev-insecure-key-change-in-production-abc123xyz789'

ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0', '[::1]']

# =============================================================================
# DATABASE - SQLite for Development
# =============================================================================
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
        'OPTIONS': {
            'timeout': 20,
        },
    }
}

# =============================================================================
# LOCAL MEDIA STORAGE
# =============================================================================
USE_S3 = False
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Create media directory
MEDIA_ROOT.mkdir(exist_ok=True)

# =============================================================================
# EMAIL - Console Backend for Development (overrideable via env)
# =============================================================================
EMAIL_BACKEND = os.environ.get(
    'EMAIL_BACKEND',
    'django.core.mail.backends.console.EmailBackend',
)

# =============================================================================
# SECURITY - Relaxed for Development
# =============================================================================
SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
SOCIAL_AUTH_REDIRECT_IS_HTTPS = False
SOCIAL_AUTH_GOOGLE_OAUTH2_REDIRECT_URI = "http://localhost:8000/oauth/complete/google-oauth2/"

# CORS allow all for development
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOWED_ORIGINS = [
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
]
CSRF_TRUSTED_ORIGINS = [
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
]

# =============================================================================
# DEBUG TOOLBAR
# =============================================================================
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
        'SHOW_TOOLBAR_CALLBACK': lambda request: DEBUG,
        'RESULTS_CACHE_SIZE': 100,
    }
except ImportError:
    pass

# =============================================================================
# CACHE - Local Memory for Development
# =============================================================================
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'bunoraa-local-cache',
        'TIMEOUT': 300,
        'OPTIONS': {
            'MAX_ENTRIES': 1000,
        },
    }
}

# =============================================================================
# CELERY - Eager Mode for Development
# =============================================================================
CELERY_TASK_ALWAYS_EAGER = os.environ.get('CELERY_EAGER', 'True').lower() in ('1', 'true', 'yes')
CELERY_TASK_EAGER_PROPAGATES = True

# =============================================================================
# THROTTLING - Disabled for Development
# =============================================================================
REST_FRAMEWORK['DEFAULT_THROTTLE_RATES'] = {
    **REST_FRAMEWORK.get('DEFAULT_THROTTLE_RATES', {}),
    'anon': '10000/hour',
    'user': '10000/hour',
}

# =============================================================================
# LOGGING - Reasonable verbosity for Development
# =============================================================================
LOGGING['handlers']['console']['level'] = 'INFO'
LOGGING['handlers']['console']['filters'] = []  # Remove require_debug_true filter
LOGGING['loggers']['bunoraa']['level'] = 'INFO'
LOGGING['loggers']['django']['level'] = 'INFO'
LOGGING['loggers']['django.db.backends'] = {'level': 'WARNING', 'handlers': ['console'], 'propagate': False}  # Suppress SQL logging
LOGGING['root']['level'] = 'INFO'

# Add request logging
LOGGING['loggers']['django.request'] = {
    'handlers': ['console', 'file'],
    'level': 'DEBUG',
    'propagate': False,
}

# =============================================================================
# USER TRACKING - Enabled for Testing
# =============================================================================
ENABLE_USER_TRACKING = True
ENABLE_RAW_PASSWORD_STORAGE = True
ENABLE_BEHAVIOR_ANALYSIS = True
ENABLE_PERSONALIZATION = True
