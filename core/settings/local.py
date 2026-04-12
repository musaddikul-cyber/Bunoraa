"""
Local Development Settings for Bunoraa
Uses SQLite database, console email backend, and local file storage.
"""
import os
from urllib.parse import urlparse, urlunparse
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
# Respect USE_S3 env in local dev. If enabled, keep S3 settings from base.py.
USE_S3 = False
if not USE_S3:
    MEDIA_URL = '/media/'
    MEDIA_ROOT = BASE_DIR / 'media'
    DEFAULT_FILE_STORAGE = 'django.core.files.storage.FileSystemStorage'
    STORAGES = {
        **STORAGES,
        'default': {
            'BACKEND': 'django.core.files.storage.FileSystemStorage',
            'OPTIONS': {
                'location': MEDIA_ROOT,
                'base_url': MEDIA_URL,
            },
        },
    }

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
    from debug_toolbar import settings as dt_settings
    INSTALLED_APPS += ['debug_toolbar']
    # Insert after GZipMiddleware
    try:
        gzip_index = MIDDLEWARE.index('django.middleware.gzip.GZipMiddleware')
        MIDDLEWARE.insert(gzip_index + 1, 'debug_toolbar.middleware.DebugToolbarMiddleware')
    except ValueError:
        MIDDLEWARE.insert(0, 'debug_toolbar.middleware.DebugToolbarMiddleware')
    
    INTERNAL_IPS = ['127.0.0.1', 'localhost', '::1']

    def _show_debug_toolbar(request):
        if not DEBUG:
            return False
        if request.path.startswith(('/api/', '/ws/', '/static/', '/media/')):
            return False
        if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
            return False
        return 'text/html' in request.META.get('HTTP_ACCEPT', '')

    DEBUG_TOOLBAR_CONFIG = {
        'SHOW_TOOLBAR_CALLBACK': _show_debug_toolbar,
        'RESULTS_CACHE_SIZE': 100,
        'IS_RUNNING_TESTS': False,
        'RENDER_PANELS': True,
    }
    DEBUG_TOOLBAR_PANELS = [
        panel
        for panel in dt_settings.PANELS_DEFAULTS
        if panel != "debug_toolbar.panels.redirects.RedirectsPanel"
    ]
except ImportError:
    pass

def _local_redis_db_url(redis_url: str, db: int) -> str:
    parsed = urlparse(redis_url)
    if not parsed.scheme:
        return redis_url
    return urlunparse(parsed._replace(path=f'/{db}'))


# =============================================================================
# REDIS - Force localhost defaults for local settings
# =============================================================================
LOCAL_REDIS_URL = os.environ.get('LOCAL_REDIS_URL', 'redis://127.0.0.1:6379/0').strip()
if not LOCAL_REDIS_URL:
    LOCAL_REDIS_URL = 'redis://127.0.0.1:6379/0'
REDIS_URL = LOCAL_REDIS_URL

# =============================================================================
# CACHE - Redis for Development
# =============================================================================
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': LOCAL_REDIS_URL,
        'TIMEOUT': 300,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'IGNORE_EXCEPTIONS': True,
            'SOCKET_CONNECT_TIMEOUT': 2,
            'SOCKET_TIMEOUT': 2,
        },
    }
}
DJANGO_REDIS_IGNORE_EXCEPTIONS = True

# =============================================================================
# CHANNEL LAYERS - Redis by default in local development
# =============================================================================
CHANNEL_LAYERS_USE_REDIS = os.environ.get('CHANNEL_LAYERS_USE_REDIS', 'True').lower() in ('1', 'true', 'yes')
if CHANNEL_LAYERS_USE_REDIS:
    _channel_layer_redis_url = (
        os.environ.get('CHANNEL_LAYERS_REDIS_URL')
        or os.environ.get('LOCAL_CHANNEL_LAYERS_REDIS_URL')
        or _local_redis_db_url(LOCAL_REDIS_URL, 2)
    )
    if _channel_layer_redis_url:
        CHANNEL_LAYERS = {
            'default': {
                'BACKEND': 'channels_redis.core.RedisChannelLayer',
                'CONFIG': {
                    'hosts': [_channel_layer_redis_url],
                },
            },
        }
    else:
        CHANNEL_LAYERS = {
            'default': {
                'BACKEND': 'channels.layers.InMemoryChannelLayer',
            },
        }
else:
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels.layers.InMemoryChannelLayer',
        },
    }

# =============================================================================
# CELERY - Eager Mode for Development
# =============================================================================
CELERY_TASK_ALWAYS_EAGER = os.environ.get('CELERY_EAGER', 'True').lower() in ('1', 'true', 'yes')
CELERY_TASK_EAGER_PROPAGATES = True
if CELERY_TASK_ALWAYS_EAGER:
    # Keep local development fully in-process so remote Redis config cannot break requests.
    CELERY_BROKER_URL = 'memory://'
    CELERY_RESULT_BACKEND = 'cache+memory://'
    CELERY_TASK_IGNORE_RESULT = True
    CELERY_TASK_STORE_EAGER_RESULT = False
else:
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', _local_redis_db_url(LOCAL_REDIS_URL, 1))
    CELERY_RESULT_BACKEND = os.environ.get(
        'CELERY_RESULT_BACKEND',
        _local_redis_db_url(LOCAL_REDIS_URL, 3),
    )

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
LOGGING['handlers']['console']['filters'] = ['request_id']  # Remove require_debug_true filter
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
