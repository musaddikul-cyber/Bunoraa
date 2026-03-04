"""
S3/Cloudflare settings for local development or testing
"""
import os
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import dj_database_url
from .base import *


def _redis_db_url(redis_url: str, db: int) -> str:
    parsed = urlparse(redis_url)
    return urlunparse(parsed._replace(path=f'/{db}'))


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ('1', 'true', 'yes')


def _normalize_rediss_url(url: str | None) -> str | None:
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


def _append_pg_option(existing_options: str, option: str) -> str:
    existing_options = (existing_options or '').strip()
    if option in existing_options:
        return existing_options
    return f"{existing_options} {option}".strip()


# Sites framework defaults for production (override base if needed)
SITE_ID = int(os.environ.get('SITE_ID', '2'))
SITE_NAME = os.environ.get('SITE_NAME', 'Bunoraa')
SITE_DOMAIN = os.environ.get('SITE_DOMAIN', 'localhost:8000')

# Parse DEBUG as boolean
DEBUG = _env_bool('DEBUG', True)

# Use S3/Cloudflare for media files
USE_S3 = True

# MEDIA_URL will be set by base.py S3 logic
# Do not set LOCAL_MEDIA_URL or MEDIA_ROOT here

# Optionally override ALLOWED_HOSTS for local testing
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Optionally set email backend for local
EMAIL_BACKEND = os.environ.get('EMAIL_BACKEND', 'django.core.mail.backends.console.EmailBackend')
# Ensure registration/verification emails are delivered in debug even when
# background workers are not running.
if DEBUG:
    EMAIL_QUEUE_SYNC_FALLBACK = _env_bool('EMAIL_QUEUE_SYNC_FALLBACK', True)

# Database behavior for S3 settings:
# - Always use PostgreSQL via DATABASE_URL.
# - In DEBUG, write permission can be constrained with DB_ALLOW_WRITE.
DATABASE_URL = os.environ.get('DATABASE_URL', '').strip()
if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in core.settings.s3")

DB_ALLOW_WRITE = _env_bool('DB_ALLOW_WRITE', True)

DATABASES = {
    'default': dj_database_url.config(
        default=DATABASE_URL,
        conn_max_age=300,
        conn_health_checks=True,
        ssl_require=False,
    )
}

if not DB_ALLOW_WRITE:
    DATABASES['default'].setdefault('OPTIONS', {})
    DATABASES['default']['OPTIONS']['options'] = _append_pg_option(
        DATABASES['default']['OPTIONS'].get('options', ''),
        '-c default_transaction_read_only=on',
    )

# Enforce env-driven ORM read/write access in s3 settings.
DATABASE_ROUTERS = ['core.db_router.EnvDatabaseAccessRouter']

# Redis-backed services always use REDIS_URL in core.settings.s3.
REDIS_URL = _normalize_rediss_url(os.environ.get('REDIS_URL', '').strip())
if not REDIS_URL:
    raise ValueError("REDIS_URL must be set in core.settings.s3")

channel_layers_redis_url = _normalize_rediss_url(
    os.environ.get('CHANNEL_LAYERS_REDIS_URL', _redis_db_url(REDIS_URL, 2)).strip()
)
CELERY_BROKER_URL = _normalize_rediss_url(os.environ.get('CELERY_BROKER_URL', _redis_db_url(REDIS_URL, 1)).strip())
CELERY_RESULT_BACKEND = _normalize_rediss_url(os.environ.get('CELERY_RESULT_BACKEND', _redis_db_url(REDIS_URL, 3)).strip())

CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 20,
                'retry_on_timeout': True,
            },
            'SOCKET_CONNECT_TIMEOUT': 5,
            'SOCKET_TIMEOUT': 5,
        },
        'KEY_PREFIX': 'bunoraa',
        'TIMEOUT': 300,
    },
    'sessions': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': _redis_db_url(REDIS_URL, 1),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'KEY_PREFIX': 'session',
        'TIMEOUT': 86400 * 30,
    }
}

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [channel_layers_redis_url],
            'capacity': 1500,
            'expiry': 10,
        },
    },
}

SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'sessions'

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

# Relax API throttle limits for non-production usage of s3 settings.
if ENVIRONMENT != 'production':
    REST_FRAMEWORK['DEFAULT_THROTTLE_RATES'] = {
        **REST_FRAMEWORK.get('DEFAULT_THROTTLE_RATES', {}),
        'anon': os.environ.get('DEV_API_ANON_THROTTLE', '10000/hour'),
        'user': os.environ.get('DEV_API_USER_THROTTLE', '10000/hour'),
    }

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
