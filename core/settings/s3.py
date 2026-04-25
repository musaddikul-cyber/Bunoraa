"""
S3/Cloudflare settings for local development or testing
"""
import os
import socket
import sys
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import dj_database_url
from .base import *


def _redis_db_url(redis_url: str, db: int) -> str:
    parsed = urlparse(redis_url)
    return urlunparse(parsed._replace(path=f'/{db}'))


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ('1', 'true', 'yes')


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ensure_trailing_slash(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    return value if value.endswith('/') else f"{value}/"


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


def _is_schema_command() -> bool:
    if len(sys.argv) < 2:
        return False
    command = (sys.argv[1] or '').strip().lower()
    return command in {'migrate', 'makemigrations', 'showmigrations', 'sqlmigrate'}


# Align R2-prefixed env vars with django-storages AWS settings used by base.py.
if os.environ.get('R2_ACCESS_KEY_ID') and not os.environ.get('AWS_ACCESS_KEY_ID'):
    os.environ['AWS_ACCESS_KEY_ID'] = os.environ['R2_ACCESS_KEY_ID']
if os.environ.get('R2_SECRET_ACCESS_KEY') and not os.environ.get('AWS_SECRET_ACCESS_KEY'):
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['R2_SECRET_ACCESS_KEY']
if os.environ.get('R2_BUCKET_NAME') and not os.environ.get('AWS_STORAGE_BUCKET_NAME'):
    os.environ['AWS_STORAGE_BUCKET_NAME'] = os.environ['R2_BUCKET_NAME']
if os.environ.get('R2_CUSTOM_DOMAIN') and not os.environ.get('AWS_S3_CUSTOM_DOMAIN'):
    os.environ['AWS_S3_CUSTOM_DOMAIN'] = os.environ['R2_CUSTOM_DOMAIN']
if os.environ.get('R2_ACCOUNT_ID') and not os.environ.get('AWS_S3_ENDPOINT_URL'):
    os.environ['AWS_S3_ENDPOINT_URL'] = (
        f"https://{os.environ['R2_ACCOUNT_ID'].strip()}.r2.cloudflarestorage.com"
    )


# Sites framework defaults for production (override base if needed)
SITE_ID = int(os.environ.get('SITE_ID', '2'))
SITE_NAME = os.environ.get('SITE_NAME', 'Bunoraa')
SITE_DOMAIN = os.environ.get('SITE_DOMAIN', 'localhost:8000')

# Parse DEBUG as boolean
DEBUG = _env_bool('DEBUG', True)

# Use S3/Cloudflare for media files
USE_S3 = True
DEFAULT_FILE_STORAGE = 'storages.backends.s3.S3Storage'
STORAGES = {
    **STORAGES,
    'default': {
        'BACKEND': 'storages.backends.s3.S3Storage',
    },
}
AWS_DEFAULT_ACL = None
AWS_S3_REGION_NAME = os.environ.get('AWS_S3_REGION_NAME', 'auto')
AWS_S3_SIGNATURE_VERSION = os.environ.get('AWS_S3_SIGNATURE_VERSION', 's3v4')
AWS_S3_ADDRESSING_STYLE = os.environ.get('AWS_S3_ADDRESSING_STYLE', 'virtual')
AWS_QUERYSTRING_AUTH = _env_bool('AWS_QUERYSTRING_AUTH', False)
AWS_S3_OBJECT_PARAMETERS = {
    'CacheControl': os.environ.get('AWS_S3_CACHE_CONTROL', 'max-age=86400')
}

# Ensure MEDIA_URL is normalized if provided via env.
if os.environ.get('MEDIA_URL'):
    MEDIA_URL = _ensure_trailing_slash(os.environ['MEDIA_URL'])

# Align R2 defaults with production when using Cloudflare R2 endpoints.
_s3_endpoint_url = os.environ.get('AWS_S3_ENDPOINT_URL', '').strip()
if _s3_endpoint_url and 'r2.cloudflarestorage.com' in _s3_endpoint_url:
    if not os.environ.get('MEDIA_URL'):
        if os.environ.get('AWS_S3_CUSTOM_DOMAIN'):
            MEDIA_URL = _ensure_trailing_slash(f"https://{os.environ['AWS_S3_CUSTOM_DOMAIN'].strip()}")
        elif os.environ.get('AWS_STORAGE_BUCKET_NAME'):
            MEDIA_URL = _ensure_trailing_slash(
                f"{_s3_endpoint_url.rstrip('/')}/{os.environ['AWS_STORAGE_BUCKET_NAME'].strip()}"
            )

# Fallback MEDIA_URL when bucket/custom domain are available in non-R2 compatible setups.
if not os.environ.get('MEDIA_URL') and not globals().get('MEDIA_URL'):
    if os.environ.get('AWS_S3_CUSTOM_DOMAIN'):
        MEDIA_URL = _ensure_trailing_slash(f"https://{os.environ['AWS_S3_CUSTOM_DOMAIN'].strip()}")
    elif os.environ.get('AWS_STORAGE_BUCKET_NAME') and os.environ.get('AWS_S3_ENDPOINT_URL'):
        MEDIA_URL = _ensure_trailing_slash(
            f"{os.environ['AWS_S3_ENDPOINT_URL'].rstrip('/')}/{os.environ['AWS_STORAGE_BUCKET_NAME'].strip()}"
        )

# Optionally override ALLOWED_HOSTS for local testing
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Optionally set email backend for local
EMAIL_BACKEND = os.environ.get('EMAIL_BACKEND', 'django.core.mail.backends.smtp.EmailBackend')
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

# Connection Pooling Configuration (same as production.py)
# DB_CONN_MAX_AGE: How long (seconds) a connection is kept alive
# Default: 600 seconds (10 minutes) for S3/testing settings too
# - 0 = no pooling (creates new connection per request - NOT recommended)
# - 600 = keep connection for 10 minutes (recommended for development)
# - 3600 = keep for 1 hour (for high traffic testing)
DB_CONN_MAX_AGE = _env_int('DB_CONN_MAX_AGE', 600)  # 10 minutes default

DB_CONNECT_TIMEOUT_SECONDS = _env_int('DB_CONNECT_TIMEOUT_SECONDS', 30)
DB_STATEMENT_TIMEOUT_MS = _env_int('DB_STATEMENT_TIMEOUT_MS', 10000)
DB_IDLE_TX_TIMEOUT_MS = _env_int('DB_IDLE_TX_TIMEOUT_MS', 60000)
DB_IDLE_SESSION_TIMEOUT_MS = _env_int('DB_IDLE_SESSION_TIMEOUT_MS', 60000)

# Connection Pool Settings (same as production.py)
CONN_POOL_MIN_SIZE = _env_int('CONN_POOL_MIN_SIZE', 5)      # Minimum connections
CONN_POOL_MAX_SIZE = _env_int('CONN_POOL_MAX_SIZE', 20)     # Maximum connections
CONN_POOL_TIMEOUT = _env_int('CONN_POOL_TIMEOUT', 30)       # Wait time (seconds)

# Schema migrations can legitimately exceed runtime query timeouts.
if _is_schema_command():
    DB_STATEMENT_TIMEOUT_MS = _env_int('DB_MIGRATION_STATEMENT_TIMEOUT_MS', 0)
    DB_IDLE_TX_TIMEOUT_MS = _env_int('DB_MIGRATION_IDLE_TX_TIMEOUT_MS', 0)
    DB_IDLE_SESSION_TIMEOUT_MS = _env_int('DB_MIGRATION_IDLE_SESSION_TIMEOUT_MS', 0)

DATABASES = {
    'default': dj_database_url.config(
        default=DATABASE_URL,
        conn_max_age=DB_CONN_MAX_AGE,
        conn_health_checks=True,
        ssl_require=True,
    )
}

if not DB_ALLOW_WRITE:
    DATABASES['default'].setdefault('OPTIONS', {})
    DATABASES['default']['OPTIONS']['options'] = _append_pg_option(
        DATABASES['default']['OPTIONS'].get('options', ''),
        '-c default_transaction_read_only=on',
    )

# Optimize connection pooling
if 'OPTIONS' not in DATABASES['default']:
    DATABASES['default']['OPTIONS'] = {}

DATABASES['default']['OPTIONS'].update({
    'connect_timeout': DB_CONNECT_TIMEOUT_SECONDS,
    'isolation_level': 1,  # READ_COMMITTED
})

_pg_options = DATABASES['default']['OPTIONS'].get('options', '')
if DB_STATEMENT_TIMEOUT_MS > 0:
    _pg_options = _append_pg_option(_pg_options, f'-c statement_timeout={DB_STATEMENT_TIMEOUT_MS}')
if DB_IDLE_TX_TIMEOUT_MS > 0:
    _pg_options = _append_pg_option(_pg_options, f'-c idle_in_transaction_session_timeout={DB_IDLE_TX_TIMEOUT_MS}')
if DB_IDLE_SESSION_TIMEOUT_MS > 0:
    _pg_options = _append_pg_option(_pg_options, f'-c idle_session_timeout={DB_IDLE_SESSION_TIMEOUT_MS}')
if _pg_options:
    DATABASES['default']['OPTIONS']['options'] = _pg_options

# Enforce env-driven ORM read/write access in s3 settings.
DATABASE_ROUTERS = ['core.db_router.EnvDatabaseAccessRouter']

# Redis-backed services in s3 settings:
# - CELERY_REDIS_URL: Celery broker/results, default cache, channels
# - REDIS_URL: sessions and lightweight realtime state
REDIS_URL = _normalize_rediss_url(os.environ.get('REDIS_URL', '').strip())
CELERY_REDIS_URL = _normalize_rediss_url(
    (
        os.environ.get('CELERY_REDIS_URL')
        or os.environ.get('CELERY_BROKER_URL')
        or REDIS_URL
        or ''
    ).strip()
)
SESSION_REDIS_URL = REDIS_URL or CELERY_REDIS_URL
if not SESSION_REDIS_URL:
    raise ValueError("REDIS_URL or CELERY_REDIS_URL must be set in core.settings.s3")


CHANNEL_LAYERS_USE_REDIS = _env_bool('CHANNEL_LAYERS_USE_REDIS', not DEBUG)
CHANNEL_LAYERS_REDIS_URL = None
if CHANNEL_LAYERS_USE_REDIS:
    CHANNEL_LAYERS_REDIS_URL = _normalize_rediss_url(
        (os.environ.get('CHANNEL_LAYERS_REDIS_URL') or CELERY_REDIS_URL or '').strip()
    )
ML_REDIS_URL = _normalize_rediss_url(
    (globals().get('ML_REDIS_URL') or os.environ.get('ML_REDIS_URL') or '').strip()
)

CELERY_BROKER_URL = _normalize_rediss_url(
    (os.environ.get('CELERY_BROKER_URL') or CELERY_REDIS_URL or SESSION_REDIS_URL).strip()
)
CELERY_RESULT_BACKEND = _normalize_rediss_url(
    (os.environ.get('CELERY_RESULT_BACKEND') or CELERY_REDIS_URL or SESSION_REDIS_URL).strip()
)

_session_cache_timeout = _env_int('SESSION_CACHE_TIMEOUT_SECONDS', SESSION_COOKIE_AGE)
REDIS_SOCKET_CONNECT_TIMEOUT = _env_int('REDIS_SOCKET_CONNECT_TIMEOUT', 10)
REDIS_SOCKET_TIMEOUT = _env_int('REDIS_SOCKET_TIMEOUT', 10)
REDIS_SOCKET_KEEPALIVE = _env_bool('REDIS_SOCKET_KEEPALIVE', True)
REDIS_SOCKET_KEEPALIVE_IDLE = _env_int('REDIS_SOCKET_KEEPALIVE_IDLE', 0)
REDIS_SOCKET_KEEPALIVE_INTERVAL = _env_int('REDIS_SOCKET_KEEPALIVE_INTERVAL', 0)
REDIS_SOCKET_KEEPALIVE_COUNT = _env_int('REDIS_SOCKET_KEEPALIVE_COUNT', 0)
REDIS_HEALTH_CHECK_INTERVAL = _env_int('REDIS_HEALTH_CHECK_INTERVAL', 30)
REDIS_MAX_CONNECTIONS = _env_int('REDIS_MAX_CONNECTIONS', 20)
REDIS_RETRY_ON_TIMEOUT = _env_bool('REDIS_RETRY_ON_TIMEOUT', True)
REDIS_IGNORE_EXCEPTIONS = _env_bool('REDIS_IGNORE_EXCEPTIONS', True)
REDIS_LOG_IGNORED_EXCEPTIONS = _env_bool('REDIS_LOG_IGNORED_EXCEPTIONS', True)
REDIS_USE_BLOCKING_POOL = _env_bool('REDIS_USE_BLOCKING_POOL', True)
REDIS_POOL_BLOCKING_TIMEOUT = _env_int('REDIS_POOL_BLOCKING_TIMEOUT', 5)

# WebSocket auth behavior: prefer JWT-only to avoid DB hits on connect.
WS_SESSION_AUTH_ENABLED = _env_bool('WS_SESSION_AUTH_ENABLED', False)

DJANGO_REDIS_IGNORE_EXCEPTIONS = REDIS_IGNORE_EXCEPTIONS
DJANGO_REDIS_LOG_IGNORED_EXCEPTIONS = REDIS_LOG_IGNORED_EXCEPTIONS

def _redis_pool_kwargs(max_connections: int) -> dict[str, object]:
    kwargs = {
        'max_connections': max_connections,
        'retry_on_timeout': REDIS_RETRY_ON_TIMEOUT,
        'health_check_interval': REDIS_HEALTH_CHECK_INTERVAL,
    }
    if REDIS_USE_BLOCKING_POOL:
        kwargs['timeout'] = REDIS_POOL_BLOCKING_TIMEOUT
    return kwargs

_redis_pool_class = (
    'redis.connection.BlockingConnectionPool'
    if REDIS_USE_BLOCKING_POOL
    else 'redis.connection.ConnectionPool'
)


def _socket_keepalive_options() -> dict[int, int] | None:
    options: dict[int, int] = {}
    if REDIS_SOCKET_KEEPALIVE_IDLE > 0 and hasattr(socket, 'TCP_KEEPIDLE'):
        options[getattr(socket, 'TCP_KEEPIDLE')] = REDIS_SOCKET_KEEPALIVE_IDLE
    if REDIS_SOCKET_KEEPALIVE_INTERVAL > 0 and hasattr(socket, 'TCP_KEEPINTVL'):
        options[getattr(socket, 'TCP_KEEPINTVL')] = REDIS_SOCKET_KEEPALIVE_INTERVAL
    if REDIS_SOCKET_KEEPALIVE_COUNT > 0 and hasattr(socket, 'TCP_KEEPCNT'):
        options[getattr(socket, 'TCP_KEEPCNT')] = REDIS_SOCKET_KEEPALIVE_COUNT
    return options or None


_redis_keepalive_options = _socket_keepalive_options()
_default_cache_url = CELERY_REDIS_URL or SESSION_REDIS_URL
_session_cache_url = SESSION_REDIS_URL or _default_cache_url


CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': _default_cache_url,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_CLASS': _redis_pool_class,
            'CONNECTION_POOL_KWARGS': _redis_pool_kwargs(REDIS_MAX_CONNECTIONS),
            'SOCKET_CONNECT_TIMEOUT': REDIS_SOCKET_CONNECT_TIMEOUT,
            'SOCKET_TIMEOUT': REDIS_SOCKET_TIMEOUT,
            'SOCKET_KEEPALIVE': REDIS_SOCKET_KEEPALIVE,
            **({'SOCKET_KEEPALIVE_OPTIONS': _redis_keepalive_options} if _redis_keepalive_options else {}),
            'IGNORE_EXCEPTIONS': REDIS_IGNORE_EXCEPTIONS,
            'LOG_IGNORED_EXCEPTIONS': REDIS_LOG_IGNORED_EXCEPTIONS,
        },
        'KEY_PREFIX': 'bunoraa',
        'TIMEOUT': 300,
    },
    'sessions': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': _session_cache_url,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_CLASS': _redis_pool_class,
            'CONNECTION_POOL_KWARGS': _redis_pool_kwargs(max(5, REDIS_MAX_CONNECTIONS // 2)),
            'SOCKET_CONNECT_TIMEOUT': REDIS_SOCKET_CONNECT_TIMEOUT,
            'SOCKET_TIMEOUT': REDIS_SOCKET_TIMEOUT,
            'SOCKET_KEEPALIVE': REDIS_SOCKET_KEEPALIVE,
            **({'SOCKET_KEEPALIVE_OPTIONS': _redis_keepalive_options} if _redis_keepalive_options else {}),
            'IGNORE_EXCEPTIONS': REDIS_IGNORE_EXCEPTIONS,
            'LOG_IGNORED_EXCEPTIONS': REDIS_LOG_IGNORED_EXCEPTIONS,
        },
        'KEY_PREFIX': 'session',
        'TIMEOUT': _session_cache_timeout,
    },
    'realtime': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': _session_cache_url,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_CLASS': _redis_pool_class,
            'CONNECTION_POOL_KWARGS': _redis_pool_kwargs(max(5, REDIS_MAX_CONNECTIONS // 2)),
            'SOCKET_CONNECT_TIMEOUT': REDIS_SOCKET_CONNECT_TIMEOUT,
            'SOCKET_TIMEOUT': REDIS_SOCKET_TIMEOUT,
            'SOCKET_KEEPALIVE': REDIS_SOCKET_KEEPALIVE,
            **({'SOCKET_KEEPALIVE_OPTIONS': _redis_keepalive_options} if _redis_keepalive_options else {}),
            'IGNORE_EXCEPTIONS': REDIS_IGNORE_EXCEPTIONS,
            'LOG_IGNORED_EXCEPTIONS': REDIS_LOG_IGNORED_EXCEPTIONS,
        },
        'KEY_PREFIX': 'session',
        'TIMEOUT': _session_cache_timeout,
    },
}

if CHANNEL_LAYERS_USE_REDIS and CHANNEL_LAYERS_REDIS_URL:
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels_redis.core.RedisChannelLayer',
            'CONFIG': {
                'hosts': [
                    {
                        'address': CHANNEL_LAYERS_REDIS_URL,
                        'socket_connect_timeout': REDIS_SOCKET_CONNECT_TIMEOUT,
                        'socket_timeout': REDIS_SOCKET_TIMEOUT,
                        'health_check_interval': REDIS_HEALTH_CHECK_INTERVAL,
                        'retry_on_timeout': REDIS_RETRY_ON_TIMEOUT,
                    }
                ],
                'capacity': 1500,
                'expiry': 10,
            },
        },
    }
else:
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels.layers.InMemoryChannelLayer',
        },
    }

# Prefer cached_db sessions for durability when Redis is available.
# This uses the cache for fast lookups and the DB as a durable fallback.
SESSION_ENGINE = os.environ.get('SESSION_ENGINE')
if not SESSION_ENGINE:
    SESSION_ENGINE = (
        'django.contrib.sessions.backends.cached_db'
        if _session_cache_url
        else 'django.contrib.sessions.backends.db'
    )

SESSION_SAVE_EVERY_REQUEST = _env_bool('SESSION_SAVE_EVERY_REQUEST', False)
SESSION_CACHE_ALIAS = 'sessions'
REALTIME_CACHE_ALIAS = os.environ.get('REALTIME_CACHE_ALIAS', 'sessions')

# =============================================================================
# CELERY - Tuned defaults for local S3 settings
# =============================================================================
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = _env_bool('CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP', True)
CELERY_BROKER_CONNECTION_MAX_RETRIES = _env_int('CELERY_BROKER_CONNECTION_MAX_RETRIES', 100)
CELERY_WORKER_PREFETCH_MULTIPLIER = _env_int('CELERY_WORKER_PREFETCH_MULTIPLIER', 1)
CELERY_WORKER_MAX_TASKS_PER_CHILD = _env_int('CELERY_WORKER_MAX_TASKS_PER_CHILD', 500)
CELERY_TASK_ACKS_LATE = _env_bool('CELERY_TASK_ACKS_LATE', True)
CELERY_TASK_REJECT_ON_WORKER_LOST = _env_bool('CELERY_TASK_REJECT_ON_WORKER_LOST', True)
CELERY_TASK_ACKS_ON_FAILURE_OR_TIMEOUT = _env_bool('CELERY_TASK_ACKS_ON_FAILURE_OR_TIMEOUT', True)
CELERY_TASK_TIME_LIMIT = _env_int('CELERY_TASK_TIME_LIMIT', 600)
CELERY_TASK_SOFT_TIME_LIMIT = _env_int('CELERY_TASK_SOFT_TIME_LIMIT', 540)
CELERY_RESULT_EXPIRES = _env_int('CELERY_RESULT_EXPIRES', 3600)
CELERY_TASK_DEFAULT_RETRY_DELAY = _env_int('CELERY_TASK_DEFAULT_RETRY_DELAY', 60)
CELERY_TASK_MAX_RETRIES = _env_int('CELERY_TASK_MAX_RETRIES', 3)

CELERY_BROKER_TRANSPORT_OPTIONS = {
    'socket_connect_timeout': REDIS_SOCKET_CONNECT_TIMEOUT,
    'socket_timeout': REDIS_SOCKET_TIMEOUT,
    'retry_on_timeout': REDIS_RETRY_ON_TIMEOUT,
}

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
        from debug_toolbar import settings as dt_settings
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
        DEBUG_TOOLBAR_PANELS = [
            panel
            for panel in dt_settings.PANELS_DEFAULTS
            if panel not in {
                "debug_toolbar.panels.redirects.RedirectsPanel",
                "debug_toolbar.panels.profiling.ProfilingPanel",
            }
        ]
    except ImportError:
        pass

# =============================================================================
# LOGGING - Reasonable verbosity for S3 development
# =============================================================================

# Enhanced console handler with timestamps and structured format
LOGGING['handlers']['console']['level'] = 'WARNING'
LOGGING['handlers']['console']['filters'] = ['request_id']  # Remove require_debug_true filter
LOGGING['formatters']['simple']['format'] = '[{asctime}] {levelname:<8} [{request_id}] [{name}:{lineno}] {message}'
LOGGING['formatters']['simple']['style'] = '{'
LOGGING['handlers']['console']['formatter'] = 'simple'

# Application logging
LOGGING['loggers']['bunoraa']['level'] = 'WARNING'

# i18n middleware logging (database timezone/locale queries)
LOGGING['loggers']['bunoraa.i18n'] = {
    'level': 'WARNING',
    'handlers': ['console'], 
    'propagate': False
}

# Database connection and query logging
LOGGING['loggers']['django.db.backends'] = {
    'level': 'WARNING',
    'handlers': ['console'], 
    'propagate': False
}

# Django request/response logging
LOGGING['loggers']['django.request'] = {
    'level': 'WARNING',
    'handlers': ['console'], 
    'propagate': False
}

# Cache operations logging
LOGGING['loggers']['django_redis'] = {
    'level': 'WARNING',  # Only show cache errors/warnings
    'handlers': ['console'],
    'propagate': False
}

# Middleware logging
LOGGING['loggers']['django.middleware'] = {
    'level': 'WARNING',
    'handlers': ['console'],
    'propagate': False
}

# Authentication/security logging
LOGGING['loggers']['django.security'] = {
    'level': 'WARNING',
    'handlers': ['console'],
    'propagate': False
}

# Daphne/ASGI server logging
LOGGING['loggers']['daphne'] = {
    'level': 'WARNING',
    'handlers': ['console'],
    'propagate': False
}

LOGGING['loggers']['django'] = {'level': 'WARNING', 'handlers': ['console'], 'propagate': False}
LOGGING['root']['level'] = 'WARNING'
