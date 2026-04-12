"""
Production Settings for Bunoraa
Optimized for performance, security, and scalability.
Uses PostgreSQL, Redis, Cloudflare R2 storage.
"""
import os
import socket
import sys
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import dj_database_url
from .base import *

# Helpers
def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def _normalize_origin(value: str) -> str:
    value = value.strip()
    if not value:
        return ''
    parsed = urlparse(value)
    if not parsed.scheme:
        value = f"https://{value}"
        parsed = urlparse(value)
    if not parsed.netloc:
        return ''
    return f"{parsed.scheme}://{parsed.netloc}"


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _append_pg_option(existing_options: str, option: str) -> str:
    existing_options = (existing_options or '').strip()
    if option in existing_options:
        return existing_options
    return f"{existing_options} {option}".strip()


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


def _redis_db_url(redis_url: str, db: int) -> str:
    parsed = urlparse(redis_url)
    return urlunparse(parsed._replace(path=f'/{db}'))


def _is_schema_command() -> bool:
    if len(sys.argv) < 2:
        return False
    command = (sys.argv[1] or '').strip().lower()
    return command in {'migrate', 'makemigrations', 'showmigrations', 'sqlmigrate'}

# Sites framework defaults for production (override base if needed)
SITE_ID = int(os.environ.get('SITE_ID', '2'))
SITE_NAME = os.environ.get('SITE_NAME', 'Bunoraa')
SITE_DOMAIN = os.environ.get('SITE_DOMAIN', 'api.bunoraa.com')

# =============================================================================
# PRODUCTION CONFIGURATION
# =============================================================================
DEBUG = False
ENVIRONMENT = 'production'

# Production prerender defaults (overridable via env).
PRERENDER_ENABLED = _env_bool('PRERENDER_ENABLED', True)
PRERENDER_MIDDLEWARE_ENABLED = _env_bool('PRERENDER_MIDDLEWARE_ENABLED', PRERENDER_ENABLED)
PRERENDER_ON_DEMAND_ENABLED = _env_bool('PRERENDER_ON_DEMAND_ENABLED', True)
PRERENDER_VERIFY_GOOGLE_DNS = _env_bool('PRERENDER_VERIFY_GOOGLE_DNS', PRERENDER_ENABLED)

if PRERENDER_MIDDLEWARE_ENABLED and 'core.middleware.bot_prerender.BotPreRenderMiddleware' not in MIDDLEWARE:
    try:
        insert_at = MIDDLEWARE.index('django.contrib.messages.middleware.MessageMiddleware')
    except ValueError:
        insert_at = len(MIDDLEWARE)
    MIDDLEWARE.insert(insert_at, 'core.middleware.bot_prerender.BotPreRenderMiddleware')
elif not PRERENDER_MIDDLEWARE_ENABLED and 'core.middleware.bot_prerender.BotPreRenderMiddleware' in MIDDLEWARE:
    MIDDLEWARE.remove('core.middleware.bot_prerender.BotPreRenderMiddleware')

# =============================================================================
# STARTUP PERFORMANCE OPTIMIZATION
# =============================================================================
# Reduce startup time by 50-75%

# Only enable ML on worker processes, not web servers
ML_ENABLED = os.environ.get('ML_ENABLED', 'False') == 'True'
PROCESS_TYPE = os.environ.get('PROCESS_TYPE', 'web')  # 'web', 'worker', 'scheduler'

# Fast startup mode - skip heavy operations
SKIP_MIGRATIONS_CHECK = os.environ.get('SKIP_MIGRATIONS_CHECK', 'True') == 'True'

# Optimize Redis connection timeouts (2s instead of 10s for faster failure)
REDIS_SOCKET_CONNECT_TIMEOUT = _env_int('REDIS_SOCKET_CONNECT_TIMEOUT', 2)  # Fast timeout
REDIS_SOCKET_TIMEOUT = _env_int('REDIS_SOCKET_TIMEOUT', 2)  # Fast timeout

# Reduce connection pool overhead during startup
STARTUP_CONN_POOL_MIN = 2  # Minimal connections at startup
STARTUP_CONN_POOL_MAX = 5  # Reduced pool during startup

# Ensure SECRET_KEY is set
if not SECRET_KEY:
    raise ValueError("SECRET_KEY must be set in production environment")

# Force-disable any legacy raw-password storage paths in production.
ENABLE_RAW_PASSWORD_STORAGE = False

# Ensure credential encryption has a stable key in production. When a dedicated
# key is not provided, derive one from SECRET_KEY (better than generating a new
# random key on every deploy, which would break decryptability).
if not globals().get("CREDENTIAL_ENCRYPTION_KEY"):
    import base64
    import hashlib

    CREDENTIAL_ENCRYPTION_KEY = base64.urlsafe_b64encode(
        hashlib.sha256(SECRET_KEY.encode("utf-8")).digest()
    ).decode("ascii")

# Parse ALLOWED_HOSTS from environment
_env_allowed = os.environ.get('ALLOWED_HOSTS', '')
if _env_allowed:
    ALLOWED_HOSTS = [h.strip() for h in _env_allowed.split(',') if h.strip()]
else:
    ALLOWED_HOSTS = ['bunoraa.com', 'www.bunoraa.com', 'api.bunoraa.com', 'media.bunoraa.com', 'bunoraa-pl26.onrender.com', 'bunoraa-django.onrender.com', '.onrender.com']

# CORS/CSRF origins - allow explicit overrides and add frontend origin if provided
_cors_env = _split_csv(os.environ.get('CORS_ALLOWED_ORIGINS', 'https://bunoraa.com,https://www.bunoraa.com,https://api.bunoraa.com,https://media.bunoraa.com,https://bunoraa-pl26.onrender.com,https://bunoraa-django.onrender.com'))
_csrf_env = _split_csv(os.environ.get('CSRF_TRUSTED_ORIGINS', 'https://bunoraa.com,https://www.bunoraa.com,https://api.bunoraa.com,https://media.bunoraa.com,https://bunoraa-pl26.onrender.com,https://bunoraa-django.onrender.com'))

CORS_ALLOWED_ORIGINS = _cors_env or [f'https://{h}' for h in ALLOWED_HOSTS if h]
CSRF_TRUSTED_ORIGINS = _csrf_env or [f'https://{h}' for h in ALLOWED_HOSTS if h]

_frontend_origins_raw = _split_csv(os.environ.get('NEXT_FRONTEND_ORIGIN', ''))
if not _frontend_origins_raw:
    _fallback_site_origin = os.environ.get('NEXT_PUBLIC_SITE_URL', '').strip()
    if _fallback_site_origin:
        _frontend_origins_raw = [_fallback_site_origin]

for raw_origin in _frontend_origins_raw:
    frontend_origin = _normalize_origin(raw_origin)
    if not frontend_origin:
        continue
    if frontend_origin not in CORS_ALLOWED_ORIGINS:
        CORS_ALLOWED_ORIGINS.append(frontend_origin)
    if frontend_origin not in CSRF_TRUSTED_ORIGINS:
        CSRF_TRUSTED_ORIGINS.append(frontend_origin)

# =============================================================================
# DATABASE - PostgreSQL with Connection Pooling
# =============================================================================
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in production environment")

# Connection Pooling Configuration
# DB_CONN_MAX_AGE: How long (seconds) a connection is kept alive
# Default: 600 seconds (10 minutes) for production
# - 0 = no pooling (creates new connection per request - BAD for production)
# - 600 = keep connection for 10 minutes (recommended)
# - 3600 = keep for 1 hour (for high traffic)
# Override with: DB_CONN_MAX_AGE=3600
DB_CONN_MAX_AGE = _env_int('DB_CONN_MAX_AGE', 600)  # 10 minutes default for production

DB_CONNECT_TIMEOUT_SECONDS = _env_int('DB_CONNECT_TIMEOUT_SECONDS', 30)
DB_STATEMENT_TIMEOUT_MS = _env_int('DB_STATEMENT_TIMEOUT_MS', 30000)
DB_IDLE_TX_TIMEOUT_MS = _env_int('DB_IDLE_TX_TIMEOUT_MS', 60000)
DB_IDLE_SESSION_TIMEOUT_MS = _env_int('DB_IDLE_SESSION_TIMEOUT_MS', 60000)

# Schema migrations can legitimately exceed runtime query timeouts.
if _is_schema_command():
    DB_STATEMENT_TIMEOUT_MS = _env_int('DB_MIGRATION_STATEMENT_TIMEOUT_MS', 0)
    DB_IDLE_TX_TIMEOUT_MS = _env_int('DB_MIGRATION_IDLE_TX_TIMEOUT_MS', 0)
    DB_IDLE_SESSION_TIMEOUT_MS = _env_int('DB_MIGRATION_IDLE_SESSION_TIMEOUT_MS', 0)

DATABASES = {
    'default': dj_database_url.config(
        default=DATABASE_URL,
        conn_max_age=DB_CONN_MAX_AGE,
        conn_health_checks=True,  # Verify connections are healthy before using
        ssl_require=True,
    )
}

# Connection pooling optimizations - Reduce connection exhaustion
_pg_options = ''
if DB_STATEMENT_TIMEOUT_MS > 0:
    _pg_options = _append_pg_option(_pg_options, f'-c statement_timeout={DB_STATEMENT_TIMEOUT_MS}')
if DB_IDLE_TX_TIMEOUT_MS > 0:
    _pg_options = _append_pg_option(_pg_options, f'-c idle_in_transaction_session_timeout={DB_IDLE_TX_TIMEOUT_MS}')
if DB_IDLE_SESSION_TIMEOUT_MS > 0:
    _pg_options = _append_pg_option(_pg_options, f'-c idle_session_timeout={DB_IDLE_SESSION_TIMEOUT_MS}')

DATABASES['default']['OPTIONS'] = {
    'connect_timeout': DB_CONNECT_TIMEOUT_SECONDS,
    'isolation_level': 1,  # READ_COMMITTED - Reduces lock memory, improves concurrency
}
if _pg_options:
    DATABASES['default']['OPTIONS']['options'] = _pg_options

# Database Connection Pool Monitoring
# Monitor these metrics to optimize pool size:
# - active_connections: Should stay below max_connections
# - idle_connections: Too many means pool is too large
# - connection_wait_time: If high, increase pool size
CONN_POOL_MIN_SIZE = _env_int('CONN_POOL_MIN_SIZE', 5)  # Minimum connections to maintain
CONN_POOL_MAX_SIZE = _env_int('CONN_POOL_MAX_SIZE', 20)  # Maximum connections allowed
CONN_POOL_TIMEOUT = _env_int('CONN_POOL_TIMEOUT', 30)   # Wait time for connection (seconds)

# =============================================================================
# SECURITY
# =============================================================================
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_SSL_REDIRECT = os.environ.get('SECURE_SSL_REDIRECT', 'True').lower() in ('1', 'true', 'yes')
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'SAMEORIGIN'

# Cookies
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = os.environ.get('SESSION_COOKIE_SAMESITE', 'Lax')
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_HTTPONLY = os.environ.get('CSRF_COOKIE_HTTPONLY', 'False').lower() in ('1', 'true', 'yes')
CSRF_COOKIE_SAMESITE = os.environ.get('CSRF_COOKIE_SAMESITE', 'Lax')

# HSTS
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# Content Security Policy
SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'

# =============================================================================
# CACHE - Redis
# =============================================================================
REDIS_URL = _normalize_rediss_url(os.environ.get('REDIS_URL'))
UPSTASH_REDIS_REST_URL = os.environ.get('UPSTASH_REDIS_REST_URL', '').strip()
UPSTASH_REDIS_REST_TOKEN = os.environ.get('UPSTASH_REDIS_REST_TOKEN', '').strip()
if REDIS_URL:
    REDIS_SOCKET_CONNECT_TIMEOUT = _env_int('REDIS_SOCKET_CONNECT_TIMEOUT', 10)
    REDIS_SOCKET_TIMEOUT = _env_int('REDIS_SOCKET_TIMEOUT', 10)
    REDIS_SOCKET_KEEPALIVE = _env_bool('REDIS_SOCKET_KEEPALIVE', True)
    REDIS_SOCKET_KEEPALIVE_IDLE = _env_int('REDIS_SOCKET_KEEPALIVE_IDLE', 0)
    REDIS_SOCKET_KEEPALIVE_INTERVAL = _env_int('REDIS_SOCKET_KEEPALIVE_INTERVAL', 0)
    REDIS_SOCKET_KEEPALIVE_COUNT = _env_int('REDIS_SOCKET_KEEPALIVE_COUNT', 0)
    REDIS_HEALTH_CHECK_INTERVAL = _env_int('REDIS_HEALTH_CHECK_INTERVAL', 30)
    REDIS_MAX_CONNECTIONS = _env_int('REDIS_MAX_CONNECTIONS', 50)
    REDIS_RETRY_ON_TIMEOUT = _env_bool('REDIS_RETRY_ON_TIMEOUT', True)
    REDIS_IGNORE_EXCEPTIONS = _env_bool('REDIS_IGNORE_EXCEPTIONS', True)
    REDIS_LOG_IGNORED_EXCEPTIONS = _env_bool('REDIS_LOG_IGNORED_EXCEPTIONS', True)
    REDIS_USE_BLOCKING_POOL = _env_bool('REDIS_USE_BLOCKING_POOL', True)
    REDIS_POOL_BLOCKING_TIMEOUT = _env_int('REDIS_POOL_BLOCKING_TIMEOUT', 5)

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

    _session_cache_timeout = _env_int('SESSION_CACHE_TIMEOUT_SECONDS', SESSION_COOKIE_AGE)
    CACHES = {
        'default': {
            'BACKEND': 'django_redis.cache.RedisCache',
            'LOCATION': REDIS_URL,
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
            'LOCATION': _redis_db_url(REDIS_URL, 1),
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
        }
    }
    
    # Prefer cache-only sessions to reduce DB connections when Redis is available.
    SESSION_ENGINE = os.environ.get('SESSION_ENGINE')
    if not SESSION_ENGINE:
        SESSION_ENGINE = (
            'django.contrib.sessions.backends.cache'
            if REDIS_URL
            else 'django.contrib.sessions.backends.cached_db'
        )
    SESSION_SAVE_EVERY_REQUEST = _env_bool('SESSION_SAVE_EVERY_REQUEST', False)
    SESSION_CACHE_ALIAS = 'sessions'

    WS_SESSION_AUTH_ENABLED = _env_bool('WS_SESSION_AUTH_ENABLED', False)

# =============================================================================
# CLOUDFLARE R2 STORAGE
# =============================================================================
if R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
    # Configure boto3 for R2
    AWS_ACCESS_KEY_ID = R2_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY = R2_SECRET_ACCESS_KEY
    AWS_STORAGE_BUCKET_NAME = R2_BUCKET_NAME
    AWS_S3_REGION_NAME = 'auto'
    AWS_S3_ENDPOINT_URL = f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com'
    AWS_S3_CUSTOM_DOMAIN = R2_CUSTOM_DOMAIN
    AWS_DEFAULT_ACL = None  # R2 doesn't support ACLs
    AWS_S3_OBJECT_PARAMETERS = {
        'CacheControl': 'max-age=31536000',  # 1 year cache
    }
    AWS_S3_SIGNATURE_VERSION = 's3v4'
    AWS_S3_ADDRESSING_STYLE = 'virtual'
    AWS_QUERYSTRING_AUTH = False  # Use public URLs
    
    # Use R2 for media files
    DEFAULT_FILE_STORAGE = 'storages.backends.s3.S3Storage'
    STORAGES = {
        **STORAGES,
        'default': {
            'BACKEND': 'storages.backends.s3.S3Storage',
        },
    }
    MEDIA_URL = f'https://{R2_CUSTOM_DOMAIN}/'

# =============================================================================
# STATIC FILES - Whitenoise
# =============================================================================
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
WHITENOISE_MAX_AGE = 31536000  # 1 year

# =============================================================================
# EMAIL - SMTP
# =============================================================================
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.sendgrid.net')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER', 'apikey')
EMAIL_HOST_PASSWORD = os.environ.get('SENDGRID_API_KEY') or os.environ.get('EMAIL_HOST_PASSWORD', '')

# =============================================================================
# CELERY - MEMORY OPTIMIZED FOR LOW RESOURCE ENVIRONMENTS
# =============================================================================
CELERY_BROKER_URL = _normalize_rediss_url(os.environ.get('CELERY_BROKER_URL', REDIS_URL))
CELERY_RESULT_BACKEND = _normalize_rediss_url(os.environ.get('CELERY_RESULT_BACKEND', REDIS_URL))
CELERY_TASK_ALWAYS_EAGER = False

# Connection retry behavior
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = _env_bool('CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP', True)
CELERY_BROKER_CONNECTION_MAX_RETRIES = _env_int('CELERY_BROKER_CONNECTION_MAX_RETRIES', 100)

# Worker memory optimization
CELERY_WORKER_PREFETCH_MULTIPLIER = _env_int('CELERY_WORKER_PREFETCH_MULTIPLIER', 1)  # One task at a time
CELERY_WORKER_MAX_TASKS_PER_CHILD = _env_int('CELERY_WORKER_MAX_TASKS_PER_CHILD', 500)  # Restart worker after 500 tasks
CELERY_TASK_ACKS_LATE = _env_bool('CELERY_TASK_ACKS_LATE', True)  # Acknowledge after completion
CELERY_TASK_REJECT_ON_WORKER_LOST = _env_bool('CELERY_TASK_REJECT_ON_WORKER_LOST', True)
CELERY_TASK_ACKS_ON_FAILURE_OR_TIMEOUT = _env_bool('CELERY_TASK_ACKS_ON_FAILURE_OR_TIMEOUT', True)

# Task timeouts and retries
CELERY_TASK_TIME_LIMIT = _env_int('CELERY_TASK_TIME_LIMIT', 600)  # 10 minutes hard limit
CELERY_TASK_SOFT_TIME_LIMIT = _env_int('CELERY_TASK_SOFT_TIME_LIMIT', 540)  # 9 minutes soft limit
CELERY_RESULT_EXPIRES = _env_int('CELERY_RESULT_EXPIRES', 3600)  # 1 hour
CELERY_TASK_DEFAULT_RETRY_DELAY = _env_int('CELERY_TASK_DEFAULT_RETRY_DELAY', 60)
CELERY_TASK_MAX_RETRIES = _env_int('CELERY_TASK_MAX_RETRIES', 3)

CELERY_BROKER_TRANSPORT_OPTIONS = {
    'socket_connect_timeout': REDIS_SOCKET_CONNECT_TIMEOUT,
    'socket_timeout': REDIS_SOCKET_TIMEOUT,
    'retry_on_timeout': REDIS_RETRY_ON_TIMEOUT,
}

# =============================================================================
# CHANNEL LAYERS - WebSockets with Redis
# =============================================================================
if REDIS_URL:
    channel_layers_redis_url = _normalize_rediss_url(
        os.environ.get('CHANNEL_LAYERS_REDIS_URL', _redis_db_url(REDIS_URL, 2))
    )
    # Use Redis for channel layers in production (required for WebSocket support)
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels_redis.core.RedisChannelLayer',
            'CONFIG': {
                'hosts': [
                    {
                        'address': channel_layers_redis_url,
                        'socket_connect_timeout': REDIS_SOCKET_CONNECT_TIMEOUT,
                        'socket_timeout': REDIS_SOCKET_TIMEOUT,
                        'health_check_interval': REDIS_HEALTH_CHECK_INTERVAL,
                        'retry_on_timeout': REDIS_RETRY_ON_TIMEOUT,
                        'socket_keepalive': REDIS_SOCKET_KEEPALIVE,
                        **({'socket_keepalive_options': _redis_keepalive_options} if _redis_keepalive_options else {}),
                    }
                ],
                'capacity': 1500,  # Max messages per channel
                'expiry': 10,  # Message expiry in seconds
            },
        },
    }

# =============================================================================
# LOGGING - Production Level
# =============================================================================
LOGGING['handlers']['console']['level'] = 'INFO'
LOGGING['handlers']['console']['filters'] = ['require_debug_false', 'request_id']
LOGGING['loggers']['django']['level'] = 'WARNING'
LOGGING['loggers']['bunoraa']['level'] = 'INFO'
LOGGING['root']['level'] = 'WARNING'

# Add Sentry logging if configured
SENTRY_DSN = os.environ.get('SENTRY_DSN')
if SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.django import DjangoIntegration
    from sentry_sdk.integrations.celery import CeleryIntegration
    from sentry_sdk.integrations.redis import RedisIntegration
    
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[
            DjangoIntegration(),
            CeleryIntegration(),
            RedisIntegration(),
        ],
        traces_sample_rate=0.1,
        profiles_sample_rate=0.1,
        send_default_pii=False,
        environment=ENVIRONMENT,
    )
