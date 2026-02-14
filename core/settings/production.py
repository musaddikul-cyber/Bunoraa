"""
Production Settings for Bunoraa
Optimized for performance, security, and scalability.
Uses PostgreSQL, Redis, Cloudflare R2 storage.
"""
import os
from urllib.parse import urlparse
import dj_database_url
from .base import *

# Sites framework defaults for production (override base if needed)
SITE_ID = int(os.environ.get('SITE_ID', '2'))
SITE_NAME = os.environ.get('SITE_NAME', 'Bunoraa')
SITE_DOMAIN = os.environ.get('SITE_DOMAIN', 'api.bunoraa.com')

# =============================================================================
# PRODUCTION CONFIGURATION
# =============================================================================
DEBUG = False
ENVIRONMENT = 'production'

# Explicitly disable prerendering in production regardless of environment vars.
PRERENDER_ENABLED = False
PRERENDER_PATHS = []

# Ensure SECRET_KEY is set
if not SECRET_KEY:
    raise ValueError("SECRET_KEY must be set in production environment")

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

_frontend_origin_raw = os.environ.get('NEXT_FRONTEND_ORIGIN', '').strip() or os.environ.get('NEXT_PUBLIC_SITE_URL', '').strip()
_frontend_origin = _normalize_origin(_frontend_origin_raw) if _frontend_origin_raw else ''
if _frontend_origin:
    if _frontend_origin not in CORS_ALLOWED_ORIGINS:
        CORS_ALLOWED_ORIGINS.append(_frontend_origin)
    if _frontend_origin not in CSRF_TRUSTED_ORIGINS:
        CSRF_TRUSTED_ORIGINS.append(_frontend_origin)

# =============================================================================
# DATABASE - PostgreSQL
# =============================================================================
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in production environment")

DATABASES = {
    'default': dj_database_url.config(
        default=DATABASE_URL,
        conn_max_age=300,
        conn_health_checks=True,
        ssl_require=True,
    )
}

# Connection pooling - Memory optimized for Render free tier
DATABASES['default']['CONN_MAX_AGE'] = 120  # Reduced from 300 - Close connections faster
DATABASES['default']['OPTIONS'] = {
    'connect_timeout': 10,
    'options': '-c statement_timeout=30000',
    'isolation_level': 1,  # READ_COMMITTED - Reduces lock memory (psycopg3 compatible)
}

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
REDIS_URL = os.environ.get('REDIS_URL')
if REDIS_URL:
    CACHES = {
        'default': {
            'BACKEND': 'django_redis.cache.RedisCache',
            'LOCATION': REDIS_URL,
            'OPTIONS': {
                'CLIENT_CLASS': 'django_redis.client.DefaultClient',
                'CONNECTION_POOL_KWARGS': {
                    'max_connections': 20,  # Reduced from 50 for free tier
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
            'LOCATION': f"{REDIS_URL}/1" if '/' not in REDIS_URL[-3:] else REDIS_URL.rsplit('/', 1)[0] + '/1',
            'OPTIONS': {
                'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            },
            'KEY_PREFIX': 'session',
            'TIMEOUT': 86400 * 30,  # 30 days
        }
    }
    
    # Use Redis for sessions
    SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
    SESSION_CACHE_ALIAS = 'sessions'

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
    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
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
EMAIL_HOST_PASSWORD = os.environ.get('SENDGRID_API_KEY', '')

# =============================================================================
# CELERY - MEMORY OPTIMIZED FOR LOW RESOURCE ENVIRONMENTS
# =============================================================================
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', REDIS_URL)
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', REDIS_URL)
CELERY_TASK_ALWAYS_EAGER = False

# Worker memory optimization
CELERY_WORKER_PREFETCH_MULTIPLIER = 1  # One task at a time
CELERY_WORKER_MAX_TASKS_PER_CHILD = 500  # Restart worker after 500 tasks
CELERY_TASK_ACKS_LATE = True  # Acknowledge after completion

# =============================================================================
# CHANNEL LAYERS - WebSockets with Redis
# =============================================================================
if REDIS_URL:
    # Use Redis for channel layers in production (required for WebSocket support)
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels_redis.core.RedisChannelLayer',
            'CONFIG': {
                'hosts': [os.environ.get('CHANNEL_LAYERS_REDIS_URL', REDIS_URL)],
                'capacity': 1500,  # Max messages per channel
                'expiry': 10,  # Message expiry in seconds
            },
        },
    }

# =============================================================================
# LOGGING - Production Level
# =============================================================================
LOGGING['handlers']['console']['level'] = 'INFO'
LOGGING['handlers']['console']['filters'] = ['require_debug_false']
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
