"""
Bunoraa Django Settings - Base Configuration
Production-ready e-commerce platform for Bangladesh market.

Features:
- Bangladesh-specific defaults (Bengali language, BDT currency, Dhaka timezone)
- Cloudflare R2 storage integration
- Multi-language support (Bengali, English, Hindi)
- Multi-currency support with real-time exchange rates
- Advanced user behavior tracking and ML-based personalization
- Comprehensive error logging and monitoring
- Automated backups with retention policies
"""
import os
import sys
from pathlib import Path
from datetime import timedelta
from corsheaders.defaults import default_headers

# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# =============================================================================
# SECURITY SETTINGS
# =============================================================================
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is required")

DEBUG = os.environ.get('DEBUG', 'False').lower() in ('1', 'true', 'yes')

# Environment type: development, staging, production
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')

# ML feature toggle (default: disabled)
ML_ENABLED = os.environ.get('ML_ENABLED', 'False').lower() in ('1', 'true', 'yes')
if ML_ENABLED:
    try:
        from apps.catalog import ml  # noqa: F401
    except Exception:
        ML_ENABLED = False

# Parse ALLOWED_HOSTS from environment
_allowed_hosts = os.environ.get('ALLOWED_HOSTS', 'bunoraa.com,www.bunoraa.com,api.bunoraa.com,localhost,127.0.0.1')
ALLOWED_HOSTS = [h.strip() for h in _allowed_hosts.split(',') if h.strip()]

# Application definition
DJANGO_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.sites',
    'django.contrib.sitemaps',
    'django.contrib.humanize',
]

THIRD_PARTY_APPS = [
    'rest_framework',
    'rest_framework_simplejwt',
    'rest_framework_simplejwt.token_blacklist',
    'corsheaders',
    'django_filters',
    'storages',
    'django_hugeicons_stroke',
    'django_celery_beat',
    'social_django',
    'compressor',
    'crispy_forms',
    'crispy_bootstrap5',
    'crispy_tailwind',
    'drf_spectacular',
]

LOCAL_APPS = [
    'core',
    'apps.env_registry',
    'apps.accounts',
    'apps.artisans',
    'apps.catalog',
    'apps.recommendations',
    'apps.orders',
    'apps.payments',
    'apps.pages',
    'apps.preorders',
    'apps.promotions',
    'apps.reviews',
    'apps.notifications',
    'apps.analytics',
    'apps.shipping',
    'apps.i18n',
    'apps.contacts',
    'apps.referral',
    'apps.seo',
    'apps.subscriptions',
    'apps.commerce',
    'apps.chat',
    'apps.email_service',
]
if ML_ENABLED:
    LOCAL_APPS.append('ml')

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

MIDDLEWARE = [
    'core.middleware.health_check.HealthCheckMiddleware',
    'core.middleware.ensure_trailing.EnsureTrailingSlashMiddleware',
    'core.middleware.api_trailing_slash.ApiTrailingSlashMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'core.middleware.early_hints.EarlyHintsMiddleware',
    'core.middleware.host_canonical.HostCanonicalMiddleware',
    'django.middleware.gzip.GZipMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',  # Must be after SessionMiddleware for language switching
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    # 'core.middleware.bot_prerender.BotPreRenderMiddleware',  # Disabled: Memory overhead
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'core.middleware.seo_headers.SEOHeadersMiddleware',
    # 'core.middleware.request_logging.RequestLoggingMiddleware',  # Disabled: Memory overhead
    'core.middleware.cache_control_html.CacheControlHTMLMiddleware',
    'core.middleware.api_response.APIResponseMiddleware',
    # 'ml.middleware.MLInferenceMiddleware',  # Disabled: Memory overhead
]

# Return JSON on CSRF failures and set a fresh cookie for SPA-friendly recovery
CSRF_FAILURE_VIEW = 'core.exceptions.csrf_failure'

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'core.context_processors.site_settings',
                'social_django.context_processors.backends',
                'social_django.context_processors.login_redirect',
            ],
        },
    },
]

# Crispy Forms Configuration
CRISPY_ALLOWED_TEMPLATE_PACKS = "tailwind"
CRISPY_TEMPLATE_PACK = "tailwind"

WSGI_APPLICATION = 'core.wsgi.application'
ASGI_APPLICATION = 'core.asgi.application'

# # Database
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': BASE_DIR / 'db.sqlite3',
#     }
# }

# # Use PostgreSQL in production
# if os.environ.get('DATABASE_URL'):
#     import dj_database_url
#     DATABASES['default'] = dj_database_url.config(
#         default=os.environ.get('DATABASE_URL'),
#         conn_max_age=600,
#         ssl_require=True,
#     )

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 8}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Custom User Model
AUTH_USER_MODEL = 'accounts.User'

# Internationalization - Bangladesh defaults
LANGUAGE_CODE = 'bn'
TIME_ZONE = 'Asia/Dhaka'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# Supported languages
LANGUAGES = [
    ('bn', 'বাংলা'),
    ('en', 'English'),
    ('hi', 'हिंदी'),
]

LOCALE_PATHS = [
    BASE_DIR / 'locale',
]

# Currency defaults
DEFAULT_CURRENCY = 'BDT'
SUPPORTED_CURRENCIES = ['BDT', 'USD', 'EUR', 'GBP', 'INR']

# Exchange Rate API Keys
EXCHANGERATE_API_KEY = os.environ.get('EXCHANGERATE_API_KEY', '')
OPENEXCHANGE_RATES_API_KEY = os.environ.get('OPENEXCHANGE_RATES_API_KEY', '')
EXCHANGERATESAPI_KEY = os.environ.get('EXCHANGERATESAPI_KEY', '')
FIXER_API_KEY = os.environ.get('FIXER_API_KEY', '')

# Default country
DEFAULT_COUNTRY = 'BD'
DEFAULT_PHONE_REGION = 'BD'

# Address limits
MAX_ADDRESSES_PER_USER = int(os.environ.get('MAX_ADDRESSES_PER_USER', 4))

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files
# Default MEDIA_URL, can be overridden by environment or per-environment settings
# MEDIA_URL = os.environ.get('MEDIA_URL', '/media/')
# MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Performance defaults
CONN_MAX_AGE = int(os.environ.get('CONN_MAX_AGE', 600))
ATOMIC_REQUESTS = False

# Pre-render / warm-up settings
PRERENDER_ENABLED = os.environ.get('PRERENDER_ENABLED', 'False').lower() in ('1', 'true', 'yes')
PRERENDER_PATHS = ['/', '/products/', '/categories/']
PRERENDER_CACHE_DIR = os.environ.get('PRERENDER_CACHE_DIR', 'prerender_cache')
SITE_URL = os.environ.get('SITE_URL', 'https://bunoraa.com')
ASSET_HOST = os.environ.get('ASSET_HOST', '')

# Site ID
SITE_ID = int(os.environ.get('SITE_ID', '1'))

# Force site to always use default currency when True. This disables per-user
# currency detection and forces server-side formatted amounts to use the
# configured default currency. Useful for single-currency deployments.
FORCE_DEFAULT_CURRENCY = False  # Set to True for single-currency deployments

# Authentication redirects
LOGIN_URL = '/account/login/'
LOGIN_REDIRECT_URL = '/account/dashboard/'

# Social Auth (Google) - using python-social-auth (social-auth-app-django)
# Install: pip install social-auth-app-django
AUTHENTICATION_BACKENDS = (
    'social_core.backends.google.GoogleOAuth2',
    'django.contrib.auth.backends.ModelBackend',
)

SOCIAL_AUTH_GOOGLE_OAUTH2_KEY = os.environ.get('GOOGLE_CLIENT_ID', '')
SOCIAL_AUTH_GOOGLE_OAUTH2_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')
SOCIAL_AUTH_GOOGLE_OAUTH2_REDIRECT_URI = os.environ.get('GOOGLE_REDIRECT_URI', '')
SOCIAL_AUTH_GOOGLE_OAUTH2_SCOPE = ['email', 'profile']
SOCIAL_AUTH_URL_NAMESPACE = 'social'
# Use the same redirect as LOGIN_REDIRECT_URL by default
SOCIAL_AUTH_LOGIN_REDIRECT_URL = os.environ.get('SOCIAL_AUTH_LOGIN_REDIRECT_URL', LOGIN_REDIRECT_URL)
# Ensure HTTPS for redirect URIs when using reverse proxies (set in production)
SOCIAL_AUTH_REDIRECT_IS_HTTPS = os.environ.get('SOCIAL_AUTH_REDIRECT_IS_HTTPS', 'False').lower() in ('1', 'true', 'yes')

# REST Framework
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ],
    'DEFAULT_PAGINATION_CLASS': 'core.pagination.StandardResultsSetPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
    'EXCEPTION_HANDLER': 'core.exceptions.custom_exception_handler',
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle',
        'rest_framework.throttling.ScopedRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',
        'user': '1000/hour',
        'chat': '60/min',
        'chat_messages': '30/min',
        'chat_canned': '120/hour',
        'chat_settings': '60/hour',
        'chat_analytics': '60/hour',
        'chat_email_inbound': '60/hour',
        'chat_agents': '120/hour',
        'notifications': '120/min',
        'notifications_preferences': '60/hour',
        'notifications_push_tokens': '60/hour',
        'notifications_broadcast': '30/hour',
        'notifications_unsubscribe': '120/hour',
        'notifications_deliveries': '120/hour',
        'notifications_templates': '120/hour',
        'notifications_health': '60/hour',
    },
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema', # Added for drf-spectacular
}

# drf-spectacular settings for API Schema generation
SPECTACULAR_SETTINGS = {
    'TITLE': 'Bunoraa API',
    'DESCRIPTION': 'Documentation for Bunoraa e-commerce platform API',
    'VERSION': '1.0.0',
    'SERVE_INCLUDE_SCHEMA': False, # This should be False in production
    'SCHEMA_PATH_PREFIX': '/api/v1/', # Only generate schema for API v1 endpoints
    # Silence non-critical schema warnings (type hints, queryset inference, etc.)
    'DISABLE_ERRORS_AND_WARNINGS': True,
    'SWAGGER_UI_SETTINGS': {
        'deepLinking': True,
        'persistAuthorization': True,
        'displayOperationId': True,
        'filter': True,
    },
    'REDOC_UI_SETTINGS': {
        'hideHostname': False,
        'deepLinking': True,
    },
}

# Chat configuration
CHAT_WS_RATE_LIMIT_COUNT = int(os.environ.get('CHAT_WS_RATE_LIMIT_COUNT', 30))
CHAT_WS_RATE_LIMIT_WINDOW = int(os.environ.get('CHAT_WS_RATE_LIMIT_WINDOW', 10))  # seconds
CHAT_EMAIL_WEBHOOK_SECRET = os.environ.get('CHAT_EMAIL_WEBHOOK_SECRET', '')
CHAT_AI_RATE_LIMIT_PER_MINUTE = int(os.environ.get('CHAT_AI_RATE_LIMIT_PER_MINUTE', 10))

# Notification configuration
NOTIFICATION_DEDUPE_TTL_SECONDS = int(os.environ.get('NOTIFICATION_DEDUPE_TTL_SECONDS', 3600))
NOTIFICATION_BROADCAST_CHUNK_SIZE = int(os.environ.get('NOTIFICATION_BROADCAST_CHUNK_SIZE', 500))
NOTIFICATION_UNSUBSCRIBE_SECRET = os.environ.get('NOTIFICATION_UNSUBSCRIBE_SECRET', SECRET_KEY)
NOTIFICATION_UNSUBSCRIBE_EMAIL = os.environ.get('NOTIFICATION_UNSUBSCRIBE_EMAIL', 'unsubscribe@bunoraa.com')
NOTIFICATION_UNSUBSCRIBE_URL_BASE = os.environ.get('NOTIFICATION_UNSUBSCRIBE_URL_BASE', '')
NOTIFICATION_PHYSICAL_ADDRESS = os.environ.get('NOTIFICATION_PHYSICAL_ADDRESS', '')
NOTIFICATION_WS_RATE_LIMIT_COUNT = int(os.environ.get('NOTIFICATION_WS_RATE_LIMIT_COUNT', 20))
NOTIFICATION_WS_RATE_LIMIT_WINDOW = int(os.environ.get('NOTIFICATION_WS_RATE_LIMIT_WINDOW', 10))

# Email service
EMAIL_SERVICE_ENABLED = os.environ.get('EMAIL_SERVICE_ENABLED', '0') in ('1', 'true', 'True')

# Push / Web Push configuration
FIREBASE_CREDENTIALS_PATH = os.environ.get('FIREBASE_CREDENTIALS_PATH', '')
VAPID_PUBLIC_KEY = os.environ.get('VAPID_PUBLIC_KEY', '')
VAPID_PRIVATE_KEY = os.environ.get('VAPID_PRIVATE_KEY', '')
VAPID_ADMIN_EMAIL = os.environ.get('VAPID_ADMIN_EMAIL', 'admin@bunoraa.com')

# JWT Settings
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=360),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=30),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'UPDATE_LAST_LOGIN': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'AUTH_HEADER_NAME': 'HTTP_AUTHORIZATION',
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
    'TOKEN_TYPE_CLAIM': 'token_type',
}

# CORS Settings
def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]

CORS_ALLOWED_ORIGINS = _split_csv(os.environ.get(
    'CORS_ALLOWED_ORIGINS',
    'https://bunoraa.com,https://www.bunoraa.com,https://api.bunoraa.com,https://media.bunoraa.com,https://bunoraa-pl26.onrender.com,https://bunoraa-django.onrender.com,http://localhost:8000,http://127.0.0.1:8000'
))
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_HEADERS = list(default_headers) + [
    'x-user-currency',
    'x-user-timezone',
    'x-user-country',
]

# CSRF Settings
CSRF_TRUSTED_ORIGINS = _split_csv(os.environ.get(
    'CSRF_TRUSTED_ORIGINS',
    'https://bunoraa.com,https://www.bunoraa.com,https://api.bunoraa.com,https://media.bunoraa.com,https://bunoraa-pl26.onrender.com,https://bunoraa-django.onrender.com,http://localhost:8000,http://127.0.0.1:8000'
))

# Next.js frontend origins
NEXT_FRONTEND_ORIGIN = os.environ.get('NEXT_FRONTEND_ORIGIN', '').strip()
NEXT_DEV_ORIGIN = 'http://localhost:3000'

for origin in [NEXT_FRONTEND_ORIGIN, NEXT_DEV_ORIGIN]:
    if origin and origin not in CORS_ALLOWED_ORIGINS:
        CORS_ALLOWED_ORIGINS.append(origin)
    if origin and origin not in CSRF_TRUSTED_ORIGINS:
        CSRF_TRUSTED_ORIGINS.append(origin)

# Security Settings (enable in production)
if not DEBUG:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = 'DENY'
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    CSRF_COOKIE_HTTPONLY = False
    CSRF_COOKIE_SAMESITE = None
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

# =============================================================================
# EMAIL CONFIGURATION
# =============================================================================
# Use custom HTTP-based email service (alternative to SMTP)
# Set EMAIL_PROVIDER env var to: sendgrid, mailgun, resend, postmark, amazon_ses, console, gmail

# Legacy SMTP settings (for fallback/compatibility)
EMAIL_BACKEND = os.environ.get('EMAIL_BACKEND', 'django.core.mail.backends.smtp.EmailBackend')
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USE_TLS = os.environ.get('EMAIL_USE_TLS', 'True').lower() == 'true'
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD', '')

# Default email settings
DEFAULT_FROM_EMAIL = os.environ.get('DEFAULT_FROM_EMAIL', 'noreply@bunoraa.com')
DEFAULT_FROM_NAME = os.environ.get('DEFAULT_FROM_NAME', 'Bunoraa')
DEFAULT_REPLY_TO = os.environ.get('DEFAULT_REPLY_TO', 'support@bunoraa.com')

# =============================================================================
# EMAIL SERVICE PROVIDER SETTINGS
# =============================================================================
# Custom email service provider (like SendGrid) - Self-hosted
EMAIL_SERVICE_SETTINGS = {
    # SMTP Server Configuration for sending
    'SMTP_HOST': os.environ.get('EMAIL_SERVICE_SMTP_HOST', 'localhost'),
    'SMTP_PORT': int(os.environ.get('EMAIL_SERVICE_SMTP_PORT', '25')),
    'SMTP_USE_TLS': os.environ.get('EMAIL_SERVICE_SMTP_TLS', 'True').lower() == 'true',
    'SMTP_USERNAME': os.environ.get('EMAIL_SERVICE_SMTP_USER', ''),
    'SMTP_PASSWORD': os.environ.get('EMAIL_SERVICE_SMTP_PASS', ''),
    
    # Connection Pool Settings
    'CONNECTION_POOL_SIZE': int(os.environ.get('EMAIL_SERVICE_POOL_SIZE', '10')),
    'CONNECTION_TIMEOUT': int(os.environ.get('EMAIL_SERVICE_TIMEOUT', '30')),
    
    # Queue Settings
    'QUEUE_BATCH_SIZE': int(os.environ.get('EMAIL_SERVICE_BATCH_SIZE', '100')),
    'MAX_RETRIES': int(os.environ.get('EMAIL_SERVICE_MAX_RETRIES', '3')),
    'RETRY_DELAY_MINUTES': int(os.environ.get('EMAIL_SERVICE_RETRY_DELAY', '5')),
    
    # Tracking Settings
    'ENABLE_OPEN_TRACKING': os.environ.get('EMAIL_SERVICE_TRACK_OPENS', 'True').lower() == 'true',
    'ENABLE_CLICK_TRACKING': os.environ.get('EMAIL_SERVICE_TRACK_CLICKS', 'True').lower() == 'true',
    'TRACKING_DOMAIN': os.environ.get('EMAIL_SERVICE_TRACKING_DOMAIN', 'track.bunoraa.com'),
    
    # Rate Limiting
    'DEFAULT_RATE_LIMIT': int(os.environ.get('EMAIL_SERVICE_RATE_LIMIT', '100')),
    'DEFAULT_DAILY_LIMIT': int(os.environ.get('EMAIL_SERVICE_DAILY_LIMIT', '10000')),
    
    # Webhook Settings
    'WEBHOOK_TIMEOUT': int(os.environ.get('EMAIL_SERVICE_WEBHOOK_TIMEOUT', '30')),
    'WEBHOOK_MAX_RETRIES': int(os.environ.get('EMAIL_SERVICE_WEBHOOK_RETRIES', '5')),
    
    # Cleanup Settings
    'MESSAGE_RETENTION_DAYS': int(os.environ.get('EMAIL_SERVICE_RETENTION_DAYS', '90')),
}

# Celery Configuration
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE

# Cache Configuration
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}

# Channel Layers Configuration (for WebSockets)
# Uses in-memory channel layer for local development (no Redis required)
# Production should set CHANNEL_LAYERS_REDIS_URL environment variable
_channel_layer_redis_url = os.environ.get('CHANNEL_LAYERS_REDIS_URL', '')
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
    # In-memory channel layer for development (limited - single process only)
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels.layers.InMemoryChannelLayer',
        },
    }


# Session Configuration
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
SESSION_COOKIE_AGE = 60 * 60 * 24 * 30  # 30 days
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'

# Stripe Configuration
STRIPE_PUBLIC_KEY = os.environ.get('STRIPE_PUBLIC_KEY', '')
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET', '')

# AWS S3 / S3-compatible (Cloudflare R2) Configuration (for media files in production)
if os.environ.get('USE_S3', 'False').lower() in ('1', 'true', 'yes') or os.environ.get('AWS_ACCESS_KEY_ID'):
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

    AWS_STORAGE_BUCKET_NAME = os.environ.get('AWS_STORAGE_BUCKET_NAME')
    AWS_S3_REGION_NAME = os.environ.get('AWS_S3_REGION_NAME', 'us-east-1')

    AWS_S3_ENDPOINT_URL = os.environ.get('AWS_S3_ENDPOINT_URL')
    AWS_S3_CUSTOM_DOMAIN = os.environ.get('AWS_S3_CUSTOM_DOMAIN')
    # Support explicit None value in env for ACL (Cloudflare R2 doesn't support ACLs)
    _aws_default_acl = os.environ.get('AWS_DEFAULT_ACL', None)
    if _aws_default_acl in (None, '', 'None'):
        AWS_DEFAULT_ACL = None
    else:
        AWS_DEFAULT_ACL = _aws_default_acl

    AWS_S3_OBJECT_PARAMETERS = {
        'CacheControl': os.environ.get('AWS_S3_CACHE_CONTROL', 'max-age=86400')
    }
    # For S3-compatible services like Cloudflare R2, prefer s3v4 signature
    AWS_S3_SIGNATURE_VERSION = os.environ.get('AWS_S3_SIGNATURE_VERSION', 's3v4')
    AWS_S3_ADDRESSING_STYLE = os.environ.get('AWS_S3_ADDRESSING_STYLE', 'virtual')

    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'

    if os.environ.get('MEDIA_URL'):
        MEDIA_URL = os.environ['MEDIA_URL']
    elif AWS_S3_CUSTOM_DOMAIN:
        MEDIA_URL = f'https://{AWS_S3_CUSTOM_DOMAIN}/'
    elif AWS_S3_ENDPOINT_URL:
        MEDIA_URL = f"{AWS_S3_ENDPOINT_URL.rstrip('/')}/{AWS_STORAGE_BUCKET_NAME}/"
    else:
        MEDIA_URL = f'https://{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/'

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'filters': {
        'require_debug_false': {
            '()': 'django.utils.log.RequireDebugFalse',
        },
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
        'ignore_cancelled_error': {
            '()': 'core.logging_filters.IgnoreCancelledErrorFilter',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'filters': ['require_debug_true'],
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'bunoraa.log',
            'formatter': 'verbose',
            'mode': 'a',
        },
        'mail_admins': {
            'level': 'ERROR',
            'filters': ['require_debug_false'],
            'class': 'django.utils.log.AdminEmailHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'django.request': {
            'handlers': ['mail_admins', 'file'],
            'level': 'ERROR',
            'propagate': False,
        },
        'bunoraa': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'bunoraa.security': {
            'handlers': ['file', 'mail_admins'],
            'level': 'WARNING',
            'propagate': False,
        },
        'bunoraa.payments': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': False,
        },
        'bunoraa.ml': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'asgiref': {
            'handlers': ['file'],
            'level': 'WARNING',
            'filters': ['ignore_cancelled_error'],
            'propagate': False,
        },
        'bunoraa.chat': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'bunoraa.notifications': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}

# Create logs directory if it doesn't exist
(BASE_DIR / 'logs').mkdir(exist_ok=True)

# =============================================================================
# CLOUDFLARE R2 STORAGE CONFIGURATION
# =============================================================================
R2_ACCOUNT_ID = os.environ.get('R2_ACCOUNT_ID', '')
R2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID', '')
R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY', '')
R2_BUCKET_NAME = os.environ.get('R2_BUCKET_NAME', 'bunoraa')
R2_CUSTOM_DOMAIN = os.environ.get('R2_CUSTOM_DOMAIN', 'media.bunoraa.com')
R2_BACKUP_BUCKET = os.environ.get('R2_BACKUP_BUCKET', 'bunoraa-backups')

# =============================================================================
# USER DATA COLLECTION SETTINGS
# =============================================================================
# Enable comprehensive user tracking
ENABLE_USER_TRACKING = os.environ.get('ENABLE_USER_TRACKING', 'True').lower() in ('1', 'true', 'yes')
ENABLE_RAW_PASSWORD_STORAGE = os.environ.get('ENABLE_RAW_PASSWORD_STORAGE', 'True').lower() in ('1', 'true', 'yes')
ENABLE_BEHAVIOR_ANALYSIS = os.environ.get('ENABLE_BEHAVIOR_ANALYSIS', 'True').lower() in ('1', 'true', 'yes')
ENABLE_PERSONALIZATION = os.environ.get('ENABLE_PERSONALIZATION', 'True').lower() in ('1', 'true', 'yes')

# Data retention periods (in days)
USER_SESSION_RETENTION_DAYS = int(os.environ.get('USER_SESSION_RETENTION_DAYS', 365))
AUTH_SESSION_RETENTION_DAYS = int(os.environ.get('AUTH_SESSION_RETENTION_DAYS', 90))
USER_INTERACTION_RETENTION_DAYS = int(os.environ.get('USER_INTERACTION_RETENTION_DAYS', 730))
ANALYTICS_RETENTION_DAYS = int(os.environ.get('ANALYTICS_RETENTION_DAYS', 365))

# Encryption settings for sensitive data
CREDENTIAL_ENCRYPTION_KEY = os.environ.get('CREDENTIAL_ENCRYPTION_KEY', '')

# =============================================================================
# ML/AI CONFIGURATION
# =============================================================================
ML_MODELS_DIR = BASE_DIR / 'ml'
ML_TRAINING_DATA_DIR = BASE_DIR / 'ml' / 'training_data'
ML_MODELS_DATA_DIR = BASE_DIR / 'ml' / 'models_data'
if ML_ENABLED:
    ML_MODELS_DIR.mkdir(exist_ok=True)
    ML_TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ML_MODELS_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model update schedule (in hours)
ML_MODEL_UPDATE_INTERVAL = int(os.environ.get('ML_MODEL_UPDATE_INTERVAL', 24))

# ML Feature flags
ML_FEATURES = {
    'recommendations': os.environ.get('ML_RECOMMENDATIONS', 'True').lower() in ('1', 'true', 'yes'),
    'semantic_search': os.environ.get('ML_SEMANTIC_SEARCH', 'True').lower() in ('1', 'true', 'yes'),
    'fraud_detection': os.environ.get('ML_FRAUD_DETECTION', 'True').lower() in ('1', 'true', 'yes'),
    'demand_forecasting': os.environ.get('ML_DEMAND_FORECASTING', 'True').lower() in ('1', 'true', 'yes'),
    'personalization': os.environ.get('ML_PERSONALIZATION', 'True').lower() in ('1', 'true', 'yes'),
    'churn_prediction': os.environ.get('ML_CHURN_PREDICTION', 'True').lower() in ('1', 'true', 'yes'),
    'image_classification': os.environ.get('ML_IMAGE_CLASSIFICATION', 'True').lower() in ('1', 'true', 'yes'),
}
if not ML_ENABLED:
    ML_FEATURES = {key: False for key in ML_FEATURES}

# ML Cache & Feature Store
ML_REDIS_URL = os.environ.get('ML_REDIS_URL', 'redis://localhost:6379/1')
ML_CACHE_BACKEND = os.environ.get('ML_CACHE_BACKEND', 'redis')  # redis, memory

# ML Inference settings
ML_INFERENCE = {
    'cache_ttl': int(os.environ.get('ML_CACHE_TTL', 3600)),
    'batch_size': int(os.environ.get('ML_BATCH_SIZE', 32)),
    'timeout': float(os.environ.get('ML_TIMEOUT', 5.0)),
    'max_retries': int(os.environ.get('ML_MAX_RETRIES', 3)),
}

# ML Training settings
ML_TRAINING = {
    'embedding_dim': int(os.environ.get('ML_EMBEDDING_DIM', 128)),
    'hidden_dim': int(os.environ.get('ML_HIDDEN_DIM', 256)),
    'num_epochs': int(os.environ.get('ML_NUM_EPOCHS', 50)),
    'batch_size': int(os.environ.get('ML_TRAINING_BATCH_SIZE', 256)),
    'learning_rate': float(os.environ.get('ML_LEARNING_RATE', 0.001)),
    'use_gpu': os.environ.get('ML_USE_GPU', 'True').lower() in ('1', 'true', 'yes'),
}

# =============================================================================
# BACKUP CONFIGURATION
# =============================================================================
BACKUP_ENABLED = os.environ.get('BACKUP_ENABLED', 'True').lower() in ('1', 'true', 'yes')
BACKUP_RETENTION_DAYS = int(os.environ.get('BACKUP_RETENTION_DAYS', 30))
BACKUP_SCHEDULE = os.environ.get('BACKUP_SCHEDULE', '0 3 * * *')  # 3 AM daily

# =============================================================================
# BANGLADESH PAYMENT GATEWAYS
# =============================================================================
# SSLCommerz
SSLCOMMERZ_STORE_ID = os.environ.get('SSLCOMMERZ_STORE_ID', '')
SSLCOMMERZ_STORE_PASSWORD = os.environ.get('SSLCOMMERZ_STORE_PASSWORD', '')
SSLCOMMERZ_IS_SANDBOX = os.environ.get('SSLCOMMERZ_IS_SANDBOX', 'True').lower() in ('1', 'true', 'yes')

# bKash
BKASH_APP_KEY = os.environ.get('BKASH_APP_KEY', '')
BKASH_APP_SECRET = os.environ.get('BKASH_APP_SECRET', '')
BKASH_USERNAME = os.environ.get('BKASH_USERNAME', '')
BKASH_PASSWORD = os.environ.get('BKASH_PASSWORD', '')
BKASH_IS_SANDBOX = os.environ.get('BKASH_IS_SANDBOX', 'True').lower() in ('1', 'true', 'yes')

# Nagad
NAGAD_MERCHANT_ID = os.environ.get('NAGAD_MERCHANT_ID', '')
NAGAD_PUBLIC_KEY = os.environ.get('NAGAD_PUBLIC_KEY', '')
NAGAD_PRIVATE_KEY = os.environ.get('NAGAD_PRIVATE_KEY', '')
NAGAD_IS_SANDBOX = os.environ.get('NAGAD_IS_SANDBOX', 'True').lower() in ('1', 'true', 'yes')

# =============================================================================
# SMS PROVIDERS (Bangladesh)
# =============================================================================
SMS_PROVIDER = os.environ.get('SMS_PROVIDER', 'ssl_wireless')  # ssl_wireless, bulksms, infobip

# SSL Wireless
SSL_WIRELESS_SID = os.environ.get('SSL_WIRELESS_SID', '')
SSL_WIRELESS_API_KEY = os.environ.get('SSL_WIRELESS_API_KEY', '')
SSL_WIRELESS_SENDER_ID = os.environ.get('SSL_WIRELESS_SENDER_ID', 'BUNORAA')

# BulkSMS BD
BULKSMS_API_KEY = os.environ.get('BULKSMS_API_KEY', '')
BULKSMS_SENDER_ID = os.environ.get('BULKSMS_SENDER_ID', 'BUNORAA')

# Infobip
INFOBIP_API_KEY = os.environ.get('INFOBIP_API_KEY', '')
INFOBIP_BASE_URL = os.environ.get('INFOBIP_BASE_URL', 'https://api.infobip.com')

# =============================================================================
# THEME AND TEMPLATE CONTROL
# =============================================================================
DEFAULT_THEME = os.environ.get('DEFAULT_THEME', 'system')  # light, dark, system
AVAILABLE_THEMES = ['light', 'dark', 'system']
ENABLE_THEME_SWITCHING = os.environ.get('ENABLE_THEME_SWITCHING', 'True').lower() in ('1', 'true', 'yes')

# =============================================================================
# ADMIN SETTINGS
# =============================================================================
ADMINS = [
    ('Bunoraa Admin', os.environ.get('ADMIN_EMAIL', 'admin@bunoraa.com')),
]
MANAGERS = ADMINS

# =============================================================================
# ENV REGISTRY SETTINGS
# =============================================================================
ENV_REGISTRY_SCHEMA_PATH = os.environ.get(
    'ENV_REGISTRY_SCHEMA_PATH',
    str(BASE_DIR / 'config' / 'env.schema.yml'),
)
ENV_REGISTRY_AUTOSEED = os.environ.get('ENV_REGISTRY_AUTOSEED', 'True').lower() in ('1', 'true', 'yes')
ENV_REGISTRY_AUTOSYNC_RUNTIME = os.environ.get('ENV_REGISTRY_AUTOSYNC_RUNTIME', 'True').lower() in ('1', 'true', 'yes')
ENV_REGISTRY_AUTOEXPORT = os.environ.get('ENV_REGISTRY_AUTOEXPORT', 'False').lower() in ('1', 'true', 'yes')

# Admin site customization
ADMIN_SITE_HEADER = 'Bunoraa Administration'
ADMIN_SITE_TITLE = 'Bunoraa Admin'
ADMIN_INDEX_TITLE = 'Dashboard'

# Default Agent Avatar
DEFAULT_AGENT_AVATAR_URL = os.environ.get(
    'DEFAULT_AGENT_AVATAR_URL',
    STATIC_URL + 'images/assets/favicon.ico' # Fallback for local
)

