"""
Monitoring, Logging, and Health Check Configuration
Production-ready monitoring with Sentry, structured logging, and health endpoints.
"""
import os

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{asctime}] [{levelname}] [{name}] [{module}.{funcName}:{lineno}] {message}',
            'style': '{',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'structured': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)s %(message)s %(correlation_id)s',
        },
        'simple': {
            'format': '[{levelname}] {message}',
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
        'request_id': {
            '()': 'core.logging_filters.RequestIdFilter',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            'filters': ['request_id'],
        },
        'console_json': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'structured',
            'filters': ['request_id'],
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.environ.get('LOG_FILE_PATH', '/var/log/bunoraa/app.log'),
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 10,
            'formatter': 'structured',
            'filters': ['request_id'],
        },
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.environ.get('ERROR_LOG_FILE_PATH', '/var/log/bunoraa/error.log'),
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 10,
            'formatter': 'structured',
            'filters': ['request_id'],
        },
        'mail_admins': {
            'level': 'ERROR',
            'class': 'django.utils.log.AdminEmailHandler',
            'formatter': 'verbose',
            'filters': ['require_debug_false'],
            'include_html': True,
        },
        'null': {
            'class': 'logging.NullHandler',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'django.request': {
            'handlers': ['console', 'error_file', 'mail_admins'],
            'level': 'ERROR',
            'propagate': False,
        },
        'django.server': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'django.db.backends': {
            'handlers': ['console'],
            'level': 'WARNING',  # Set to DEBUG to log all queries
            'propagate': False,
        },
        'bunoraa': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'INFO',
            'propagate': False,
        },
        'apps': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'INFO',
            'propagate': False,
        },
        'celery': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'redis': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}

# Use JSON logging in production
if os.environ.get('ENVIRONMENT') == 'production':
    LOGGING['handlers']['console'] = LOGGING['handlers']['console_json']

# =============================================================================
# SENTRY CONFIGURATION (Error Tracking)
# =============================================================================

SENTRY_DSN = os.environ.get('SENTRY_DSN')
SENTRY_ENABLED = bool(SENTRY_DSN)

if SENTRY_ENABLED:
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
        traces_sample_rate=float(os.environ.get('SENTRY_TRACES_SAMPLE_RATE', '0.1')),
        profiles_sample_rate=float(os.environ.get('SENTRY_PROFILES_SAMPLE_RATE', '0.01')),
        environment=os.environ.get('ENVIRONMENT', 'production'),
        release=os.environ.get('GIT_COMMIT_SHA', 'unknown'),
        send_default_pii=False,  # Don't send user PII by default
    )

# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

HEALTH_CHECK_CONFIG = {
    'DISALLOWED_NAMESPACES': ['admin', 'django'],
    'WARNINGS_AS_ERRORS': False,
    'DATABASE_TIMEOUT': 5,
    'REDIS_TIMEOUT': 5,
}

# Health check URLs that don't require authentication
HEALTH_CHECK_PUBLIC_ENDPOINTS = [
    '/health/',
    '/health/ping/',
    '/health/ready/',
    '/health/live/',
]

# =============================================================================
# METRICS & TELEMETRY
# =============================================================================

# Enable Prometheus metrics (if using django-prometheus)
PROMETHEUS_ENABLED = os.environ.get('PROMETHEUS_ENABLED', 'False').lower() == 'true'
if PROMETHEUS_ENABLED:
    PROMETHEUS_EXPORT_MIGRATIONS = False
    PROMETHEUS_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'))

# Custom metrics configuration
METRICS_CONFIG = {
    'enabled': PROMETHEUS_ENABLED,
    'endpoint': '/metrics/',
    'namespace': 'bunoraa',
    'subsystem': 'web',
}

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

# Slow query threshold (in seconds)
SLOW_QUERY_THRESHOLD = float(os.environ.get('SLOW_QUERY_THRESHOLD', '1.0'))

# Enable query profiling
QUERY_PROFILING_ENABLED = os.environ.get('QUERY_PROFILING_ENABLED', 'False').lower() == 'true'

# Request timing headers
REQUEST_TIMING_HEADER = 'X-Request-Duration'

# =============================================================================
# ADMIN ERROR REPORTING
# =============================================================================

ADMINS = [
    ('Admin', os.environ.get('ADMIN_EMAIL', 'admin@bunoraa.com')),
]

MANAGERS = ADMINS

# Email configuration for error reporting
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', '587'))
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD', '')
DEFAULT_FROM_EMAIL = os.environ.get('DEFAULT_FROM_EMAIL', 'noreply@bunoraa.com')
SERVER_EMAIL = os.environ.get('SERVER_EMAIL', 'errors@bunoraa.com')

# =============================================================================
# SECURITY HEADERS
# =============================================================================

# Security middleware settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# SSL/HTTPS settings (only in production)
if os.environ.get('ENVIRONMENT') == 'production':
    SECURE_SSL_REDIRECT = True
    SECURE_HSTS_SECONDS = 31536000  # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# =============================================================================
# CORRELATION ID FOR REQUEST TRACKING
# =============================================================================

CORRELATION_ID_HEADER = 'X-Correlation-ID'
CORRELATION_ID_RESPONSE_HEADER = 'X-Correlation-ID'

# Generate correlation ID if not provided
CORRELATION_ID_AUTO_GENERATE = True
