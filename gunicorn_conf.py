import logging
import os

# ============================================
# STARTUP OPTIMIZATION - CRITICAL
# ============================================
# Fast startup mode - skip heavy operations
os.environ.setdefault('SKIP_MIGRATIONS_CHECK', 'True')
os.environ.setdefault('PROCESS_TYPE', 'web')
os.environ.setdefault('ML_ENABLED', 'True') # Enable ML features by default, can be overridden

# ============================================
# WORKER CONFIGURATION - Memory & Speed Optimized
# ============================================
# Reduce workers for memory-constrained environments (Render free/starter)
# Each worker consumes significant memory, especially with Django ORM
# Default to a single worker to avoid exhausting small DB connection limits.
_default_workers = 1
workers = int(os.environ.get('GUNICORN_WORKERS', str(_default_workers)))

bind = '0.0.0.0:' + os.environ.get('PORT', '8000')

# ============================================
# FAST STARTUP (CRITICAL)
# ============================================
# Default to disabled unless explicitly enabled.
preload_app = os.environ.get('GUNICORN_PRELOAD', 'False').lower() in ('true', '1', 'yes')

# ============================================
# TIMEOUT SETTINGS
# ============================================
timeout = int(os.environ.get('GUNICORN_TIMEOUT', '60'))  # Longer timeout for slower initial requests
graceful_timeout = int(os.environ.get('GUNICORN_GRACEFUL_TIMEOUT', '30'))
keepalive = int(os.environ.get('GUNICORN_KEEPALIVE', '5'))

# ============================================
# MEMORY MANAGEMENT - Critical for small instances
# ============================================
# Force worker recycling to prevent memory leaks from accumulating
max_requests = int(os.environ.get('GUNICORN_MAX_REQUESTS', 200))  # Recycle more often on low RAM
max_requests_jitter = int(os.environ.get('GUNICORN_MAX_REQUESTS_JITTER', 50))  # Add randomness

# /dev/shm uses RAM; prefer disk-backed tmp on tiny instances
worker_tmp_dir = '/tmp'

# ============================================
# WORKER CLASS & THREADING
# ============================================
# sync = fastest startup
# gthread = threaded workers (good for blocking I/O, lower memory than sync)
worker_class = os.environ.get('GUNICORN_WORKER_CLASS', 'sync')  # Changed to sync for faster startup
threads = int(os.environ.get('GUNICORN_THREADS', '1'))  # Reduced to 1 for sync workers
worker_connections = int(os.environ.get('GUNICORN_WORKER_CONNECTIONS', '100'))  # Reduced from 500

# ============================================
# LOGGING CONFIGURATION
# ============================================
accesslog = '-'
errorlog = '-'
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'warning')  # Changed from 'info' to reduce logging overhead


def _count_open_database_connections() -> int:
    try:
        from django import setup as django_setup
        from django.conf import settings
        from django.db import connections
    except Exception:
        return -1

    if not settings.configured:
        try:
            django_setup()
        except Exception:
            return -1

    active_connections = 0
    for alias in connections:
        connection = connections[alias]
        if getattr(connection, 'connection', None) is not None:
            active_connections += 1
    return active_connections


def _log_database_connection_warning(context: str) -> None:
    count = _count_open_database_connections()
    if count < 0:
        logging.warning('[%s] Unable to determine active DB connection count.', context)
    else:
        logging.warning('[%s] Active DB connections: %d', context, count)


def post_worker_init(worker):
    _log_database_connection_warning('post_worker_init')


