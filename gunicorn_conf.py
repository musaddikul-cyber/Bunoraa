"""
Gunicorn Configuration - Production Optimized with Enhanced Logging
====================================================================

This configuration addresses:
1. WORKER TIMEOUT issues - by increasing timeouts and optimizing worker settings
2. MEMORY LIMIT issues - by implementing aggressive memory management and recycling
3. Better logging and monitoring for production debugging

Environment Variables:
    GUNICORN_WORKERS: Number of workers (default: 1 for memory-constrained environments)
    GUNICORN_THREADS: Threads per worker (default: 4)
    GUNICORN_TIMEOUT: Request timeout in seconds (default: 180)
    GUNICORN_MAX_REQUESTS: Max requests before worker restart (default: 100)
    GUNICORN_LOG_LEVEL: Logging level (default: info)
    MEMORY_LIMIT_MB: Memory threshold for emergency recycling (default: 850)
"""

import logging
import os
import sys
import resource
import gc
import time
import signal

# ============================================
# STARTUP OPTIMIZATION - CRITICAL
# ============================================
# Fast startup mode - skip heavy operations
os.environ.setdefault('SKIP_MIGRATIONS_CHECK', 'True')
os.environ.setdefault('PROCESS_TYPE', 'web')
os.environ.setdefault('ML_ENABLED', 'True')

# ============================================
# MEMORY MONITORING SETUP
# ============================================
logger = logging.getLogger('gunicorn.error')

# Memory limit threshold (MB) - trigger aggressive cleanup
MEMORY_LIMIT_MB = int(os.environ.get('MEMORY_LIMIT_MB', '850'))
MEMORY_WARNING_MB = int(os.environ.get('MEMORY_WARNING_MB', '700'))


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is in KB on Linux, bytes on macOS
            if sys.platform == 'darwin':
                return usage.ru_maxrss / 1024 / 1024
            return usage.ru_maxrss / 1024
        except Exception:
            return None


def log_memory(tag: str, worker=None):
    """Log current memory usage with context."""
    mem = get_memory_usage_mb()
    if mem is not None:
        if worker:
            logger.warning('[MEMORY] [%s] Worker pid=%s mem=%.2fMB', tag, worker.pid, mem)
        else:
            logger.warning('[MEMORY] [%s] Master mem=%.2fMB', tag, mem)
        return mem
    return 0


def check_memory_threshold(worker):
    """Check if memory usage exceeds threshold and trigger cleanup."""
    mem = get_memory_usage_mb()
    if mem is None:
        return False
    
    if mem > MEMORY_LIMIT_MB:
        logger.error(
            '[MEMORY] CRITICAL: Worker pid=%s using %.2fMB (limit: %dMB). Triggering emergency cleanup.',
            worker.pid, mem, MEMORY_LIMIT_MB
        )
        # Force garbage collection
        gc.collect()
        # Close database connections
        try:
            from django import db
            db.connections.close_all()
            logger.info('[MEMORY] Closed all database connections for worker pid=%s', worker.pid)
        except Exception as e:
            logger.warning('[MEMORY] Error closing DB connections: %s', e)
        return True
    elif mem > MEMORY_WARNING_MB:
        logger.warning(
            '[MEMORY] WARNING: Worker pid=%s using %.2fMB (warning: %dMB)',
            worker.pid, mem, MEMORY_WARNING_MB
        )
    return False


# ============================================
# WORKER CONFIGURATION - Memory & Speed Optimized
# ============================================
# Calculate optimal workers based on memory constraints
# Render free tier: 512MB-1GB RAM
# Use single worker with threads for memory efficiency
_workers_env = os.environ.get('GUNICORN_WORKERS', '1')
if _workers_env.lower() == 'auto':
    # Auto-detect: use 1 worker for memory efficiency
    workers = 1
else:
    workers = int(_workers_env)

bind = '0.0.0.0:' + os.environ.get('PORT', '8000')

# ============================================
# FAST STARTUP (CRITICAL)
# ============================================
# Preload disabled for Render - causes memory issues with single worker
preload_app = False

# ============================================
# TIMEOUT SETTINGS - OPTIMIZED FOR WEB SERVICES
# ============================================
# Higher timeout for slow database queries and cold starts
timeout = int(os.environ.get('GUNICORN_TIMEOUT', '180'))  # 3 minutes
graceful_timeout = int(os.environ.get('GUNICORN_GRACEFUL_TIMEOUT', '30'))
keepalive = int(os.environ.get('GUNICORN_KEEPALIVE', '2'))

# ============================================
# MEMORY MANAGEMENT - CRITICAL FOR RENDER
# ============================================
# Aggressive worker recycling to prevent memory bloat
max_requests = int(os.environ.get('GUNICORN_MAX_REQUESTS', 50))  # Restart every 50 requests
max_requests_jitter = int(os.environ.get('GUNICORN_MAX_REQUESTS_JITTER', 10))
worker_tmp_dir = '/tmp'

# ============================================
# WORKER CLASS - SYNC FOR STABILITY
# ============================================
# Use 'sync' worker class for stability with SQLite/PostgreSQL
# 'gthread' can cause issues with connection pooling
worker_class = 'sync'
threads = int(os.environ.get('GUNICORN_THREADS', '1'))  # Not used with sync
worker_connections = int(os.environ.get('GUNICORN_WORKER_CONNECTIONS', '1000'))

# ============================================
# SERVER SOCKET & QUEUE SETTINGS
# ============================================
backlog = int(os.environ.get('GUNICORN_BACKLOG', '2048'))
limit_request_line = int(os.environ.get('GUNICORN_LIMIT_REQUEST_LINE', '4094'))
limit_request_fields = int(os.environ.get('GUNICORN_LIMIT_REQUEST_FIELDS', '100'))
limit_request_field_size = int(os.environ.get('GUNICORN_LIMIT_REQUEST_FIELD_SIZE', '8190'))

# ============================================
# LOGGING CONFIGURATION - ENHANCED
# ============================================
accesslog = '-'
errorlog = '-'
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'info')
capture_output = True
enable_stdio_inheritance = True

# ============================================
# WORKER LIFECYCLE HOOKS
# ============================================
def on_starting(server):
    """Called just before master process is initialized."""
    logger.info('[STARTUP] Gunicorn master starting with %d worker(s)', workers)
    logger.info('[CONFIG] timeout=%ds max_requests=%d memory_limit=%dMB', 
                timeout, max_requests, MEMORY_LIMIT_MB)


def on_reload(server):
    """Called when receiving SIGHUP."""
    logger.info('[RELOAD] Configuration reload requested')


def when_ready(server):
    """Called just after server is started."""
    logger.info('[READY] Gunicorn server is ready to accept connections')
    log_memory('master-ready')


def worker_int(worker):
    """Called when worker receives SIGINT or SIGQUIT."""
    logger.warning('[WORKER] Worker pid=%s received interrupt signal', worker.pid)
    log_memory('worker-interrupt', worker)


def worker_abort(worker):
    """Called when worker receives SIGABRT."""
    logger.error('[WORKER] Worker pid=%s aborted (timeout)', worker.pid)
    log_memory('worker-abort', worker)


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    logger.info('[WORKER] Worker pid=%s forked and ready', worker.pid)
    log_memory('worker-forked', worker)
    
    # Disable ML by default in workers to save memory
    os.environ['ML_ENABLED'] = 'False'


def pre_exec(server):
    """Called just before a new master process is forked."""
    logger.info('[RELOAD] New master process forking')


def pre_request(worker, req):
    """Called just before a worker processes the request."""
    worker._request_start_time = time.time()
    
    # Periodic memory check every 10 requests
    if not hasattr(worker, '_request_count'):
        worker._request_count = 0
    worker._request_count += 1
    
    if worker._request_count % 10 == 0:
        check_memory_threshold(worker)


def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    duration = time.time() - getattr(worker, '_request_start_time', time.time())
    
    # Log slow requests
    if duration > 10:  # Requests taking more than 10 seconds
        logger.warning(
            '[SLOW] Request to %s took %.2fs from worker pid=%s',
            req.path, duration, worker.pid
        )


def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    logger.info('[WORKER] Worker pid=%s exiting (processed ~%d requests)', 
                worker.pid, getattr(worker, '_request_count', 0))
    log_memory('worker-exit', worker)


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
    """Called after worker has been initialized."""
    log_memory('worker_init', worker)
    _log_database_connection_warning('post_worker_init')
    logger.info('[WORKER] Worker pid=%s initialized with %s threads', worker.pid, threads)


def worker_int(worker):
    """Called when worker receives SIGINT or SIGQUIT."""
    logger.warning('[WORKER] Worker pid=%s received interrupt signal - shutting down', worker.pid)
    log_memory('worker_interrupt', worker)


def worker_abort(worker):
    """Called when worker receives SIGABRT."""
    logger.error('[WORKER] Worker pid=%s aborted (timeout/memory exceeded)', worker.pid)
    log_memory('worker_abort', worker)


def pre_fork(server, worker):
    """Called before forking a worker."""
    server_mem = log_memory('pre_fork')
    if server_mem and server_mem > 512:  # 512MB threshold
        logger.warning('[MEMORY] High master memory usage detected: %.2fMB', server_mem)


def post_fork(server, worker):
    """Called after forking a worker."""
    log_memory('post_fork', worker)
    logger.info('[WORKER] Forked worker pid=%s', worker.pid)


def worker_exit(server, worker):
    """Called after worker has exited."""
    log_memory('worker_exit', worker)
    logger.info('[WORKER] Worker pid=%s exited', worker.pid)


def nworkers_changed(server, new_value, old_value):
    """Called when number of workers changed."""
    logger.info('[WORKERS] Count changed: %s -> %s', old_value, new_value)
    log_memory('workers_changed')


def on_exit(server):
    """Called before master process exits."""
    logger.info('[MASTER] Master process exiting')
    log_memory('master_exit')


def when_ready(server):
    """Called when server is ready."""
    logger.info('[MASTER] Server ready with %s workers', workers)
    log_memory('server_ready')


def pre_exec(server):
    """Called before new master process is forked."""
    logger.info('[MASTER] Executing new master process')
    log_memory('pre_exec')


