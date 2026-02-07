import multiprocessing
import os

# ============================================
# WORKER CONFIGURATION - Memory Optimized
# ============================================
# Reduce workers for memory-constrained environments (Render free/starter)
# Each worker consumes significant memory, especially with Django ORM
max_workers = int(os.environ.get('GUNICORN_WORKERS', '1'))  # Default to 1 for 512MB instances
workers = min(max_workers, max(1, multiprocessing.cpu_count() // 2))  # Reduced from default

bind = '0.0.0.0:' + os.environ.get('PORT', '8000')
preload_app = True

# ============================================
# TIMEOUT SETTINGS
# ============================================
timeout = int(os.environ.get('GUNICORN_TIMEOUT', '60'))  # Reduced from 120
graceful_timeout = int(os.environ.get('GUNICORN_GRACEFUL_TIMEOUT', '30'))
keepalive = int(os.environ.get('GUNICORN_KEEPALIVE', '65'))

# ============================================
# MEMORY MANAGEMENT - Critical for small instances
# ============================================
# Force worker recycling to prevent memory leaks from accumulating
max_requests = int(os.environ.get('GUNICORN_MAX_REQUESTS', 1000))  # Recycle every 1000 requests
max_requests_jitter = int(os.environ.get('GUNICORN_MAX_REQUESTS_JITTER', 100))  # Add randomness

worker_tmp_dir = '/dev/shm'

# ============================================
# WORKER CLASS & THREADING
# ============================================
# gthread = threaded workers (good for blocking I/O, lower memory than sync)
worker_class = os.environ.get('GUNICORN_WORKER_CLASS', 'gthread')
threads = int(os.environ.get('GUNICORN_THREADS', '2'))  # Reduced from 4

# ============================================
# LOGGING CONFIGURATION
# ============================================
accesslog = '-'
errorlog = '-'
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'warning')  # Changed from 'info' to reduce logging overhead
