"""
Memory Monitoring Middleware
============================

Tracks memory usage per request and logs warnings when thresholds are exceeded.
Helps identify memory leaks and resource-intensive endpoints.

Environment Variables:
    MEMORY_MONITOR_ENABLED: Enable monitoring (default: True in production)
    MEMORY_WARN_THRESHOLD_MB: Warning threshold in MB (default: 100)
    MEMORY_CRITICAL_THRESHOLD_MB: Critical threshold in MB (default: 250)
    MEMORY_LOG_ALL_REQUESTS: Log all requests regardless of threshold (default: False)
"""

import logging
import os
import time
import tracemalloc
from typing import Optional

from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger('bunoraa.memory')

# Configuration
MEMORY_MONITOR_ENABLED = os.environ.get('MEMORY_MONITOR_ENABLED', 'True').lower() in ('true', '1', 'yes')
MEMORY_WARN_THRESHOLD_MB = int(os.environ.get('MEMORY_WARN_THRESHOLD_MB', '100'))
MEMORY_CRITICAL_THRESHOLD_MB = int(os.environ.get('MEMORY_CRITICAL_THRESHOLD_MB', '250'))
MEMORY_LOG_ALL_REQUESTS = os.environ.get('MEMORY_LOG_ALL_REQUESTS', 'False').lower() in ('true', '1', 'yes')


def get_memory_usage_mb() -> Optional[float]:
    """Get current memory usage in MB using best available method."""
    try:
        import psutil
        import os as os_module
        process = psutil.Process(os_module.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        pass

    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux, Bytes on macOS
        if os.uname().sysname == 'Darwin':
            return usage.ru_maxrss / 1024 / 1024
        return usage.ru_maxrss / 1024
    except Exception:
        pass

    try:
        # Fallback using tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        return current / 1024 / 1024
    except Exception:
        pass

    return None


def format_memory_delta(delta_mb: float) -> str:
    """Format memory delta with color indicator."""
    if delta_mb > 50:
        return f"+{delta_mb:.2f}MB ⚠️"
    elif delta_mb > 20:
        return f"+{delta_mb:.2f}MB 📈"
    elif delta_mb > 0:
        return f"+{delta_mb:.2f}MB"
    elif delta_mb < -10:
        return f"{delta_mb:.2f}MB 📉"
    else:
        return f"{delta_mb:.2f}MB"


class MemoryMonitorMiddleware(MiddlewareMixin):
    """Middleware to monitor memory usage per request."""

    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.enabled = MEMORY_MONITOR_ENABLED
        self.warn_threshold = MEMORY_WARN_THRESHOLD_MB
        self.critical_threshold = MEMORY_CRITICAL_THRESHOLD_MB

    def process_request(self, request):
        """Capture memory at start of request."""
        if not self.enabled:
            return None

        request._memory_start = get_memory_usage_mb()
        request._time_start = time.time()

        # Start tracemalloc for detailed tracking if not already running
        if not tracemalloc.is_tracing():
            try:
                tracemalloc.start()
            except Exception:
                pass

        return None

    def process_response(self, request, response):
        """Log memory delta and response time."""
        if not self.enabled:
            return response

        # Calculate metrics
        memory_end = get_memory_usage_mb()
        memory_start = getattr(request, '_memory_start', None)
        time_start = getattr(request, '_time_start', None)

        memory_delta_mb = 0.0
        if memory_start is not None and memory_end is not None:
            memory_delta_mb = memory_end - memory_start

        duration_ms = 0.0
        if time_start is not None:
            duration_ms = (time.time() - time_start) * 1000

        # Build log message
        path = request.path[:100]  # Truncate long paths
        method = request.method
        status = response.status_code

        # Determine log level based on thresholds
        log_level = logging.DEBUG
        warning_flags = []

        if memory_end and memory_end > self.critical_threshold:
            log_level = logging.ERROR
            warning_flags.append("CRITICAL_MEMORY")
        elif memory_end and memory_end > self.warn_threshold:
            log_level = logging.WARNING
            warning_flags.append("HIGH_MEMORY")

        if duration_ms > 5000:  # 5 seconds
            log_level = max(log_level, logging.WARNING)
            warning_flags.append("SLOW_REQUEST")

        if memory_delta_mb > 50:  # 50MB increase
            log_level = max(log_level, logging.WARNING)
            warning_flags.append("MEMORY_SPIKE")

        # Format log message
        memory_current = f"{memory_end:.2f}MB" if memory_end else "N/A"
        memory_delta_str = format_memory_delta(memory_delta_mb)
        flags_str = f" [{' | '.join(warning_flags)}]" if warning_flags else ""

        log_message = (
            f"[MEMORY] {method} {path} => {status} | "
            f"Time: {duration_ms:.2f}ms | "
            f"Mem: {memory_current} ({memory_delta_str}){flags_str}"
        )

        # Log at appropriate level
        if log_level >= logging.WARNING or MEMORY_LOG_ALL_REQUESTS:
            logger.log(log_level, log_message)

        # Add memory header to response for debugging (if DEBUG)
        if os.environ.get('DEBUG', 'False').lower() == 'true':
            response['X-Memory-Usage-MB'] = str(memory_end) if memory_end else 'N/A'
            response['X-Request-Duration-Ms'] = str(int(duration_ms))

        return response

    def process_exception(self, request, exception):
        """Log memory on exception."""
        if not self.enabled:
            return None

        memory_end = get_memory_usage_mb()
        memory_start = getattr(request, '_memory_start', None)
        time_start = getattr(request, '_time_start', None)

        memory_delta_mb = 0.0
        if memory_start is not None and memory_end is not None:
            memory_delta_mb = memory_end - memory_start

        duration_ms = 0.0
        if time_start is not None:
            duration_ms = (time.time() - time_start) * 1000

        logger.error(
            "[MEMORY_EXCEPTION] %s %s => %s | "
            "Time: %.2fms | Mem: %.2fMB (+%.2fMB)",
            request.method,
            request.path[:100],
            type(exception).__name__,
            duration_ms,
            memory_end or 0,
            memory_delta_mb
        )

        return None
