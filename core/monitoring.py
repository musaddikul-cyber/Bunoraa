"""
Comprehensive Monitoring System for Bunoraa
===========================================

Provides metrics collection, health checks, and alerting capabilities.
"""
import gc
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional

from django.conf import settings
from django.core.cache import cache
from django.db import connections, connection

logger = logging.getLogger('bunoraa.monitoring')

try:
    import resource
except ImportError:  # pragma: no cover - resource is not available on Windows
    resource = None


def _get_filesystem_root() -> str:
    """Return a valid filesystem root for the current platform."""
    if os.name != 'nt':
        return os.sep

    drive = os.environ.get('SystemDrive') or os.path.splitdrive(os.getcwd())[0] or 'C:'
    return f"{drive}{os.sep}"


def _get_load_average() -> tuple[float, float, float]:
    """Return system load average when supported by the platform."""
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return (0.0, 0.0, 0.0)


def _get_resource_memory_usage_mb() -> float:
    """Return process memory usage using the Unix-only resource module."""
    if resource is None:
        return 0.0

    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    if platform.system() == 'Darwin':
        return usage.ru_maxrss / 1024 / 1024
    return usage.ru_maxrss / 1024


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0
    disk_usage_percent: float = 0.0
    load_average: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0))


@dataclass
class RequestMetrics:
    """Request performance metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    path: str = ""
    method: str = ""
    duration_ms: float = 0.0
    status_code: int = 200
    response_size: int = 0
    db_queries: int = 0
    db_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class MetricsCollector:
    """
    Collect and store application metrics.
    """
    
    METRICS_KEY = "bunoraa:metrics:rolling"
    MAX_METRICS = 1000
    
    @classmethod
    def collect_system_metrics(cls) -> SystemMetrics:
        """Collect current system metrics."""
        metrics = SystemMetrics()
        
        try:
            import psutil
            
            # CPU
            metrics.cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory
            mem = psutil.virtual_memory()
            metrics.memory_used_mb = mem.used / (1024 * 1024)
            metrics.memory_percent = mem.percent
            
            # Disk
            disk = psutil.disk_usage(_get_filesystem_root())
            metrics.disk_usage_percent = disk.percent
            
            # Load average
            metrics.load_average = _get_load_average()
            
        except ImportError:
            # Fallback without psutil
            try:
                metrics.memory_used_mb = _get_resource_memory_usage_mb()
            except Exception:
                pass
        
        return metrics
    
    @classmethod
    def record_request(cls, metrics: RequestMetrics) -> None:
        """Record request metrics for analysis."""
        try:
            # Store in cache (rolling window)
            key = f"{cls.METRICS_KEY}:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
            
            current = cache.get(key, [])
            current.append({
                'timestamp': metrics.timestamp.isoformat(),
                'path': metrics.path,
                'method': metrics.method,
                'duration_ms': metrics.duration_ms,
                'status_code': metrics.status_code,
                'db_queries': metrics.db_queries,
                'db_time_ms': metrics.db_time_ms
            })
            
            # Keep only recent metrics
            if len(current) > cls.MAX_METRICS:
                current = current[-cls.MAX_METRICS:]
            
            cache.set(key, current, 3600)  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"[METRICS ERROR] Failed to record: {e}")
    
    @classmethod
    def get_performance_summary(cls, minutes: int = 60) -> dict:
        """Get performance summary for recent period."""
        summary = {
            'total_requests': 0,
            'avg_response_time': 0.0,
            'p95_response_time': 0.0,
            'error_rate': 0.0,
            'top_slow_paths': []
        }
        
        all_durations = []
        error_count = 0
        path_durations = {}
        
        # Collect metrics from rolling windows
        for i in range(minutes):
            key = f"{cls.METRICS_KEY}:{(datetime.utcnow() - timedelta(minutes=i)).strftime('%Y%m%d%H%M')}"
            metrics = cache.get(key, [])
            
            for m in metrics:
                all_durations.append(m['duration_ms'])
                
                if m['status_code'] >= 400:
                    error_count += 1
                
                path = m['path']
                if path not in path_durations:
                    path_durations[path] = []
                path_durations[path].append(m['duration_ms'])
        
        if all_durations:
            all_durations.sort()
            summary['total_requests'] = len(all_durations)
            summary['avg_response_time'] = sum(all_durations) / len(all_durations)
            summary['p95_response_time'] = all_durations[int(len(all_durations) * 0.95)]
            summary['error_rate'] = error_count / len(all_durations)
        
        # Top slow paths
        slow_paths = [
            (path, sum(durations) / len(durations))
            for path, durations in path_durations.items()
        ]
        slow_paths.sort(key=lambda x: x[1], reverse=True)
        summary['top_slow_paths'] = slow_paths[:10]
        
        return summary


class HealthChecker:
    """
    Health check utilities for the application.
    """
    
    CHECKS = {}
    
    @classmethod
    def register(cls, name: str, check_func: Callable[[], tuple[bool, str]]):
        """Register a health check."""
        cls.CHECKS[name] = check_func
    
    @classmethod
    def run_all(cls) -> dict:
        """Run all registered health checks."""
        results = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        for name, check_func in cls.CHECKS.items():
            try:
                is_healthy, message = check_func()
                results['checks'][name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'message': message
                }
                if not is_healthy:
                    results['status'] = 'unhealthy'
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'message': str(e)
                }
                results['status'] = 'unhealthy'
        
        return results


def check_database() -> tuple[bool, str]:
    """Check database connectivity."""
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            return True, "Database connection OK"
    except Exception as e:
        return False, f"Database connection failed: {e}"


def check_cache() -> tuple[bool, str]:
    """Check cache connectivity."""
    try:
        test_key = "health_check:test"
        cache.set(test_key, "ok", 10)
        value = cache.get(test_key)
        if value == "ok":
            return True, "Cache connection OK"
        return False, "Cache read/write mismatch"
    except Exception as e:
        return False, f"Cache connection failed: {e}"


def check_redis() -> tuple[bool, str]:
    """Check Redis connectivity."""
    try:
        from django_redis import get_redis_connection
        redis_conn = get_redis_connection('default')
        redis_conn.ping()
        
        # Check memory usage
        info = redis_conn.info('memory')
        used_mb = info.get('used_memory', 0) / (1024 * 1024)
        max_mb = info.get('maxmemory', 0) / (1024 * 1024)
        
        if max_mb > 0 and used_mb / max_mb > 0.9:
            return True, f"Redis OK but memory usage high ({used_mb:.1f}/{max_mb:.1f} MB)"
        
        return True, f"Redis connection OK (memory: {used_mb:.1f} MB)"
    except Exception as e:
        return False, f"Redis connection failed: {e}"


def check_disk_space() -> tuple[bool, str]:
    """Check available disk space."""
    try:
        import shutil
        stat = shutil.disk_usage(_get_filesystem_root())
        free_gb = stat.free / (1024 ** 3)
        total_gb = stat.total / (1024 ** 3)
        percent_used = (stat.used / stat.total) * 100
        
        if percent_used > 90:
            return False, f"Disk space critical: {percent_used:.1f}% used ({free_gb:.1f} GB free)"
        elif percent_used > 80:
            return True, f"Disk space warning: {percent_used:.1f}% used ({free_gb:.1f} GB free)"
        
        return True, f"Disk space OK: {percent_used:.1f}% used ({free_gb:.1f} GB free)"
    except Exception as e:
        return False, f"Disk check failed: {e}"


def check_memory() -> tuple[bool, str]:
    """Check system memory."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        
        if mem.percent > 90:
            return False, f"Memory critical: {mem.percent:.1f}% used"
        elif mem.percent > 80:
            return True, f"Memory warning: {mem.percent:.1f}% used ({mem.available / (1024**3):.1f} GB available)"
        
        return True, f"Memory OK: {mem.percent:.1f}% used"
    except ImportError:
        return True, "Memory check skipped (psutil not installed)"


# Register default health checks
HealthChecker.register('database', check_database)
HealthChecker.register('cache', check_cache)
HealthChecker.register('redis', check_redis)
HealthChecker.register('disk', check_disk_space)
HealthChecker.register('memory', check_memory)


class PerformanceMonitor:
    """
    Decorator and utilities for monitoring function performance.
    """
    
    def __init__(self, name: Optional[str] = None, log_threshold_ms: float = 100):
        self.name = name
        self.log_threshold_ms = log_threshold_ms
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = e
                raise
            finally:
                duration_ms = (time.time() - start) * 1000
                func_name = self.name or func.__name__
                
                if not success:
                    logger.error(f"[PERF] {func_name} FAILED in {duration_ms:.2f}ms: {error}")
                elif duration_ms > self.log_threshold_ms:
                    logger.warning(f"[PERF SLOW] {func_name} took {duration_ms:.2f}ms")
                else:
                    logger.debug(f"[PERF] {func_name} took {duration_ms:.2f}ms")
            
            return result
        
        return wrapper


def monitor_memory(threshold_mb: float = 500) -> Callable:
    """Decorator to monitor function memory usage."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            gc.collect()
            mem_before = get_memory_usage_mb()
            
            try:
                result = func(*args, **kwargs)
            finally:
                gc.collect()
                mem_after = get_memory_usage_mb()
                mem_diff = mem_after - mem_before
                
                if mem_diff > threshold_mb:
                    logger.warning(
                        f"[MEMORY] {func.__name__} increased by {mem_diff:.1f} MB "
                        f"(before: {mem_before:.1f}, after: {mem_after:.1f})"
                    )
            
            return result
        
        return wrapper
    return decorator


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            return _get_resource_memory_usage_mb()
        except Exception:
            return 0.0


def log_system_stats() -> None:
    """Log current system statistics."""
    metrics = MetricsCollector.collect_system_metrics()
    
    logger.info(
        f"[SYSTEM STATS] CPU: {metrics.cpu_percent:.1f}%, "
        f"Memory: {metrics.memory_percent:.1f}% ({metrics.memory_used_mb:.1f} MB), "
        f"Disk: {metrics.disk_usage_percent:.1f}%"
    )
