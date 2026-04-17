"""
Health Check Endpoints - Production-Ready System Health Monitoring
Implements liveness, readiness, and deep health checks for Kubernetes and monitoring systems.
"""
import time
from functools import wraps
from typing import Callable, Dict, Any

from django.db import connection, DatabaseError
from django.http import JsonResponse
from django.core.cache import cache
from django.conf import settings


class HealthStatus:
    """Health check status constants."""
    PASS = 'pass'
    FAIL = 'fail'
    WARN = 'warn'


class HealthCheckResult:
    """Represents the result of a health check."""
    
    def __init__(
        self,
        name: str,
        status: str,
        response_time_ms: float,
        details: Dict[str, Any] = None,
        error: str = None
    ):
        self.name = name
        self.status = status
        self.response_time_ms = response_time_ms
        self.details = details or {}
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'name': self.name,
            'status': self.status,
            'responseTimeMs': round(self.response_time_ms, 2),
        }
        if self.details:
            result['details'] = self.details
        if self.error:
            result['error'] = self.error
        return result


def timed_health_check(func: Callable) -> Callable:
    """Decorator to time health check execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            result.response_time_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            return HealthCheckResult(
                name=func.__name__.replace('check_', ''),
                status=HealthStatus.FAIL,
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    return wrapper


@timed_health_check
def check_database() -> HealthCheckResult:
    """Check database connectivity."""
    start_time = time.time()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        
        return HealthCheckResult(
            name='database',
            status=HealthStatus.PASS,
            response_time_ms=(time.time() - start_time) * 1000,
            details={'engine': settings.DATABASES['default']['ENGINE']}
        )
    except DatabaseError as e:
        return HealthCheckResult(
            name='database',
            status=HealthStatus.FAIL,
            response_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


@timed_health_check
def check_cache() -> HealthCheckResult:
    """Check Redis cache connectivity."""
    start_time = time.time()
    try:
        test_key = 'health_check_test'
        test_value = 'ok'
        cache.set(test_key, test_value, timeout=10)
        retrieved = cache.get(test_key)
        cache.delete(test_key)
        
        if retrieved == test_value:
            return HealthCheckResult(
                name='cache',
                status=HealthStatus.PASS,
                response_time_ms=(time.time() - start_time) * 1000,
                details={'backend': settings.CACHES['default']['BACKEND']}
            )
        else:
            return HealthCheckResult(
                name='cache',
                status=HealthStatus.FAIL,
                response_time_ms=(time.time() - start_time) * 1000,
                error='Cache read/write mismatch'
            )
    except Exception as e:
        return HealthCheckResult(
            name='cache',
            status=HealthStatus.FAIL,
            response_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


@timed_health_check
def check_disk_space() -> HealthCheckResult:
    """Check available disk space."""
    import shutil
    start_time = time.time()
    
    try:
        usage = shutil.disk_usage('/')
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        percent_used = (usage.used / usage.total) * 100
        
        status = HealthStatus.PASS
        if percent_used > 90:
            status = HealthStatus.FAIL
        elif percent_used > 80:
            status = HealthStatus.WARN
        
        return HealthCheckResult(
            name='disk',
            status=status,
            response_time_ms=(time.time() - start_time) * 1000,
            details={
                'free_gb': round(free_gb, 2),
                'total_gb': round(total_gb, 2),
                'percent_used': round(percent_used, 2)
            }
        )
    except Exception as e:
        return HealthCheckResult(
            name='disk',
            status=HealthStatus.FAIL,
            response_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


@timed_health_check
def check_memory() -> HealthCheckResult:
    """Check system memory usage."""
    import psutil
    start_time = time.time()
    
    try:
        memory = psutil.virtual_memory()
        
        status = HealthStatus.PASS
        if memory.percent > 95:
            status = HealthStatus.FAIL
        elif memory.percent > 85:
            status = HealthStatus.WARN
        
        return HealthCheckResult(
            name='memory',
            status=status,
            response_time_ms=(time.time() - start_time) * 1000,
            details={
                'percent_used': memory.percent,
                'available_mb': memory.available // (1024 * 1024),
                'total_mb': memory.total // (1024 * 1024)
            }
        )
    except Exception as e:
        return HealthCheckResult(
            name='memory',
            status=HealthStatus.FAIL,
            response_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


# List of all health checks
ALL_HEALTH_CHECKS = [
    check_database,
    check_cache,
    check_disk_space,
]

# Check if psutil is available
try:
    import psutil
    ALL_HEALTH_CHECKS.append(check_memory)
except ImportError:
    pass


def run_health_checks(checks: list = None) -> Dict[str, Any]:
    """Run specified health checks and return results."""
    checks = checks or ALL_HEALTH_CHECKS
    results = []
    overall_status = HealthStatus.PASS
    
    for check in checks:
        result = check()
        results.append(result.to_dict())
        
        # Overall status is the worst of all checks
        if result.status == HealthStatus.FAIL:
            overall_status = HealthStatus.FAIL
        elif result.status == HealthStatus.WARN and overall_status == HealthStatus.PASS:
            overall_status = HealthStatus.WARN
    
    return {
        'status': overall_status,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version': getattr(settings, 'VERSION', 'unknown'),
        'checks': results
    }


# Django view functions for health check endpoints

def ping(request):
    """Simple ping endpoint for load balancer health checks."""
    return JsonResponse({'status': 'ok'}, status=200)


def liveness(request):
    """Kubernetes liveness probe - checks if the app is running."""
    return JsonResponse({
        'status': HealthStatus.PASS,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }, status=200)


def readiness(request):
    """Kubernetes readiness probe - checks if the app is ready to serve traffic."""
    # Only check critical services
    critical_checks = [check_database, check_cache]
    result = run_health_checks(critical_checks)
    
    status_code = 200 if result['status'] == HealthStatus.PASS else 503
    return JsonResponse(result, status=status_code)


def health(request):
    """Comprehensive health check endpoint."""
    result = run_health_checks()
    
    status_code = 200
    if result['status'] == HealthStatus.FAIL:
        status_code = 503
    elif result['status'] == HealthStatus.WARN:
        status_code = 200  # Still return 200 but with warning status
    
    return JsonResponse(result, status=status_code)
