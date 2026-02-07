"""
Health check views for production monitoring.
"""
import time
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.views.decorators.cache import never_cache
from django.db import connection
from django.core.cache import cache
from django.conf import settings
import redis


@require_GET
@never_cache
def health_check(request):
    """
    Basic health check endpoint.
    Returns 200 if the service is running.
    """
    return JsonResponse({
        'status': 'ok',
        'service': 'bunoraa',
        'timestamp': time.time()
    })


@require_GET
@never_cache
def health_check_detailed(request):
    """
    Detailed health check endpoint.
    Checks database, cache, and other services.
    """
    # Only allow internal/authenticated access to detailed health
    auth_key = request.headers.get('X-Health-Check-Key')
    expected_key = getattr(settings, 'HEALTH_CHECK_KEY', None)
    
    if expected_key and auth_key != expected_key:
        return JsonResponse({'status': 'unauthorized'}, status=401)
    
    checks = {
        'database': check_database(),
        'cache': check_cache(),
        'redis': check_redis(),
        'storage': check_storage(),
    }
    
    # Calculate overall status
    all_ok = all(c['status'] == 'ok' for c in checks.values())
    
    return JsonResponse({
        'status': 'ok' if all_ok else 'degraded',
        'service': 'bunoraa',
        'version': getattr(settings, 'VERSION', '1.0.0'),
        'environment': 'production' if not settings.DEBUG else 'development',
        'checks': checks,
        'timestamp': time.time()
    }, status=200 if all_ok else 503)


def check_database():
    """Check database connectivity."""
    try:
        start = time.time()
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1')
            cursor.fetchone()
        latency = round((time.time() - start) * 1000, 2)
        
        return {
            'status': 'ok',
            'latency_ms': latency,
            'engine': settings.DATABASES['default']['ENGINE'].split('.')[-1]
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def check_cache():
    """Check cache connectivity."""
    try:
        start = time.time()
        test_key = 'health_check_test'
        cache.set(test_key, 'ok', timeout=10)
        result = cache.get(test_key)
        cache.delete(test_key)
        latency = round((time.time() - start) * 1000, 2)
        
        return {
            'status': 'ok' if result == 'ok' else 'error',
            'latency_ms': latency,
            'backend': settings.CACHES['default']['BACKEND'].split('.')[-1]
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def check_redis():
    """Check Redis connectivity (for Celery/Channels)."""
    redis_url = getattr(settings, 'CELERY_BROKER_URL', None)
    
    if not redis_url:
        return {'status': 'skipped', 'reason': 'Redis not configured'}
    
    try:
        start = time.time()
        r = redis.from_url(redis_url)
        r.ping()
        latency = round((time.time() - start) * 1000, 2)
        
        info = r.info()
        return {
            'status': 'ok',
            'latency_ms': latency,
            'version': info.get('redis_version'),
            'connected_clients': info.get('connected_clients'),
            'used_memory_human': info.get('used_memory_human')
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def check_storage():
    """Check storage backend connectivity."""
    try:
        from django.core.files.storage import default_storage
        
        start = time.time()
        
        # Try to list files (minimal operation)
        try:
            default_storage.listdir('.')
            operation = 'listdir'
        except NotImplementedError:
            # Some backends don't support listdir
            operation = 'exists'
            default_storage.exists('test_file_that_does_not_exist.txt')
        
        latency = round((time.time() - start) * 1000, 2)
        
        return {
            'status': 'ok',
            'latency_ms': latency,
            'backend': default_storage.__class__.__name__
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


@require_GET
@never_cache
def readiness_check(request):
    """
    Kubernetes-style readiness probe.
    Returns 200 only if the service is ready to accept traffic.
    """
    # Check critical services
    db_check = check_database()
    cache_check = check_cache()
    
    ready = (
        db_check['status'] == 'ok' and 
        cache_check['status'] in ['ok', 'skipped']
    )
    
    return JsonResponse({
        'ready': ready,
        'database': db_check['status'],
        'cache': cache_check['status']
    }, status=200 if ready else 503)


@require_GET
@never_cache
def liveness_check(request):
    """
    Kubernetes-style liveness probe.
    Returns 200 if the process is alive.
    """
    return JsonResponse({
        'alive': True,
        'timestamp': time.time()
    })
