"""
Core views
"""
import time
from django.views.generic import TemplateView
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.views.decorators.cache import never_cache
from django.db import connection
from django.core.cache import cache
from django.conf import settings


class HomeView(TemplateView):
    """Home page view."""
    template_name = 'home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Welcome to Bunoraa'
        context['meta_description'] = 'Discover premium products at Bunoraa. Shop our curated collection of high-quality items.'
        return context


@require_GET
@never_cache
def health_check(request):
    """
    Basic health check endpoint.
    Returns 200 if the service is running.
    """
    db_check = check_database()
    ok = db_check.get('status') == 'ok'
    return JsonResponse({
        'status': 'ok' if ok else 'degraded',
        'service': 'bunoraa',
        'database': db_check.get('status'),
        'timestamp': time.time()
    }, status=200 if ok else 503)


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
        'upstash_rest': check_upstash_rest(),
        'storage': check_storage(),
    }

    # Calculate overall status
    all_ok = all(c.get('status') == 'ok' for c in checks.values())

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
    """Check all configured Redis backends."""
    redis_targets = {
        'cache_celery': getattr(settings, 'CELERY_REDIS_URL', None) or getattr(settings, 'CELERY_BROKER_URL', None),
        'sessions': getattr(settings, 'SESSION_REDIS_URL', None) or getattr(settings, 'REDIS_URL', None),
    }

    channel_backend = (
        getattr(settings, 'CHANNEL_LAYERS', {})
        .get('default', {})
        .get('BACKEND', '')
    )
    if channel_backend == 'channels_redis.core.RedisChannelLayer':
        redis_targets['channels'] = getattr(settings, 'CHANNEL_LAYERS_REDIS_URL', None)

    if getattr(settings, 'ML_ENABLED', False):
        redis_targets['ml'] = getattr(settings, 'ML_REDIS_URL', None)

    redis_targets = {name: url for name, url in redis_targets.items() if url}

    if not redis_targets:
        return {'status': 'skipped', 'reason': 'Redis not configured'}

    try:
        import redis
    except Exception:
        return {'status': 'skipped', 'reason': 'Redis library not installed'}

    results = {}
    overall_status = 'ok'

    for name, redis_url in redis_targets.items():
        try:
            start = time.time()
            r = redis.from_url(redis_url)
            r.ping()
            latency = round((time.time() - start) * 1000, 2)

            details = {
                'status': 'ok',
                'latency_ms': latency,
            }

            try:
                info = r.info()
                details.update({
                    'version': info.get('redis_version'),
                    'connected_clients': info.get('connected_clients'),
                    'used_memory_human': info.get('used_memory_human'),
                })
            except Exception:
                # Some managed Redis providers restrict INFO; ping success is enough for health.
                pass

            results[name] = details
        except Exception as e:
            overall_status = 'error'
            results[name] = {
                'status': 'error',
                'error': str(e),
            }

    return {
        'status': overall_status,
        'targets': results,
    }


def check_upstash_rest():
    """Check Upstash Redis REST connectivity (optional)."""
    try:
        from core.services.upstash_rest import health_check as upstash_health
    except Exception:
        return {'status': 'skipped', 'reason': 'Upstash REST helper not available'}
    return upstash_health()


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
        db_check.get('status') == 'ok' and
        cache_check.get('status') in ['ok', 'skipped']
    )

    return JsonResponse({
        'ready': ready,
        'database': db_check.get('status'),
        'cache': cache_check.get('status')
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


# =============================================================================
# CUSTOM ERROR HANDLERS
# =============================================================================

def custom_404_view(request, exception=None):
    """
    Custom 404 error handler.
    Returns JSON for API requests, HTML for browser requests.
    """
    accept_header = request.headers.get('Accept', '')
    is_json_request = 'application/json' in accept_header or request.path.startswith('/api/')
    
    if is_json_request:
        return JsonResponse({
            'error': 'Not Found',
            'message': 'The requested resource was not found.',
            'path': request.path,
            'status_code': 404
        }, status=404)
    
    # For browser requests, render custom 404 template
    from django.shortcuts import render
    return render(request, '404.html', {
        'request_path': request.path,
        'page_title': 'Page Not Found - Bunoraa'
    }, status=404)


def custom_500_view(request):
    """
    Custom 500 error handler.
    Returns JSON for API requests, HTML for browser requests.
    """
    import logging
    logger = logging.getLogger('bunoraa.errors')
    
    # Log the error with request details
    logger.error(
        f"500 Error - Path: {request.path}, Method: {request.method}, "
        f"User: {request.user if hasattr(request, 'user') else 'Anonymous'}"
    )
    
    accept_header = request.headers.get('Accept', '')
    is_json_request = 'application/json' in accept_header or request.path.startswith('/api/')
    
    if is_json_request:
        return JsonResponse({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred. Our team has been notified.',
            'status_code': 500,
            'reference': f"ERR-{int(time.time())}"
        }, status=500)
    
    # For browser requests, render custom 500 template
    from django.shortcuts import render
    return render(request, '500.html', {
        'page_title': 'Server Error - Bunoraa',
        'error_reference': f"ERR-{int(time.time())}"
    }, status=500)
