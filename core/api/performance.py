"""
Performance API Endpoints
=========================

Internal API for monitoring cache, database, and system performance.
"""
import logging
from functools import wraps

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from core.cache_manager import CacheManager, CacheWarmer
from core.db_optimizer import get_db_stats, reset_query_log
from core.monitoring import HealthChecker, MetricsCollector

logger = logging.getLogger('bunoraa.api.performance')


def require_internal_service(func):
    """Decorator to restrict access to internal services."""
    @wraps(func)
    def wrapper(request, *args, **kwargs):
        # Allow if DEBUG mode
        if settings.DEBUG:
            return func(request, *args, **kwargs)
        
        # Check for internal service token
        auth_header = request.headers.get('X-Internal-Service-Key', '')
        expected_key = getattr(settings, 'INTERNAL_SERVICE_KEY', None)
        
        if expected_key and auth_header == expected_key:
            return func(request, *args, **kwargs)
        
        # Check for valid API key in header
        api_key = request.headers.get('X-API-Key', '')
        if api_key and hasattr(settings, 'PERFORMANCE_API_KEY'):
            if api_key == settings.PERFORMANCE_API_KEY:
                return func(request, *args, **kwargs)
        
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    
    return wrapper


@csrf_exempt
@require_http_methods(['GET'])
@never_cache
@require_internal_service
def health_check(request):
    """Get system health status."""
    results = HealthChecker.run_all()
    status_code = 200 if results['status'] == 'healthy' else 503
    return JsonResponse(results, status=status_code)


@csrf_exempt
@require_http_methods(['GET'])
@never_cache
@require_internal_service
def metrics_overview(request):
    """Get key performance metrics."""
    try:
        system_metrics = MetricsCollector.collect_system_metrics()
        db_stats = get_db_stats()
        performance_summary = MetricsCollector.get_performance_summary(minutes=60)
        
        data = {
            'status': 'success',
            'timestamp': system_metrics.timestamp.isoformat(),
            'system': {
                'cpu_percent': system_metrics.cpu_percent,
                'memory_used_mb': round(system_metrics.memory_used_mb, 2),
                'memory_percent': system_metrics.memory_percent,
                'disk_usage_percent': system_metrics.disk_usage_percent,
                'load_average': system_metrics.load_average,
            },
            'database': db_stats,
            'performance': performance_summary,
        }
        
        return JsonResponse(data)
    
    except Exception as e:
        logger.exception("Error fetching metrics")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(['POST'])
@never_cache
@require_internal_service
def warm_cache(request):
    """Trigger cache warming for critical data."""
    try:
        import json
        body = json.loads(request.body) if request.body else {}
        warm_type = body.get('type', 'all')
        
        results = {}
        
        if warm_type in ['all', 'products']:
            results['products'] = CacheWarmer.warm_products()
        
        if warm_type in ['all', 'categories']:
            results['categories'] = CacheWarmer.warm_categories()
        
        return JsonResponse({
            'status': 'success',
            'message': f'Cache warming completed for {warm_type}',
            'results': results
        })
    
    except Exception as e:
        logger.exception("Error warming cache")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(['POST'])
@never_cache
@require_internal_service
def clear_cache(request):
    """Clear cache by pattern or all."""
    try:
        import json
        body = json.loads(request.body) if request.body else {}
        pattern = body.get('pattern')
        
        if pattern:
            count = CacheManager.delete_pattern(pattern)
            message = f"Deleted {count} keys matching '{pattern}'"
        else:
            from django.core.cache import cache
            cache.clear()
            message = "All cache cleared"
        
        return JsonResponse({
            'status': 'success',
            'message': message
        })
    
    except Exception as e:
        logger.exception("Error clearing cache")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(['POST'])
@never_cache
@require_internal_service
def reset_db_log(request):
    """Reset database query log."""
    try:
        reset_query_log()
        return JsonResponse({
            'status': 'success',
            'message': 'Database query log reset'
        })
    except Exception as e:
        logger.exception("Error resetting DB log")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
