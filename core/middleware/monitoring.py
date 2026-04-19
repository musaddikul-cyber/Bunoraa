"""
Monitoring Middleware
=====================

Collects request metrics and performance data.
"""
import logging
import time
from typing import Optional

from django.conf import settings
from django.db import connection
from django.db.backends.utils import CursorWrapper
from django.utils.deprecation import MiddlewareMixin

from core.monitoring import RequestMetrics, MetricsCollector

logger = logging.getLogger('bunoraa.monitoring')


class MetricsCollectorMiddleware(MiddlewareMixin):
    """
    Collect request timing and performance metrics.
    """
    
    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.original_execute = None
    
    def process_request(self, request):
        """Start timing the request."""
        request._metrics_start_time = time.time()
        request._metrics_query_count = len(connection.queries)
        request._metrics_cache_hits = 0
        request._metrics_cache_misses = 0
    
    def process_response(self, request, response):
        """Record metrics after response."""
        if not hasattr(request, '_metrics_start_time'):
            return response
        
        try:
            duration_ms = (time.time() - request._metrics_start_time) * 1000
            query_count = len(connection.queries) - getattr(request, '_metrics_query_count', 0)
            
            # Calculate DB time
            db_time_ms = 0.0
            for query in connection.queries[-query_count:]:
                try:
                    db_time_ms += float(query.get('time', 0)) * 1000
                except (ValueError, TypeError):
                    pass
            
            # Create metrics
            metrics = RequestMetrics(
                path=request.path,
                method=request.method,
                duration_ms=duration_ms,
                status_code=response.status_code,
                response_size=len(response.content) if hasattr(response, 'content') else 0,
                db_queries=query_count,
                db_time_ms=db_time_ms,
                cache_hits=getattr(request, '_metrics_cache_hits', 0),
                cache_misses=getattr(request, '_metrics_cache_misses', 0)
            )
            
            # Record to collector
            MetricsCollector.record_request(metrics)
            
            # Add headers in debug mode or if explicitly enabled
            if settings.DEBUG or getattr(settings, 'ENABLE_PERFORMANCE_MONITORING', False):
                response['X-Duration-Ms'] = f"{duration_ms:.2f}"
                response['X-DB-Queries'] = str(query_count)
                response['X-DB-Time-Ms'] = f"{db_time_ms:.2f}"
            
            # Log slow requests
            slow_threshold = getattr(settings, 'PERFORMANCE_MONITORING_LOG_THRESHOLD_MS', 1000)
            if duration_ms > slow_threshold:
                logger.warning(
                    f"[SLOW REQUEST] {request.method} {request.path} "
                    f"took {duration_ms:.2f}ms ({query_count} queries, {db_time_ms:.2f}ms db)"
                )
        
        except Exception as e:
            logger.error(f"Error in metrics middleware: {e}")
        
        return response


class CacheMetricsMiddleware(MiddlewareMixin):
    """
    Track cache hit/miss metrics.
    """
    
    def __init__(self, get_response=None):
        super().__init__(get_response)
        self._patch_cache()
    
    def _patch_cache(self):
        """Patch cache to track hits/misses."""
        from django.core.cache import cache
        
        original_get = cache.get
        
        def patched_get(key, default=None, version=None):
            result = original_get(key, default, version)
            
            # Track in thread-local storage
            import threading
            _local = threading.local()
            if not hasattr(_local, 'cache_hits'):
                _local.cache_hits = 0
                _local.cache_misses = 0
            
            if result is not None:
                _local.cache_hits += 1
            else:
                _local.cache_misses += 1
            
            return result
        
        cache.get = patched_get
    
    def process_request(self, request):
        """Initialize cache counters."""
        import threading
        _local = threading.local()
        request._metrics_cache_hits = getattr(_local, 'cache_hits', 0)
        request._metrics_cache_misses = getattr(_local, 'cache_misses', 0)
        _local.cache_hits = 0
        _local.cache_misses = 0
    
    def process_response(self, request, response):
        """Update cache counters."""
        import threading
        _local = threading.local()
        request._metrics_cache_hits = getattr(_local, 'cache_hits', 0)
        request._metrics_cache_misses = getattr(_local, 'cache_misses', 0)
        return response


class QueryProfilerMiddleware(MiddlewareMixin):
    """
    Profile slow database queries in development.
    """
    
    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.enabled = getattr(settings, 'ENABLE_QUERY_PROFILING', False)
        self.threshold_ms = getattr(settings, 'QUERY_PROFILING_SLOW_THRESHOLD_MS', 100)
    
    def process_request(self, request):
        if not self.enabled:
            return None
        
        request._profile_start_queries = len(connection.queries)
    
    def process_response(self, request, response):
        if not self.enabled or not hasattr(request, '_profile_start_queries'):
            return response
        
        try:
            query_count = len(connection.queries) - request._profile_start_queries
            new_queries = connection.queries[request._profile_start_queries:]
            
            slow_queries = []
            total_time = 0.0
            
            for query in new_queries:
                try:
                    query_time = float(query.get('time', 0)) * 1000
                    total_time += query_time
                    
                    if query_time > self.threshold_ms:
                        slow_queries.append({
                            'sql': query.get('sql', '')[:150],
                            'time': query_time
                        })
                except (ValueError, TypeError):
                    pass
            
            if slow_queries or query_count > 20:  # Log if many queries or slow ones
                logger.info(
                    f"[QUERY PROFILE] {request.method} {request.path}: "
                    f"{query_count} queries, {total_time:.2f}ms total"
                )
                
                for sq in slow_queries[:5]:  # Top 5 slow
                    logger.warning(f"  SLOW: {sq['time']:.2f}ms - {sq['sql']}...")
        
        except Exception as e:
            logger.error(f"Error in query profiler: {e}")
        
        return response
