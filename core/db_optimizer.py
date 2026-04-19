"""
Database Query Optimizer for Bunoraa
====================================

Provides query optimization, indexing recommendations, and slow query analysis.
"""
import logging
import time
from contextlib import contextmanager
from typing import Optional, TypeVar

from django.db import connection, connections
from django.db.models import Model, QuerySet
from django.db.models.query import ValuesListIterable

logger = logging.getLogger('bunoraa.db')

T = TypeVar('T', bound=Model)


class QueryProfiler:
    """
    Profile database queries for performance analysis.
    """
    
    def __init__(self, threshold_ms: float = 100):
        self.threshold_ms = threshold_ms
        self.query_count = 0
        self.total_time = 0.0
        self.slow_queries = []
    
    def __enter__(self):
        self._original_debug = connection.force_debug_cursor
        connection.force_debug_cursor = True
        self.queries_before = len(connection.queries)
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        connection.force_debug_cursor = self._original_debug
        
        self.total_time = (self.end_time - self.start_time) * 1000
        queries_after = len(connection.queries)
        self.query_count = queries_after - self.queries_before
        
        # Analyze slow queries
        new_queries = connection.queries[self.queries_before:queries_after]
        for query in new_queries:
            query_time = float(query.get('time', 0)) * 1000
            if query_time > self.threshold_ms:
                self.slow_queries.append({
                    'sql': query.get('sql', '')[:200],
                    'time': query_time
                })
        
        # Log summary
        if self.query_count > 0:
            avg_time = self.total_time / self.query_count
            logger.info(
                f"[DB PROFILE] Queries: {self.query_count}, "
                f"Total: {self.total_time:.2f}ms, "
                f"Avg: {avg_time:.2f}ms, "
                f"Slow: {len(self.slow_queries)}"
            )
            
            for slow in self.slow_queries:
                logger.warning(
                    f"[SLOW QUERY] {slow['time']:.2f}ms: {slow['sql']}..."
                )


class QueryOptimizer:
    """
    Optimize Django ORM queries automatically.
    """
    
    @staticmethod
    def optimize_product_queryset(qs: QuerySet[T]) -> QuerySet[T]:
        """Optimize product queryset with prefetching."""
        return qs.select_related(
            'primary_category',
            'currency'
        ).prefetch_related(
            'images',
            'variants',
            'tags',
            'eco_certifications'
        )
    
    @staticmethod
    def optimize_category_queryset(qs: QuerySet[T]) -> QuerySet[T]:
        """Optimize category queryset."""
        return qs.select_related('parent').prefetch_related('children')
    
    @staticmethod
    def optimize_order_queryset(qs: QuerySet[T]) -> QuerySet[T]:
        """Optimize order queryset with related data."""
        return qs.select_related(
            'user',
            'shipping_address',
            'billing_address'
        ).prefetch_related(
            'items__product',
            'items__variant'
        )
    
    @staticmethod
    def add_index_recommendations(model: type[Model]) -> list[str]:
        """Analyze model and recommend indexes."""
        recommendations = []
        
        # Check for ForeignKeys without indexes
        for field in model._meta.fields:
            if field.get_internal_type() == 'ForeignKey':
                if not field.db_index:
                    recommendations.append(
                        f"models.Index(fields=['{field.name}_id'], name='{model._meta.db_table}_{field.name}_idx')"
                    )
        
        # Check common filtering fields
        common_filter_fields = ['slug', 'is_active', 'created_at', 'updated_at']
        for field_name in common_filter_fields:
            try:
                field = model._meta.get_field(field_name)
                if not field.db_index and field.get_internal_type() in ['CharField', 'DateTimeField', 'BooleanField']:
                    recommendations.append(
                        f"models.Index(fields=['{field_name}'], name='{model._meta.db_table}_{field_name}_idx')"
                    )
            except:
                pass
        
        return recommendations


@contextmanager
def profile_queries(threshold_ms: float = 100, name: str = ""):
    """Context manager for profiling database queries."""
    profiler = QueryProfiler(threshold_ms)
    with profiler:
        yield profiler
    
    if name:
        logger.info(f"[QUERY PROFILE: {name}] {profiler.query_count} queries in {profiler.total_time:.2f}ms")


def explain_query(queryset: QuerySet) -> dict:
    """Get EXPLAIN ANALYZE output for a queryset."""
    sql, params = queryset.query.sql_with_params()
    
    with connection.cursor() as cursor:
        cursor.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}", params)
        result = cursor.fetchone()
        if result:
            import json
            return json.loads(result[0])
    
    return {}


def get_db_stats() -> dict:
    """Get database connection statistics."""
    stats = {
        'connections': {},
        'totals': {
            'queries': 0,
            'time': 0.0
        }
    }
    
    for db_name in connections:
        conn = connections[db_name]
        queries = len(conn.queries)
        total_time = sum(float(q.get('time', 0)) for q in conn.queries)
        
        if queries > 0:
            avg_time = total_time / queries
        else:
            avg_time = 0.0
        
        stats['connections'][db_name] = {
            'queries': queries,
            'total_time': round(total_time, 3),
            'avg_time': round(avg_time, 3)
        }
        
        stats['totals']['queries'] += queries
        stats['totals']['time'] += total_time
    
    return stats


def reset_query_log() -> None:
    """Reset query log for all connections."""
    for db_name in connections:
        connections[db_name].queries_log.clear()
