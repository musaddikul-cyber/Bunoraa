"""
Advanced Cache Manager for Bunoraa
===============================

Provides intelligent caching with:
- Multi-level caching (Memory + Redis)
- Cache warming strategies
- Smart invalidation
- Performance metrics
"""
import hashlib
import json
import logging
import pickle
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

from django.core.cache import cache
from django.db.models import Model, QuerySet

logger = logging.getLogger('bunoraa.cache')

T = TypeVar('T')


class CacheManager:
    """
    Advanced cache management with intelligent strategies.
    """
    
    # Cache key prefixes
    PREFIXES = {
        'product': 'prod',
        'category': 'cat',
        'user': 'usr',
        'cart': 'cart',
        'order': 'ord',
        'search': 'srch',
        'page': 'page',
        'api': 'api',
    }
    
    # Default TTLs (seconds)
    DEFAULT_TTL = {
        'product': 3600,  # 1 hour
        'category': 7200,  # 2 hours
        'user': 1800,  # 30 minutes
        'cart': 900,  # 15 minutes
        'order': 3600,  # 1 hour
        'search': 600,  # 10 minutes
        'page': 3600,  # 1 hour
        'api': 300,  # 5 minutes
    }
    
    @staticmethod
    def generate_key(prefix: str, identifier: Union[str, int], suffix: Optional[str] = None) -> str:
        """Generate standardized cache key."""
        key = f"{CacheManager.PREFIXES.get(prefix, prefix)}:{identifier}"
        if suffix:
            key = f"{key}:{suffix}"
        return key
    
    @staticmethod
    def generate_query_key(model_name: str, query_params: dict) -> str:
        """Generate cache key from query parameters."""
        params_str = json.dumps(query_params, sort_keys=True, default=str)
        hash_suffix = hashlib.md5(params_str.encode()).hexdigest()[:12]
        return f"{CacheManager.PREFIXES.get('api', 'api')}:{model_name}:q:{hash_suffix}"
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get value from cache with logging."""
        start = time.time()
        value = cache.get(key, default)
        duration = (time.time() - start) * 1000
        
        if value is not None:
            logger.debug(f"[CACHE HIT] {key} ({duration:.2f}ms)")
        else:
            logger.debug(f"[CACHE MISS] {key} ({duration:.2f}ms)")
        
        return value
    
    @classmethod
    def set(
        cls,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        prefix: Optional[str] = None
    ) -> bool:
        """Set value in cache with automatic TTL."""
        if ttl is None and prefix:
            ttl = cls.DEFAULT_TTL.get(prefix, 300)
        elif ttl is None:
            ttl = 300
        
        try:
            cache.set(key, value, ttl)
            logger.debug(f"[CACHE SET] {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"[CACHE ERROR] Failed to set {key}: {e}")
            return False
    
    @classmethod
    def delete(cls, key: str) -> bool:
        """Delete value from cache."""
        try:
            cache.delete(key)
            logger.debug(f"[CACHE DELETE] {key}")
            return True
        except Exception as e:
            logger.error(f"[CACHE ERROR] Failed to delete {key}: {e}")
            return False
    
    @classmethod
    def delete_pattern(cls, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            # This requires Redis
            from django_redis import get_redis_connection
            redis_conn = get_redis_connection('default')
            keys = redis_conn.keys(f"*{pattern}*")
            if keys:
                redis_conn.delete(*keys)
                logger.info(f"[CACHE PURGE] Deleted {len(keys)} keys matching '{pattern}'")
                return len(keys)
            return 0
        except Exception as e:
            logger.error(f"[CACHE ERROR] Failed to delete pattern {pattern}: {e}")
            return 0
    
    @classmethod
    def invalidate_model(cls, model_name: str) -> int:
        """Invalidate all cache entries for a model."""
        return cls.delete_pattern(f":{model_name}:")
    
    @classmethod
    def warm_cache(cls, key_prefix: str, data_loader: Callable[[], T]) -> T:
        """Warm cache by pre-loading data."""
        data = data_loader()
        cls.set(key_prefix, data, prefix='api')
        return data


def cached(
    timeout: int = 300,
    key_prefix: Optional[str] = None,
    key_func: Optional[Callable] = None,
    cache_none: bool = False
):
    """
    Decorator for caching function results.
    
    Args:
        timeout: Cache TTL in seconds
        key_prefix: Cache key prefix
        key_func: Function to generate custom cache key args
        cache_none: Whether to cache None results
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key: function_name:arg1:arg2:...
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args[1:] if hasattr(arg, '__str__')])  # Skip self
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                cache_key = ":".join(key_parts)
                # Hash if too long
                if len(cache_key) > 250:
                    cache_key = hashlib.md5(cache_key.encode()).hexdigest()
            
            if key_prefix:
                cache_key = f"{key_prefix}:{cache_key}"
            
            # Try cache
            result = cache.get(cache_key)
            if result is not None or (result is None and cache_none):
                logger.debug(f"[CACHE HIT] {cache_key}")
                return result
            
            # Execute function
            logger.debug(f"[CACHE MISS] {cache_key}")
            result = func(*args, **kwargs)
            
            # Store in cache
            if result is not None or cache_none:
                cache.set(cache_key, result, timeout)
            
            return result
        
        return wrapper
    return decorator


class QuerySetCache:
    """
    Cache manager for Django QuerySets with intelligent invalidation.
    """
    
    def __init__(self, model: type[Model], cache_key: Optional[str] = None):
        self.model = model
        self.cache_key = cache_key or f"qs:{model._meta.label_lower}"
    
    def get(
        self,
        filter_kwargs: Optional[dict] = None,
        order_by: Optional[list] = None,
        select_related: Optional[list] = None,
        prefetch_related: Optional[list] = None,
        timeout: int = 3600
    ) -> QuerySet:
        """Get cached QuerySet or fetch from DB."""
        # Generate unique key for this query
        query_sig = {
            'filters': filter_kwargs or {},
            'order': order_by or [],
            'select': select_related or [],
            'prefetch': prefetch_related or [],
        }
        key = CacheManager.generate_query_key(self.model.__name__, query_sig)
        
        # Try cache
        cached_pks = cache.get(key)
        if cached_pks is not None:
            logger.debug(f"[QUERYSET CACHE HIT] {key}")
            return self.model.objects.filter(pk__in=cached_pks)
        
        # Build queryset
        logger.debug(f"[QUERYSET CACHE MISS] {key}")
        qs = self.model.objects.all()
        
        if filter_kwargs:
            qs = qs.filter(**filter_kwargs)
        if select_related:
            qs = qs.select_related(*select_related)
        if prefetch_related:
            qs = qs.prefetch_related(*prefetch_related)
        if order_by:
            qs = qs.order_by(*order_by)
        
        # Cache the PKs, not the objects
        pks = list(qs.values_list('pk', flat=True))
        cache.set(key, pks, timeout)
        
        return qs
    
    def invalidate(self) -> None:
        """Invalidate all cached queries for this model."""
        CacheManager.delete_pattern(f"qs:{self.model.__name__}")


class CacheWarmer:
    """
    Cache warming utilities for pre-populating cache.
    """
    
    @staticmethod
    def warm_products() -> dict:
        """Pre-cache popular products."""
        from apps.catalog.models import Product
        
        results = {
            'featured': 0,
            'bestsellers': 0,
            'new_arrivals': 0,
        }
        
        # Featured products
        featured = Product.objects.filter(
            is_active=True, is_featured=True
        ).select_related('primary_category')[:20]
        for product in featured:
            key = CacheManager.generate_key('product', product.pk)
            CacheManager.set(key, product, prefix='product')
        results['featured'] = len(featured)
        
        # Bestsellers
        bestsellers = Product.objects.filter(
            is_active=True, is_bestseller=True
        ).select_related('primary_category')[:20]
        for product in bestsellers:
            key = CacheManager.generate_key('product', product.pk, 'bestseller')
            CacheManager.set(key, product, prefix='product')
        results['bestsellers'] = len(bestsellers)
        
        # New arrivals
        new_arrivals = Product.objects.filter(
            is_active=True, is_new_arrival=True
        ).select_related('primary_category')[:20]
        for product in new_arrivals:
            key = CacheManager.generate_key('product', product.pk, 'new')
            CacheManager.set(key, product, prefix='product')
        results['new_arrivals'] = len(new_arrivals)
        
        logger.info(f"[CACHE WARM] Products: {results}")
        return results
    
    @staticmethod
    def warm_categories() -> int:
        """Pre-cache category tree."""
        from apps.catalog.models import Category
        
        categories = Category.objects.filter(is_active=True, is_deleted=False)
        count = 0
        for category in categories:
            key = CacheManager.generate_key('category', category.pk)
            CacheManager.set(key, category, prefix='category')
            count += 1
        
        # Cache tree structure
        tree_key = "cat:tree:all"
        tree_data = list(categories.values('id', 'name', 'slug', 'parent_id', 'depth'))
        CacheManager.set(tree_key, tree_data, prefix='category')
        
        logger.info(f"[CACHE WARM] Categories: {count}")
        return count
    
    @staticmethod
    def warm_all() -> dict:
        """Run all cache warming routines."""
        return {
            'products': CacheWarmer.warm_products(),
            'categories': CacheWarmer.warm_categories(),
        }


# Performance monitoring context manager
class CacheStats:
    """Context manager for tracking cache performance."""
    
    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = None
        self.hits = 0
        self.misses = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        logger.info(
            f"[CACHE STATS] {self.operation}: "
            f"duration={duration:.3f}s, hits={self.hits}, misses={self.misses}"
        )
