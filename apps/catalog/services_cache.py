"""
Cached Product Services for Bunoraa
====================================

Demonstrates integration of CacheManager with catalog models.
This shows production-ready usage patterns for product caching.
"""
import logging
from typing import Optional, Union
from uuid import UUID

from django.db.models import Prefetch

from core.cache_manager import CacheManager, cached, QuerySetCache
from core.db_optimizer import QueryOptimizer

from apps.catalog.models import Product, Category

logger = logging.getLogger('bunoraa.catalog.cache')


class CachedProductService:
    """
    Product retrieval service with intelligent caching.
    Demonstrates the cache manager integration.
    """
    
    CACHE_PREFIX = 'product'
    CACHE_TTL = 3600  # 1 hour
    
    @classmethod
    def get_product(cls, product_id: Union[str, UUID], use_cache: bool = True) -> Optional[Product]:
        """
        Get single product by ID with caching.
        
        Usage:
            product = CachedProductService.get_product('uuid-here')
        """
        cache_key = CacheManager.generate_key(cls.CACHE_PREFIX, str(product_id))
        
        if use_cache:
            # Try cache first
            cached_data = CacheManager.get(cache_key)
            if cached_data is not None:
                logger.debug(f"[CACHE HIT] Product {product_id}")
                return cached_data
        
        # Fetch from database with optimized query
        try:
            product = QueryOptimizer.optimize_product_queryset(
                Product.objects.all()
            ).get(pk=product_id)
            
            # Store in cache
            if use_cache:
                CacheManager.set(cache_key, product, prefix=cls.CACHE_PREFIX)
                logger.debug(f"[CACHE SET] Product {product_id}")
            
            return product
            
        except Product.DoesNotExist:
            return None
    
    @classmethod
    def get_product_by_slug(cls, slug: str, use_cache: bool = True) -> Optional[Product]:
        """Get product by slug with caching."""
        cache_key = CacheManager.generate_key(cls.CACHE_PREFIX, f"slug:{slug}")
        
        if use_cache:
            cached = CacheManager.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            product = QueryOptimizer.optimize_product_queryset(
                Product.objects.all()
            ).get(slug__iexact=slug, is_active=True)
            
            if use_cache:
                CacheManager.set(cache_key, product, prefix=cls.CACHE_PREFIX)
            
            return product
            
        except Product.DoesNotExist:
            return None
    
    @classmethod
    def get_featured_products(cls, limit: int = 20, use_cache: bool = True):
        """
        Get featured products with caching.
        
        Usage:
            featured = CachedProductService.get_featured_products(limit=10)
        """
        cache_key = CacheManager.generate_key(cls.CACHE_PREFIX, 'featured', f'limit_{limit}')
        
        if use_cache:
            cached = CacheManager.get(cache_key)
            if cached is not None:
                return cached
        
        # Optimized query
        products = list(
            QueryOptimizer.optimize_product_queryset(
                Product.objects.filter(is_active=True, is_featured=True)
            ).select_related('primary_category')[:limit]
        )
        
        if use_cache:
            CacheManager.set(cache_key, products, prefix=cls.CACHE_PREFIX, ttl=1800)
        
        return products
    
    @classmethod
    def get_best_sellers(cls, limit: int = 20, use_cache: bool = True):
        """Get best-selling products with caching."""
        cache_key = CacheManager.generate_key(cls.CACHE_PREFIX, 'bestsellers', f'limit_{limit}')
        
        if use_cache:
            cached = CacheManager.get(cache_key)
            if cached is not None:
                return cached
        
        products = list(
            QueryOptimizer.optimize_product_queryset(
                Product.objects.filter(is_active=True, is_bestseller=True)
            )[:limit]
        )
        
        if use_cache:
            CacheManager.set(cache_key, products, prefix=cls.CACHE_PREFIX, ttl=3600)
        
        return products
    
    @classmethod
    def get_new_arrivals(cls, limit: int = 20, use_cache: bool = True):
        """Get new arrival products with caching."""
        cache_key = CacheManager.generate_key(cls.CACHE_PREFIX, 'new_arrivals', f'limit_{limit}')
        
        if use_cache:
            cached = CacheManager.get(cache_key)
            if cached is not None:
                return cached
        
        products = list(
            QueryOptimizer.optimize_product_queryset(
                Product.objects.filter(is_active=True, is_new_arrival=True)
            ).order_by('-created_at')[:limit]
        )
        
        if use_cache:
            CacheManager.set(cache_key, products, prefix=cls.CACHE_PREFIX, ttl=1800)
        
        return products
    
    @classmethod
    def get_products_by_category(
        cls, 
        category_id: Union[str, UUID], 
        include_subcategories: bool = True,
        limit: Optional[int] = None,
        use_cache: bool = True
    ):
        """Get products in a category with caching."""
        cache_key = CacheManager.generate_key(
            cls.CACHE_PREFIX, 
            f"cat:{category_id}",
            f"sub:{include_subcategories}_limit:{limit}"
        )
        
        if use_cache:
            cached = CacheManager.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            category = Category.objects.get(pk=category_id)
            qs = QueryOptimizer.optimize_product_queryset(
                category.get_products(include_subcategories=include_subcategories)
            )
            
            if limit:
                qs = qs[:limit]
            
            products = list(qs)
            
            if use_cache:
                CacheManager.set(cache_key, products, prefix=cls.CACHE_PREFIX, ttl=1800)
            
            return products
            
        except Category.DoesNotExist:
            return []
    
    @classmethod
    def invalidate_product(cls, product_id: Union[str, UUID]) -> None:
        """Invalidate all cache entries for a product."""
        # Delete main product cache
        CacheManager.delete(CacheManager.generate_key(cls.CACHE_PREFIX, str(product_id)))
        
        # Delete list caches that might contain this product
        CacheManager.delete_pattern(f"{cls.CACHE_PREFIX}:featured")
        CacheManager.delete_pattern(f"{cls.CACHE_PREFIX}:bestsellers")
        CacheManager.delete_pattern(f"{cls.CACHE_PREFIX}:new_arrivals")
        
        logger.info(f"[CACHE INVALIDATE] Product {product_id}")
    
    @classmethod
    def invalidate_category(cls, category_id: Union[str, UUID]) -> None:
        """Invalidate cache for a category's products."""
        CacheManager.delete_pattern(f"{cls.CACHE_PREFIX}:cat:{category_id}")
        logger.info(f"[CACHE INVALIDATE] Category {category_id}")


class CachedCategoryService:
    """Cached category service."""
    
    CACHE_PREFIX = 'category'
    
    @classmethod
    def get_category_tree(cls, use_cache: bool = True):
        """Get full category tree with caching."""
        cache_key = CacheManager.generate_key(cls.CACHE_PREFIX, 'tree', 'all')
        
        if use_cache:
            cached = CacheManager.get(cache_key)
            if cached is not None:
                return cached
        
        categories = list(
            Category.objects.filter(
                is_active=True, 
                is_deleted=False
            ).select_related('parent')
        )
        
        if use_cache:
            CacheManager.set(cache_key, categories, prefix=cls.CACHE_PREFIX, ttl=7200)
        
        return categories
    
    @classmethod
    def get_category(cls, category_id: Union[str, UUID], use_cache: bool = True):
        """Get single category with caching."""
        cache_key = CacheManager.generate_key(cls.CACHE_PREFIX, str(category_id))
        
        if use_cache:
            cached = CacheManager.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            category = Category.objects.get(pk=category_id)
            if use_cache:
                CacheManager.set(cache_key, category, prefix=cls.CACHE_PREFIX)
            return category
        except Category.DoesNotExist:
            return None
    
    @classmethod
    def invalidate_category(cls, category_id: Union[str, UUID]) -> None:
        """Invalidate category cache."""
        CacheManager.delete(CacheManager.generate_key(cls.CACHE_PREFIX, str(category_id)))
        CacheManager.delete(CacheManager.generate_key(cls.CACHE_PREFIX, 'tree', 'all'))


# Decorator versions for views
@cached(timeout=1800, key_prefix='api')
def get_popular_products_api(limit: int = 20):
    """API-optimized version with caching decorator."""
    return list(
        Product.objects.filter(
            is_active=True
        ).order_by('-views_count').select_related('primary_category')[:limit]
    )


@cached(timeout=3600, key_prefix='api')
def get_category_filters_api(category_id: str):
    """Get category filters (attributes) with caching."""
    try:
        category = Category.objects.get(pk=category_id)
        return list(category.filters.filter(is_active=True))
    except Category.DoesNotExist:
        return []
