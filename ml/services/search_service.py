"""
Search Service

Django service for semantic search and product retrieval.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import json

try:
    from django.core.cache import cache
    from django.conf import settings
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    cache = None
    settings = None

import numpy as np

logger = logging.getLogger("bunoraa.ml.services.search")


class SearchService:
    """
    Service for semantic product search.
    
    Provides hybrid search combining keyword and semantic matching.
    """
    
    def __init__(self):
        self._models = {}
        self._cache_ttl = 1800  # 30 minutes
        self._model_registry = None
    
    def _get_registry(self):
        """Lazy load model registry."""
        if self._model_registry is None:
            from ..core.registry import ModelRegistry
            self._model_registry = ModelRegistry()
        return self._model_registry
    
    def _get_model(self, model_name: str):
        """Get or load a model."""
        if model_name not in self._models:
            registry = self._get_registry()
            self._models[model_name] = registry.get_latest(model_name)
        return self._models[model_name]
    
    def _cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key."""
        key_data = json.dumps(kwargs, sort_keys=True)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"search:{prefix}:{key_hash}"
    
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "relevance",
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search products with hybrid semantic + keyword matching.
        
        Args:
            query: Search query
            filters: Optional filters (category, price, etc.)
            page: Page number
            page_size: Results per page
            sort_by: Sorting method
            user_id: User ID for personalization
        
        Returns:
            Search results with products and facets
        """
        # Check cache (for non-personalized queries)
        cache_key = None
        if user_id is None and cache:
            cache_key = self._cache_key(
                "query",
                query=query,
                filters=filters,
                page=page,
                page_size=page_size,
                sort_by=sort_by
            )
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        # Log search
        self._log_search(query, user_id)
        
        results = {
            "query": query,
            "total": 0,
            "page": page,
            "page_size": page_size,
            "products": [],
            "facets": {},
            "suggestions": [],
        }
        
        try:
            # Get semantic results
            semantic_results = self._semantic_search(query, page_size * 2)
            
            # Get keyword results
            keyword_results = self._keyword_search(query, filters, page_size * 2)
            
            # Merge results
            merged = self._merge_search_results(
                semantic_results,
                keyword_results,
                weights=(0.6, 0.4)
            )
            
            # Apply filters
            if filters:
                merged = self._apply_filters(merged, filters)
            
            # Personalize if user provided
            if user_id:
                merged = self._personalize_results(merged, user_id)
            
            # Sort
            if sort_by != "relevance":
                merged = self._sort_results(merged, sort_by)
            
            # Paginate
            total = len(merged)
            start = (page - 1) * page_size
            end = start + page_size
            page_results = merged[start:end]
            
            # Get facets
            facets = self._get_facets(merged)
            
            # Get suggestions
            suggestions = self._get_suggestions(query)
            
            results = {
                "query": query,
                "total": total,
                "page": page,
                "page_size": page_size,
                "products": page_results,
                "facets": facets,
                "suggestions": suggestions,
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Fallback to keyword search only
            results["products"] = self._keyword_search(query, filters, page_size)
        
        # Cache results
        if cache_key and cache:
            cache.set(cache_key, results, self._cache_ttl)
        
        return results
    
    def autocomplete(
        self,
        query: str,
        num_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get autocomplete suggestions.
        
        Args:
            query: Partial query
            num_suggestions: Number of suggestions
        
        Returns:
            List of suggestions
        """
        cache_key = self._cache_key("auto", query=query[:20], num=num_suggestions)
        
        if cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        suggestions = []
        
        try:
            # Get query suggestions from search history
            query_suggestions = self._get_query_suggestions(query, num_suggestions)
            
            # Get product name matches
            product_suggestions = self._get_product_suggestions(query, num_suggestions)
            
            # Get category suggestions
            category_suggestions = self._get_category_suggestions(query, num_suggestions // 2)
            
            # Merge and dedupe
            seen = set()
            
            for s in query_suggestions:
                if s["text"].lower() not in seen:
                    seen.add(s["text"].lower())
                    suggestions.append({
                        "type": "query",
                        "text": s["text"],
                        "count": s.get("count", 0)
                    })
            
            for s in product_suggestions:
                if len(suggestions) >= num_suggestions:
                    break
                if s["name"].lower() not in seen:
                    seen.add(s["name"].lower())
                    suggestions.append({
                        "type": "product",
                        "text": s["name"],
                        "product_id": s["id"],
                        "image": s.get("image"),
                    })
            
            for s in category_suggestions:
                if len(suggestions) >= num_suggestions:
                    break
                suggestions.append({
                    "type": "category",
                    "text": s["name"],
                    "category_id": s["id"],
                })
            
        except Exception as e:
            logger.error(f"Autocomplete failed: {e}")
        
        if cache and suggestions:
            cache.set(cache_key, suggestions, 300)  # 5 minutes
        
        return suggestions
    
    def find_similar_products(
        self,
        product_id: int,
        num_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find visually and semantically similar products.
        
        Args:
            product_id: Product ID
            num_results: Number of results
        
        Returns:
            List of similar products
        """
        cache_key = self._cache_key("similar", product_id=product_id, num=num_results)
        
        if cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        similar = []
        
        try:
            # Get product embedding
            model = self._get_model("product_embeddings")
            if model:
                similar = model.find_similar(product_id, num_results)
            else:
                # Fallback to database similarity
                similar = self._db_similar_products(product_id, num_results)
            
            # Enrich with product data
            similar = self._enrich_products([s["product_id"] for s in similar])
            
        except Exception as e:
            logger.error(f"Similar products search failed: {e}")
        
        if cache and similar:
            cache.set(cache_key, similar, self._cache_ttl * 4)
        
        return similar
    
    def visual_search(
        self,
        image_data: bytes,
        num_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search products by image.
        
        Args:
            image_data: Image bytes
            num_results: Number of results
        
        Returns:
            List of matching products
        """
        try:
            from PIL import Image
            import io
            
            # Load image
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Get vision model
            model = self._get_model("product_classifier")
            if not model:
                logger.warning("Vision model not available")
                return []
            
            # Find similar
            import numpy as np
            image_array = np.array(image)
            similar = model.find_similar(image_array, num_results)
            
            # Enrich with product data
            return self._enrich_products([s["product_id"] for s in similar])
            
        except Exception as e:
            logger.error(f"Visual search failed: {e}")
            return []
    
    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------
    
    def _semantic_search(
        self,
        query: str,
        num_results: int
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        try:
            model = self._get_model("semantic_search")
            if not model:
                return []
            
            results = model.search(query, num_results)
            return results
        except Exception as e:
            logger.debug(f"Semantic search failed: {e}")
            return []
    
    def _keyword_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        num_results: int
    ) -> List[Dict[str, Any]]:
        """Perform keyword-based database search."""
        try:
            from apps.catalog.models import Product
            from django.db.models import Q
            
            # Build query
            q = Q(is_active=True)
            
            # Text search
            words = query.lower().split()
            text_q = Q()
            for word in words:
                text_q |= Q(name__icontains=word)
                text_q |= Q(description__icontains=word)
            
            q &= text_q
            
            # Apply filters
            if filters:
                if "category_id" in filters:
                    q &= Q(category_id=filters["category_id"])
                if "min_price" in filters:
                    q &= Q(price__gte=filters["min_price"])
                if "max_price" in filters:
                    q &= Q(price__lte=filters["max_price"])
                if "in_stock" in filters and filters["in_stock"]:
                    q &= Q(stock_quantity__gt=0)
            
            products = Product.objects.filter(q).values(
                "id", "name", "price", "image", "category_id", "rating"
            )[:num_results]
            
            return [
                {"product_id": p["id"], "score": 1.0, **p}
                for p in products
            ]
        except Exception as e:
            logger.debug(f"Keyword search failed: {e}")
            return []
    
    def _merge_search_results(
        self,
        semantic: List[Dict],
        keyword: List[Dict],
        weights: tuple = (0.6, 0.4)
    ) -> List[Dict[str, Any]]:
        """Merge semantic and keyword search results."""
        product_scores = {}
        product_data = {}
        
        # Process semantic results
        for i, result in enumerate(semantic):
            pid = result["product_id"]
            score = weights[0] * (1.0 / (i + 1) + result.get("score", 0))
            product_scores[pid] = score
            product_data[pid] = result
        
        # Process keyword results
        for i, result in enumerate(keyword):
            pid = result["product_id"]
            score = weights[1] * (1.0 / (i + 1))
            
            if pid in product_scores:
                product_scores[pid] += score
            else:
                product_scores[pid] = score
                product_data[pid] = result
        
        # Sort by combined score
        sorted_products = sorted(
            product_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {**product_data[pid], "score": score}
            for pid, score in sorted_products
        ]
    
    def _apply_filters(
        self,
        results: List[Dict],
        filters: Dict[str, Any]
    ) -> List[Dict]:
        """Apply filters to search results."""
        filtered = results
        
        if "category_id" in filters:
            filtered = [r for r in filtered if r.get("category_id") == filters["category_id"]]
        
        if "min_price" in filters:
            filtered = [r for r in filtered if r.get("price", 0) >= filters["min_price"]]
        
        if "max_price" in filters:
            filtered = [r for r in filtered if r.get("price", float("inf")) <= filters["max_price"]]
        
        if "in_stock" in filters and filters["in_stock"]:
            filtered = [r for r in filtered if r.get("stock_quantity", 0) > 0]
        
        return filtered
    
    def _personalize_results(
        self,
        results: List[Dict],
        user_id: int
    ) -> List[Dict]:
        """Personalize results based on user preferences."""
        try:
            from apps.accounts.behavior_models import UserBehaviorProfile
            
            profile = UserBehaviorProfile.objects.filter(user_id=user_id).first()
            
            if not profile:
                return results
            
            # Get preferred categories (stored as dict with category_id -> score)
            preferred_categories = list(profile.category_preferences.keys()) if profile.category_preferences else []
            
            # Boost products in preferred categories
            for result in results:
                if str(result.get("category_id")) in preferred_categories:
                    result["score"] *= 1.2
            
            # Re-sort
            return sorted(results, key=lambda x: x["score"], reverse=True)
            
        except Exception:
            return results
    
    def _sort_results(
        self,
        results: List[Dict],
        sort_by: str
    ) -> List[Dict]:
        """Sort results by specified field."""
        sort_map = {
            "price_asc": ("price", False),
            "price_desc": ("price", True),
            "rating": ("rating", True),
            "newest": ("created_at", True),
            "popular": ("view_count", True),
        }
        
        if sort_by in sort_map:
            field, reverse = sort_map[sort_by]
            return sorted(
                results,
                key=lambda x: x.get(field, 0) or 0,
                reverse=reverse
            )
        
        return results
    
    def _get_facets(self, results: List[Dict]) -> Dict[str, Any]:
        """Get facets (aggregations) from results."""
        facets = {
            "categories": {},
            "price_ranges": {},
        }
        
        for result in results:
            # Category facet
            cat_id = result.get("category_id")
            if cat_id:
                facets["categories"][cat_id] = facets["categories"].get(cat_id, 0) + 1
            
            # Price range facet
            price = result.get("price", 0)
            if price < 50:
                range_key = "under_50"
            elif price < 100:
                range_key = "50_100"
            elif price < 200:
                range_key = "100_200"
            else:
                range_key = "over_200"
            facets["price_ranges"][range_key] = facets["price_ranges"].get(range_key, 0) + 1
        
        return facets
    
    def _get_suggestions(self, query: str) -> List[str]:
        """Get query suggestions/corrections."""
        try:
            # Simple suggestions based on popular searches
            from apps.analytics.models import SearchLog
            from django.db.models import Count
            
            # Find similar queries
            words = query.lower().split()
            if not words:
                return []
            
            suggestions = SearchLog.objects.filter(
                query__icontains=words[0]
            ).exclude(
                query__iexact=query
            ).values("query").annotate(
                count=Count("query")
            ).order_by("-count")[:3]
            
            return [s["query"] for s in suggestions]
        except Exception:
            return []
    
    def _get_query_suggestions(
        self,
        query: str,
        num: int
    ) -> List[Dict[str, Any]]:
        """Get query autocomplete suggestions."""
        try:
            from apps.analytics.models import SearchLog
            from django.db.models import Count
            
            suggestions = SearchLog.objects.filter(
                query__istartswith=query
            ).values("query").annotate(
                count=Count("query")
            ).order_by("-count")[:num]
            
            return [{"text": s["query"], "count": s["count"]} for s in suggestions]
        except Exception:
            return []
    
    def _get_product_suggestions(
        self,
        query: str,
        num: int
    ) -> List[Dict[str, Any]]:
        """Get product name suggestions."""
        try:
            from apps.catalog.models import Product
            
            products = Product.objects.filter(
                is_active=True,
                name__icontains=query
            ).values("id", "name", "image")[:num]
            
            return list(products)
        except Exception:
            return []
    
    def _get_category_suggestions(
        self,
        query: str,
        num: int
    ) -> List[Dict[str, Any]]:
        """Get category suggestions."""
        try:
            from apps.catalog.models import Category
            
            categories = Category.objects.filter(
                name__icontains=query
            ).values("id", "name")[:num]
            
            return list(categories)
        except Exception:
            return []
    
    def _db_similar_products(
        self,
        product_id: int,
        num: int
    ) -> List[Dict[str, Any]]:
        """Fallback: Get similar products from database."""
        try:
            from apps.catalog.models import Product
            
            product = Product.objects.get(id=product_id)
            
            similar = Product.objects.filter(
                category_id=product.category_id,
                is_active=True
            ).exclude(
                id=product_id
            ).values("id")[:num]
            
            return [{"product_id": p["id"]} for p in similar]
        except Exception:
            return []
    
    def _enrich_products(
        self,
        product_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Enrich product IDs with full product data."""
        try:
            from apps.catalog.models import Product
            
            products = Product.objects.filter(
                id__in=product_ids,
                is_active=True
            ).values("id", "name", "price", "image", "category_id", "rating")
            
            product_map = {p["id"]: p for p in products}
            
            return [
                product_map[pid]
                for pid in product_ids
                if pid in product_map
            ]
        except Exception:
            return []
    
    def _log_search(self, query: str, user_id: Optional[int]):
        """Log search query for analytics."""
        try:
            from apps.analytics.models import SearchLog
            
            SearchLog.objects.create(
                query=query,
                user_id=user_id,
            )
        except Exception as e:
            logger.debug(f"Failed to log search: {e}")
