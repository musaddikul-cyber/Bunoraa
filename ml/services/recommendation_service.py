"""
Recommendation Service

Django service for product recommendations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
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
import requests # Added
from io import BytesIO # Added
from PIL import Image # Added
from apps.catalog.models import Product, ProductImage # Added
from ml.models.vision import ProductImageClassifier # Added

logger = logging.getLogger("bunoraa.ml.services.recommendations")


class RecommendationService:
    """
    Service for generating product recommendations.
    
    Integrates ML models with Django for real-time recommendations.
    """
    
    def __init__(self):
        self._models = {}
        self._cache_ttl = 3600  # 1 hour
        self._model_registry = None
        self._feature_store = None
    
    def _get_registry(self):
        """Lazy load model registry."""
        if self._model_registry is None:
            from ..core.registry import ModelRegistry
            self._model_registry = ModelRegistry()
        return self._model_registry
    
    def _get_feature_store(self):
        """Lazy load feature store."""
        if self._feature_store is None:
            from ..core.feature_store import FeatureStore
            self._feature_store = FeatureStore()
        return self._feature_store
    
    def _get_model(self, model_name: str):
        """Get or load a model."""
        if model_name not in self._models:
            registry = self._get_registry()
            self._models[model_name] = registry.get_latest(model_name)
        return self._models[model_name]
    
    def _cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key."""
        # Convert UUIDs and other non-serializable types to strings
        def serialize_value(v):
            if hasattr(v, 'hex'):  # UUID
                return str(v)
            return v
        
        serializable_kwargs = {k: serialize_value(v) for k, v in kwargs.items()}
        key_data = json.dumps(serializable_kwargs, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:8]
        return f"rec:{prefix}:{key_hash}"
    
    def get_personalized_recommendations(
        self,
        user_id: int,
        num_items: int = 20,
        category_id: Optional[int] = None,
        exclude_product_ids: Optional[List[int]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations for a user.
        
        Args:
            user_id: User ID
            num_items: Number of items to recommend
            category_id: Optional category filter
            exclude_product_ids: Products to exclude
            context: Additional context (device, time, etc.)
        
        Returns:
            List of recommended products
        """
        # Check cache
        cache_key = self._cache_key(
            "personalized",
            user_id=user_id,
            num_items=num_items,
            category_id=category_id
        )
        
        if cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        recommendations = []
        
        try:
            # Get user features
            user_features = self._get_user_features(user_id)
            
            # Try different recommendation strategies
            # 1. Collaborative filtering
            cf_recs = self._get_cf_recommendations(user_id, num_items * 2)
            
            # 2. Content-based (if user has history)
            cb_recs = []
            if user_features.get("has_history"):
                cb_recs = self._get_content_recommendations(user_id, num_items)
            
            # 3. Sequential (recent behavior)
            seq_recs = self._get_sequential_recommendations(user_id, num_items)
            
            # Merge and deduplicate
            all_recs = self._merge_recommendations([
                (cf_recs, 0.4),
                (cb_recs, 0.3),
                (seq_recs, 0.3),
            ])
            
            # Apply filters
            if category_id:
                all_recs = [r for r in all_recs if r.get("category_id") == category_id]
            
            if exclude_product_ids:
                exclude_set = set(exclude_product_ids)
                all_recs = [r for r in all_recs if r["product_id"] not in exclude_set]
            
            # Get top N
            recommendations = all_recs[:num_items]
            
            # Enrich with product data
            recommendations = self._enrich_recommendations(recommendations)
            
        except Exception as e:
            logger.error(f"Failed to get personalized recommendations: {e}")
            # Fallback to popular items
            recommendations = self.get_popular_items(num_items, category_id)
        
        # Cache results
        if cache and recommendations:
            cache.set(cache_key, recommendations, self._cache_ttl)
        
        return recommendations
    
    def get_similar_products(
        self,
        product_id: int,
        num_items: int = 10,
        similarity_type: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Get products similar to a given product.
        
        Args:
            product_id: Product ID
            num_items: Number of similar products
            similarity_type: Type of similarity (content, collaborative, hybrid)
        
        Returns:
            List of similar products
        """
        cache_key = self._cache_key(
            "similar",
            product_id=product_id,
            num_items=num_items,
            similarity_type=similarity_type
        )
        
        if cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        similar_products = []
        
        try:
            if similarity_type == "content":
                similar_products = self._get_content_similar(product_id, num_items)
            elif similarity_type == "collaborative":
                similar_products = self._get_collaborative_similar(product_id, num_items)
            else:  # hybrid
                content_similar = self._get_content_similar(product_id, num_items)
                collab_similar = self._get_collaborative_similar(product_id, num_items)
                
                similar_products = self._merge_recommendations([
                    (content_similar, 0.5),
                    (collab_similar, 0.5),
                ])[:num_items]
            
            similar_products = self._enrich_recommendations(similar_products)
            
        except Exception as e:
            logger.error(f"Failed to get similar products: {e}")
        
        if cache and similar_products:
            cache.set(cache_key, similar_products, self._cache_ttl)
        
        return similar_products
    
    def get_frequently_bought_together(
        self,
        product_id: int,
        num_items: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get products frequently bought together.
        
        Args:
            product_id: Product ID
            num_items: Number of items
        
        Returns:
            List of frequently bought together products
        """
        cache_key = self._cache_key("fbt", product_id=product_id, num_items=num_items)
        
        if cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        fbt_products = []
        
        try:
            # Get from co-purchase data
            from apps.orders.models import OrderItem
            from django.db.models import Count
            
            # Find orders containing this product
            order_ids = OrderItem.objects.filter(
                product_id=product_id
            ).values_list("order_id", flat=True)
            
            # Find other products in those orders
            co_products = OrderItem.objects.filter(
                order_id__in=order_ids
            ).exclude(
                product_id=product_id
            ).values("product_id").annotate(
                count=Count("product_id")
            ).order_by("-count")[:num_items]
            
            fbt_products = [
                {"product_id": p["product_id"], "score": p["count"]}
                for p in co_products
            ]
            
            fbt_products = self._enrich_recommendations(fbt_products)
            
        except Exception as e:
            logger.error(f"Failed to get FBT products: {e}")
        
        if cache and fbt_products:
            cache.set(cache_key, fbt_products, self._cache_ttl * 24)  # Longer cache
        
        return fbt_products
    
    def get_popular_items(
        self,
        num_items: int = 20,
        category_id: Optional[int] = None,
        time_window_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get popular/trending items.
        
        Args:
            num_items: Number of items
            category_id: Optional category filter
            time_window_days: Time window for popularity calculation
        
        Returns:
            List of popular products
        """
        cache_key = self._cache_key(
            "popular",
            num_items=num_items,
            category_id=category_id,
            time_window=time_window_days
        )
        
        if cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        popular_items = []
        
        try:
            from apps.catalog.models import Product
            from apps.analytics.models import ProductView
            from django.db.models import Count
            
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            
            # Get view counts
            query = ProductView.objects.filter(
                created_at__gte=cutoff_date
            )
            
            if category_id:
                query = query.filter(product__category_id=category_id)
            
            popular = query.values("product_id").annotate(
                views=Count("product_id")
            ).order_by("-views")[:num_items]
            
            popular_items = [
                {"product_id": p["product_id"], "score": p["views"]}
                for p in popular
            ]
            
            popular_items = self._enrich_recommendations(popular_items)
            
        except Exception as e:
            logger.error(f"Failed to get popular items: {e}")
        
        if cache and popular_items:
            cache.set(cache_key, popular_items, self._cache_ttl)
        
        return popular_items
    
    def get_recently_viewed(
        self,
        user_id: int,
        num_items: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get user's recently viewed products.
        
        Args:
            user_id: User ID
            num_items: Number of items
        
        Returns:
            List of recently viewed products
        """
        try:
            from apps.analytics.models import ProductView
            
            views = ProductView.objects.filter(
                user_id=user_id
            ).order_by("-created_at").values("product_id")[:num_items]
            
            products = [{"product_id": v["product_id"]} for v in views]
            return self._enrich_recommendations(products)
            
        except Exception as e:
            logger.error(f"Failed to get recently viewed: {e}")
            return []
    
    def get_cart_recommendations(
        self,
        user_id: int,
        cart_product_ids: List[int],
        num_items: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on cart contents.
        
        Args:
            user_id: User ID
            cart_product_ids: Products in cart
            num_items: Number of recommendations
        
        Returns:
            List of recommended products
        """
        if not cart_product_ids:
            return self.get_personalized_recommendations(user_id, num_items)
        
        recommendations = []
        
        for product_id in cart_product_ids[:3]:  # Use first 3 cart items
            fbt = self.get_frequently_bought_together(product_id, num_items)
            for item in fbt:
                if item["product_id"] not in cart_product_ids:
                    recommendations.append(item)
        
        # Deduplicate and sort by score
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec["product_id"] not in seen:
                seen.add(rec["product_id"])
                unique_recs.append(rec)
        
        return unique_recs[:num_items]
    
    def get_visually_similar_products(
        self,
        product_id: int,
        num_items: int = 10,
        exclude_product_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get visually similar products using a pre-trained image classifier and similarity index.
        
        Args:
            product_id: The ID of the product for which to find similar items.
            num_items: The number of similar items to return.
            exclude_product_ids: Optional list of product IDs to exclude from results.
        
        Returns:
            List of visually similar products, enriched with product data.
        """
        from apps.catalog.models import Product, ProductImage
        from ml.models.vision import ProductImageClassifier
        from django.conf import settings
        import requests
        from io import BytesIO
        from PIL import Image
        # numpy is already imported
        
        cache_key = self._cache_key(
            "visually_similar",
            product_id=product_id,
            num_items=num_items
        )
        
        if cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        similar_products_data = []
        try:
            # 1. Load the ProductImageClassifier model with its pre-built index
            registry = self._get_registry()
            model_entry = registry.get_production_model("visual_similarity_classifier")
            
            if not model_entry:
                logger.warning("Visual similarity classifier not found in registry. Returning empty list.")
                return []
            
            # Use model_class to load the specific type and its custom load method
            model: ProductImageClassifier = registry.load_model(model_entry.model_id, model_class=ProductImageClassifier)
            
            if not model:
                logger.error("Failed to load visual similarity classifier model.")
                return []

            # 2. Get the primary image of the target product
            target_product = Product.objects.filter(id=product_id, is_active=True).first()
            if not target_product:
                logger.warning(f"Product with ID {product_id} not found or not active.")
                return []
            
            primary_image_obj = target_product.images.filter(is_primary=True).first()
            if not primary_image_obj:
                primary_image_obj = target_product.images.first() # Fallback to first image
            
            if not primary_image_obj or not primary_image_obj.image:
                logger.warning(f"No primary image found for product {product_id}.")
                return []
            
            # 3. Download the image
            media_url_base = settings.MEDIA_URL
            if not media_url_base.endswith('/'):
                media_url_base += '/'
            
            image_url = f"{media_url_base}{primary_image_obj.image.name}"
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            
            target_image = Image.open(BytesIO(response.content)).convert('RGB')
            target_image_np = np.array(target_image)

            # 4. Find similar products using the loaded model
            # Exclude the product itself from the results, and any other specified IDs
            exclude_ids_list = [str(product_id)]
            if exclude_product_ids:
                exclude_ids_list.extend([str(pid) for pid in exclude_product_ids])

            similar_results = model.find_similar(
                image=target_image_np,
                top_k=num_items + len(exclude_ids_list), # Fetch more to account for exclusions
                exclude_ids=exclude_ids_list
            )
            
            # Extract product IDs and sort by similarity score
            similar_product_ids_scores = [
                {"product_id": res["product_id"], "score": res["similarity"]}
                for res in similar_results
            ]
            
            # 5. Enrich the results with product data
            similar_products_data = self._enrich_recommendations(similar_product_ids_scores[:num_items])
            
        except Exception as e:
            logger.error(f"Failed to get visually similar products for {product_id}: {e}")
            # Fallback to content-based or popular if visual fails
            similar_products_data = self.get_similar_products(product_id, num_items, similarity_type="content")
            if not similar_products_data:
                similar_products_data = self.get_popular_items(num_items)

        if cache and similar_products_data:
            cache.set(cache_key, similar_products_data, self._cache_ttl)
            
        return similar_products_data
    
    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------    
    def _get_user_features(self, user_id: int) -> Dict[str, Any]:
        """Get user features from feature store."""
        try:
            feature_store = self._get_feature_store()
            
            # Try to get user feature group, fall back to computing features
            features = feature_store.get_feature_group("user_features", f"user:{user_id}")
            
            if not features:
                # Compute basic features
                features = self._compute_user_features(user_id)
            
            return features
        except Exception as e:
            logger.warning(f"Failed to get user features: {e}")
            return self._compute_user_features(user_id)
    
    def _compute_user_features(self, user_id: int) -> Dict[str, Any]:
        """Compute user features from database."""
        try:
            from apps.recommendations.models import Interaction
            from apps.orders.models import Order
            
            # Check if user has history
            interaction_count = Interaction.objects.filter(user_id=user_id).count()
            order_count = Order.objects.filter(user_id=user_id).count()
            
            return {
                "has_history": interaction_count > 0,
                "interaction_count": interaction_count,
                "order_count": order_count,
            }
        except Exception:
            return {"has_history": False}
    
    def _get_cf_recommendations(
        self,
        user_id: int,
        num_items: int
    ) -> List[Dict[str, Any]]:
        """Get collaborative filtering recommendations."""
        try:
            model = self._get_model("recommendation_ncf")
            if model is None:
                return []
            
            predictions = model.predict(
                user_id=user_id,
                num_items=num_items
            )
            
            return [
                {"product_id": p["product_id"], "score": p["score"]}
                for p in predictions
            ]
        except Exception as e:
            logger.debug(f"CF recommendations failed: {e}")
            return []
    
    def _get_content_recommendations(
        self,
        user_id: int,
        num_items: int
    ) -> List[Dict[str, Any]]:
        """Get content-based recommendations."""
        try:
            # Get user's preferred products
            from apps.recommendations.models import Interaction
            
            liked_products = Interaction.objects.filter(
                user_id=user_id,
                interaction_type__in=["purchase", "add_to_cart", "wishlist"]
            ).order_by("-created_at").values_list("product_id", flat=True)[:10]
            
            if not liked_products:
                return []
            
            # Get similar products
            all_similar = []
            for product_id in liked_products[:5]:
                similar = self._get_content_similar(product_id, num_items // 2)
                all_similar.extend(similar)
            
            # Deduplicate
            seen = set(liked_products)
            unique = []
            for item in all_similar:
                if item["product_id"] not in seen:
                    seen.add(item["product_id"])
                    unique.append(item)
            
            return unique[:num_items]
            
        except Exception as e:
            logger.debug(f"Content recommendations failed: {e}")
            return []
    
    def _get_sequential_recommendations(
        self,
        user_id: int,
        num_items: int
    ) -> List[Dict[str, Any]]:
        """Get sequential recommendations based on recent behavior."""
        try:
            model = self._get_model("recommendation_sequence")
            if model is None:
                return []
            
            # Get user's recent sequence
            from apps.recommendations.models import Interaction
            
            recent = Interaction.objects.filter(
                user_id=user_id
            ).order_by("-created_at").values_list("product_id", flat=True)[:50]
            
            if not recent:
                return []
            
            predictions = model.predict_next(
                sequence=list(recent),
                num_items=num_items
            )
            
            return [
                {"product_id": p["product_id"], "score": p["score"]}
                for p in predictions
            ]
        except Exception as e:
            logger.debug(f"Sequential recommendations failed: {e}")
            return []
    
    def _get_content_similar(
        self,
        product_id: int,
        num_items: int
    ) -> List[Dict[str, Any]]:
        """Get content-based similar products."""
        try:
            model = self._get_model("product_embeddings")
            if model is None:
                return []
            
            similar = model.find_similar(product_id, num_items)
            return similar
        except Exception as e:
            logger.debug(f"Content similar failed: {e}")
            return []
    
    def _get_collaborative_similar(
        self,
        product_id: int,
        num_items: int
    ) -> List[Dict[str, Any]]:
        """Get collaborative filtering similar products."""
        try:
            # Users who bought this also bought
            from apps.orders.models import OrderItem
            from django.db.models import Count
            
            # Get users who bought this product
            user_ids = OrderItem.objects.filter(
                product_id=product_id
            ).values_list("order__user_id", flat=True)
            
            # Get other products these users bought
            similar = OrderItem.objects.filter(
                order__user_id__in=user_ids
            ).exclude(
                product_id=product_id
            ).values("product_id").annotate(
                count=Count("product_id")
            ).order_by("-count")[:num_items]
            
            return [
                {"product_id": p["product_id"], "score": p["count"]}
                for p in similar
            ]
        except Exception as e:
            logger.debug(f"Collaborative similar failed: {e}")
            return []
    
    def _merge_recommendations(
        self,
        rec_lists: List[Tuple[List[Dict], float]]
    ) -> List[Dict[str, Any]]:
        """
        Merge multiple recommendation lists with weights.
        
        Args:
            rec_lists: List of (recommendations, weight) tuples
        
        Returns:
            Merged and sorted recommendations
        """
        product_scores = {}
        
        for recs, weight in rec_lists:
            for i, rec in enumerate(recs):
                product_id = rec["product_id"]
                # Position-based score + original score
                position_score = 1.0 / (i + 1)
                original_score = rec.get("score", 1.0)
                
                if product_id not in product_scores:
                    product_scores[product_id] = 0
                
                product_scores[product_id] += weight * (0.5 * position_score + 0.5 * original_score)
        
        # Sort by score
        sorted_products = sorted(
            product_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"product_id": pid, "score": score}
            for pid, score in sorted_products
        ]
    
    def _enrich_recommendations(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich recommendations with product data."""
        if not recommendations:
            return []
        
        try:
            from apps.catalog.models import Product
            
            product_ids = [r["product_id"] for r in recommendations]
            
            products = Product.objects.filter(
                id__in=product_ids,
                is_active=True
            ).values("id", "name", "price", "image", "category_id", "rating")
            
            product_map = {p["id"]: p for p in products}
            
            enriched = []
            for rec in recommendations:
                product_id = rec["product_id"]
                if product_id in product_map:
                    enriched.append({
                        **rec,
                        **product_map[product_id],
                    })
            
            return enriched
            
        except Exception as e:
            logger.warning(f"Failed to enrich recommendations: {e}")
            return recommendations
