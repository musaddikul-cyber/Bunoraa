"""
ML-based Recommendation and Personalization System for Bunoraa
Uses scikit-learn for collaborative filtering and content-based recommendations
"""
import os
import json
import pickle
import logging
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Optional, Tuple
from decimal import Decimal

import numpy as np
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone
from django.db.models import Count, Avg, F, Q, Sum

logger = logging.getLogger('bunoraa.ml')

# Model paths
ML_MODELS_DIR = Path(settings.BASE_DIR) / 'ml_models'
ML_MODELS_DIR.mkdir(exist_ok=True)


class ProductRecommender:
    """
    Product recommendation system using collaborative filtering and content-based filtering.
    """
    
    def __init__(self):
        self.model = None
        self.product_features = None
        self.user_product_matrix = None
        self.product_similarity_matrix = None
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models from disk."""
        try:
            model_path = ML_MODELS_DIR / 'recommendation_model.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.product_features = data.get('product_features')
                    self.user_product_matrix = data.get('user_product_matrix')
                    self.product_similarity_matrix = data.get('similarity_matrix')
                logger.info("Recommendation models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load recommendation models: {e}")
    
    def get_similar_products(self, product_id: str, limit: int = 10) -> List[str]:
        """
        Get similar products based on content features.
        
        Args:
            product_id: UUID of the product
            limit: Number of similar products to return
            
        Returns:
            List of similar product UUIDs
        """
        cache_key = f'similar_products_{product_id}_{limit}'
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            from apps.products.models import Product
            
            product = Product.objects.get(id=product_id, is_active=True, is_deleted=False)
            
            # Get products from same categories
            category_ids = list(product.categories.values_list('id', flat=True))
            
            similar = Product.objects.filter(
                is_active=True,
                is_deleted=False,
                categories__id__in=category_ids
            ).exclude(id=product_id).annotate(
                relevance=Count('categories', filter=Q(categories__id__in=category_ids))
            ).order_by('-relevance', '-sold_count', '-view_count')[:limit]
            
            result = list(similar.values_list('id', flat=True))
            cache.set(cache_key, result, timeout=3600)  # 1 hour
            return result
            
        except Exception as e:
            logger.error(f"Error getting similar products: {e}")
            return []
    
    def get_personalized_recommendations(
        self,
        user_id: Optional[str] = None,
        session_key: Optional[str] = None,
        limit: int = 10,
        exclude_purchased: bool = True
    ) -> List[str]:
        """
        Get personalized product recommendations for a user.
        
        Args:
            user_id: UUID of the user (optional)
            session_key: Session key for anonymous users
            limit: Number of recommendations to return
            exclude_purchased: Whether to exclude already purchased products
            
        Returns:
            List of recommended product UUIDs
        """
        cache_key = f'recommendations_{user_id or session_key}_{limit}'
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            from apps.products.models import Product
            from apps.analytics.models import ProductView
            from apps.orders.models import OrderItem
            
            # Get user's viewed products
            viewed_products = set()
            purchased_products = set()
            
            if user_id:
                viewed_products = set(
                    ProductView.objects.filter(user_id=user_id)
                    .values_list('product_id', flat=True)[:100]
                )
                if exclude_purchased:
                    purchased_products = set(
                        OrderItem.objects.filter(order__user_id=user_id)
                        .values_list('product_id', flat=True)
                    )
            elif session_key:
                viewed_products = set(
                    ProductView.objects.filter(session_key=session_key)
                    .values_list('product_id', flat=True)[:50]
                )
            
            # If no history, return popular products
            if not viewed_products:
                return self.get_popular_products(limit)
            
            # Get categories from viewed products
            from apps.products.models import Product
            viewed_categories = set(
                Product.objects.filter(id__in=viewed_products)
                .values_list('categories__id', flat=True)
            )
            
            # Recommend products from similar categories
            exclude_ids = viewed_products | purchased_products
            
            recommendations = Product.objects.filter(
                is_active=True,
                is_deleted=False,
                categories__id__in=viewed_categories
            ).exclude(
                id__in=exclude_ids
            ).annotate(
                relevance_score=Count('categories', filter=Q(categories__id__in=viewed_categories)) * 10 +
                               F('sold_count') * 2 +
                               F('view_count')
            ).order_by('-relevance_score')[:limit]
            
            result = list(recommendations.values_list('id', flat=True))
            cache.set(cache_key, result, timeout=1800)  # 30 minutes
            return result
            
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return self.get_popular_products(limit)
    
    def get_popular_products(self, limit: int = 10) -> List[str]:
        """Get popular products as fallback."""
        cache_key = f'popular_products_{limit}'
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            from apps.products.models import Product
            
            products = Product.objects.filter(
                is_active=True,
                is_deleted=False
            ).order_by('-sold_count', '-view_count')[:limit]
            
            result = list(products.values_list('id', flat=True))
            cache.set(cache_key, result, timeout=3600)
            return result
            
        except Exception as e:
            logger.error(f"Error getting popular products: {e}")
            return []
    
    def get_trending_products(self, limit: int = 10, days: int = 7) -> List[str]:
        """Get trending products based on recent views and sales."""
        cache_key = f'trending_products_{limit}_{days}'
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            from apps.products.models import Product
            from apps.analytics.models import ProductView
            
            since = timezone.now() - timedelta(days=days)
            
            # Get products with most views in last N days
            trending = ProductView.objects.filter(
                created_at__gte=since,
                product__is_active=True,
                product__is_deleted=False
            ).values('product_id').annotate(
                view_count=Count('id')
            ).order_by('-view_count')[:limit]
            
            result = [str(t['product_id']) for t in trending]
            cache.set(cache_key, result, timeout=1800)
            return result
            
        except Exception as e:
            logger.error(f"Error getting trending products: {e}")
            return []
    
    def get_frequently_bought_together(self, product_id: str, limit: int = 5) -> List[str]:
        """Get products frequently bought together with the given product."""
        cache_key = f'bought_together_{product_id}_{limit}'
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        try:
            from apps.orders.models import Order, OrderItem
            
            # Find orders containing this product
            order_ids = OrderItem.objects.filter(
                product_id=product_id
            ).values_list('order_id', flat=True)[:100]
            
            # Find other products in those orders
            co_purchased = OrderItem.objects.filter(
                order_id__in=order_ids
            ).exclude(
                product_id=product_id
            ).values('product_id').annotate(
                frequency=Count('id')
            ).order_by('-frequency')[:limit]
            
            result = [str(c['product_id']) for c in co_purchased]
            cache.set(cache_key, result, timeout=3600)
            return result
            
        except Exception as e:
            logger.error(f"Error getting frequently bought together: {e}")
            return []


class UserSegmentationEngine:
    """
    User segmentation using clustering for targeted marketing.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained segmentation model."""
        try:
            model_path = ML_MODELS_DIR / 'segmentation_model.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.scaler = data.get('scaler')
        except Exception as e:
            logger.warning(f"Could not load segmentation model: {e}")
    
    def get_user_segment(self, user_id: str) -> Dict:
        """
        Get user's segment and characteristics.
        
        Returns:
            Dict with segment_id, segment_name, and characteristics
        """
        try:
            from apps.accounts.behavior_models import UserBehaviorProfile
            
            profile = UserBehaviorProfile.objects.get(user_id=user_id)
            
            # Simple rule-based segmentation if ML model not available
            if not self.model:
                return self._rule_based_segmentation(profile)
            
            # ML-based segmentation
            features = self._extract_features(profile)
            scaled_features = self.scaler.transform([features])
            segment_id = int(self.model.predict(scaled_features)[0])
            
            return {
                'segment_id': segment_id,
                'segment_name': self._get_segment_name(segment_id),
                'characteristics': self._get_segment_characteristics(segment_id)
            }
            
        except Exception as e:
            logger.error(f"Error getting user segment: {e}")
            return {'segment_id': 0, 'segment_name': 'Unknown', 'characteristics': {}}
    
    def _rule_based_segmentation(self, profile) -> Dict:
        """Rule-based segmentation as fallback."""
        # VIP customers
        if profile.total_spent > 50000 and profile.total_orders > 10:
            return {
                'segment_id': 1,
                'segment_name': 'VIP Customer',
                'characteristics': {
                    'spending_level': 'high',
                    'engagement': 'high',
                    'loyalty': 'high'
                }
            }
        
        # Regular customers
        if profile.total_orders > 3:
            return {
                'segment_id': 2,
                'segment_name': 'Regular Customer',
                'characteristics': {
                    'spending_level': 'medium',
                    'engagement': 'medium',
                    'loyalty': 'medium'
                }
            }
        
        # Window shoppers
        if profile.products_viewed > 50 and profile.total_orders < 2:
            return {
                'segment_id': 3,
                'segment_name': 'Window Shopper',
                'characteristics': {
                    'spending_level': 'low',
                    'engagement': 'high',
                    'loyalty': 'low'
                }
            }
        
        # New users
        return {
            'segment_id': 4,
            'segment_name': 'New User',
            'characteristics': {
                'spending_level': 'unknown',
                'engagement': 'low',
                'loyalty': 'unknown'
            }
        }
    
    def _extract_features(self, profile) -> List[float]:
        """Extract features for ML model."""
        return [
            float(profile.total_sessions or 0),
            float(profile.total_page_views or 0),
            float(profile.products_viewed or 0),
            float(profile.products_purchased or 0),
            float(profile.total_orders or 0),
            float(profile.total_spent or 0),
            float(profile.avg_order_value or 0),
            float(profile.engagement_score or 0),
            float(profile.loyalty_score or 0),
            float(profile.recency_score or 0),
        ]
    
    def _get_segment_name(self, segment_id: int) -> str:
        """Get segment name from ID."""
        segments = {
            0: 'New User',
            1: 'VIP Customer',
            2: 'Regular Customer',
            3: 'Window Shopper',
            4: 'Inactive User',
            5: 'Price Sensitive',
        }
        return segments.get(segment_id, 'Unknown')
    
    def _get_segment_characteristics(self, segment_id: int) -> Dict:
        """Get segment characteristics."""
        characteristics = {
            0: {'engagement': 'low', 'value': 'unknown'},
            1: {'engagement': 'high', 'value': 'high'},
            2: {'engagement': 'medium', 'value': 'medium'},
            3: {'engagement': 'high', 'value': 'low'},
            4: {'engagement': 'none', 'value': 'low'},
            5: {'engagement': 'medium', 'value': 'medium', 'price_sensitive': True},
        }
        return characteristics.get(segment_id, {})


class PersonalizationEngine:
    """
    Main personalization engine that combines recommendations and segmentation.
    """
    
    def __init__(self):
        self.recommender = ProductRecommender()
        self.segmentation = UserSegmentationEngine()
    
    def get_homepage_content(
        self,
        user_id: Optional[str] = None,
        session_key: Optional[str] = None
    ) -> Dict:
        """
        Get personalized homepage content.
        
        Returns:
            Dict with personalized product sections
        """
        content = {
            'recommended_for_you': [],
            'trending_now': [],
            'popular_products': [],
            'recently_viewed': [],
        }
        
        try:
            # Personalized recommendations
            if user_id or session_key:
                content['recommended_for_you'] = self.recommender.get_personalized_recommendations(
                    user_id=user_id,
                    session_key=session_key,
                    limit=12
                )
            
            # Trending products
            content['trending_now'] = self.recommender.get_trending_products(limit=8)
            
            # Popular products
            content['popular_products'] = self.recommender.get_popular_products(limit=8)
            
            # Recently viewed (from session)
            if session_key:
                from apps.analytics.models import ProductView
                recently_viewed = ProductView.objects.filter(
                    session_key=session_key
                ).order_by('-created_at').values_list('product_id', flat=True)[:8]
                content['recently_viewed'] = list(recently_viewed)
            
        except Exception as e:
            logger.error(f"Error getting homepage content: {e}")
        
        return content
    
    def get_product_page_recommendations(
        self,
        product_id: str,
        user_id: Optional[str] = None
    ) -> Dict:
        """
        Get recommendations for product detail page.
        """
        return {
            'similar_products': self.recommender.get_similar_products(product_id, limit=8),
            'frequently_bought_together': self.recommender.get_frequently_bought_together(product_id, limit=4),
            'you_may_also_like': self.recommender.get_personalized_recommendations(
                user_id=user_id,
                limit=4,
                exclude_purchased=True
            ) if user_id else [],
        }


# Singleton instances
recommender = ProductRecommender()
segmentation = UserSegmentationEngine()
personalization = PersonalizationEngine()
