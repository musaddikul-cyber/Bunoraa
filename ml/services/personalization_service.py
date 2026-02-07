"""
Personalization Service

Django service for real-time personalization.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib
import json
from collections import defaultdict

from django.utils import timezone

try:
    from django.core.cache import cache
    from django.conf import settings
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    cache = None

import numpy as np

logger = logging.getLogger("bunoraa.ml.services.personalization")


class PersonalizationService:
    """
    Service for real-time personalization.
    
    Provides personalized content, rankings, and experiences.
    """
    
    def __init__(self):
        self._models = {}
        self._cache_ttl = 1800  # 30 minutes
        self._model_registry = None
        self._feature_store = None
    
    def _get_registry(self):
        if self._model_registry is None:
            from ..core.registry import ModelRegistry
            self._model_registry = ModelRegistry()
        return self._model_registry
    
    def _get_feature_store(self):
        if self._feature_store is None:
            from ..core.feature_store import FeatureStore
            self._feature_store = FeatureStore()
        return self._feature_store
    
    def _get_model(self, model_name: str):
        if model_name not in self._models:
            registry = self._get_registry()
            self._models[model_name] = registry.get_latest(model_name)
        return self._models[model_name]
    
    def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """
        Get comprehensive user profile for personalization.
        
        Args:
            user_id: User ID
        
        Returns:
            User profile with preferences and segments
        """
        cache_key = f"personalization:profile:{user_id}"
        
        if cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        profile = {
            "user_id": user_id,
            "preferences": {},
            "segments": [],
            "behavior": {},
            "computed_at": datetime.now().isoformat(),
        }
        
        try:
            # Get stored preferences
            profile["preferences"] = self._get_user_preferences(user_id)
            
            # Get behavior data
            profile["behavior"] = self._get_user_behavior(user_id)
            
            # Compute segments
            profile["segments"] = self._compute_user_segments(user_id, profile["behavior"])
            
            # Get user embedding
            user_embedding = self._get_user_embedding(user_id)
            if user_embedding is not None:
                profile["embedding_available"] = True
            
        except Exception as e:
            logger.error(f"Failed to build user profile: {e}")
        
        if cache:
            cache.set(cache_key, profile, self._cache_ttl)
        
        return profile
    
    def personalize_homepage(
        self,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get personalized homepage content.
        
        Args:
            user_id: User ID
            context: Additional context (device, time, etc.)
        
        Returns:
            Personalized homepage sections
        """
        from .recommendation_service import RecommendationService
        
        profile = self.get_user_profile(user_id)
        rec_service = RecommendationService()
        
        sections = []
        
        try:
            # Recently viewed (continue shopping)
            recently_viewed = rec_service.get_recently_viewed(user_id, num_items=8)
            if recently_viewed:
                sections.append({
                    "type": "recently_viewed",
                    "title": "Continue Shopping",
                    "products": recently_viewed,
                    "position": 1,
                })
            
            # Personalized recommendations
            personalized = rec_service.get_personalized_recommendations(
                user_id, 
                num_items=12,
                context=context
            )
            if personalized:
                sections.append({
                    "type": "personalized",
                    "title": "Recommended for You",
                    "products": personalized,
                    "position": 2,
                })
            
            # Category-specific recommendations
            preferred_categories = profile.get("preferences", {}).get("categories", [])
            for i, cat_id in enumerate(preferred_categories[:2]):
                cat_recs = rec_service.get_personalized_recommendations(
                    user_id,
                    num_items=8,
                    category_id=cat_id
                )
                if cat_recs:
                    sections.append({
                        "type": "category",
                        "title": self._get_category_name(cat_id),
                        "category_id": cat_id,
                        "products": cat_recs,
                        "position": 3 + i,
                    })
            
            # Trending/popular
            popular = rec_service.get_popular_items(num_items=8)
            sections.append({
                "type": "trending",
                "title": "Trending Now",
                "products": popular,
                "position": 10,
            })
            
            # New arrivals
            new_arrivals = self._get_new_arrivals(8)
            if new_arrivals:
                sections.append({
                    "type": "new_arrivals",
                    "title": "New Arrivals",
                    "products": new_arrivals,
                    "position": 11,
                })
            
            # Sort by position
            sections.sort(key=lambda x: x["position"])
            
        except Exception as e:
            logger.error(f"Homepage personalization failed: {e}")
        
        return {
            "user_id": user_id,
            "sections": sections,
            "profile_summary": {
                "segments": profile.get("segments", []),
            }
        }
    
    def personalize_category_page(
        self,
        user_id: int,
        category_id: int,
        products: List[Dict[str, Any]],
        page: int = 1,
        page_size: int = 24
    ) -> Dict[str, Any]:
        """
        Personalize product ordering on category page.
        
        Args:
            user_id: User ID
            category_id: Category ID
            products: Products to rank
            page: Page number
            page_size: Products per page
        
        Returns:
            Personalized product list
        """
        try:
            # Get user preferences
            profile = self.get_user_profile(user_id)
            
            # Score products
            scored_products = []
            for product in products:
                score = self._compute_product_score(product, profile)
                scored_products.append((product, score))
            
            # Sort by score
            scored_products.sort(key=lambda x: x[1], reverse=True)
            
            # Paginate
            start = (page - 1) * page_size
            end = start + page_size
            page_products = [p[0] for p in scored_products[start:end]]
            
            return {
                "products": page_products,
                "total": len(products),
                "page": page,
                "page_size": page_size,
                "personalized": True,
            }
            
        except Exception as e:
            logger.error(f"Category personalization failed: {e}")
            # Return unpersonalized
            start = (page - 1) * page_size
            return {
                "products": products[start:start + page_size],
                "total": len(products),
                "page": page,
                "page_size": page_size,
                "personalized": False,
            }
    
    def personalize_search_results(
        self,
        user_id: int,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Personalize search result ranking.
        
        Args:
            user_id: User ID
            query: Search query
            results: Search results to rerank
        
        Returns:
            Reranked results
        """
        try:
            profile = self.get_user_profile(user_id)
            
            # Score and rerank
            scored = []
            for result in results:
                base_score = result.get("score", 1.0)
                personal_score = self._compute_product_score(result, profile)
                
                # Combine search relevance with personalization
                combined_score = 0.7 * base_score + 0.3 * personal_score
                result["score"] = combined_score
                scored.append((result, combined_score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            
            return [r[0] for r in scored]
            
        except Exception as e:
            logger.error(f"Search personalization failed: {e}")
            return results
    
    def get_personalized_banners(
        self,
        user_id: int,
        num_banners: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get personalized promotional banners.
        
        Args:
            user_id: User ID
            num_banners: Number of banners
        
        Returns:
            Personalized banners
        """
        try:
            profile = self.get_user_profile(user_id)
            segments = profile.get("segments", [])
            
            # Get all available banners
            banners = self._get_available_banners()
            
            # Score banners based on user segments
            scored_banners = []
            for banner in banners:
                score = 1.0
                
                # Match target segments
                target_segments = banner.get("target_segments", [])
                if target_segments:
                    matching = len(set(target_segments) & set(segments))
                    if matching > 0:
                        score *= (1 + matching * 0.5)
                    else:
                        score *= 0.5
                
                # Match target categories
                target_categories = banner.get("target_categories", [])
                user_categories = profile.get("preferences", {}).get("categories", [])
                if target_categories and user_categories:
                    matching = len(set(target_categories) & set(user_categories))
                    if matching > 0:
                        score *= (1 + matching * 0.3)
                
                scored_banners.append((banner, score))
            
            # Sort and return top
            scored_banners.sort(key=lambda x: x[1], reverse=True)
            
            return [b[0] for b in scored_banners[:num_banners]]
            
        except Exception as e:
            logger.error(f"Banner personalization failed: {e}")
            return []
    
    def get_email_personalization(
        self,
        user_id: int,
        email_type: str
    ) -> Dict[str, Any]:
        """
        Get personalized content for emails.
        
        Args:
            user_id: User ID
            email_type: Type of email (welcome, abandoned_cart, etc.)
        
        Returns:
            Personalized email content
        """
        from .recommendation_service import RecommendationService
        
        profile = self.get_user_profile(user_id)
        rec_service = RecommendationService()
        
        content = {
            "user_id": user_id,
            "email_type": email_type,
            "products": [],
            "subject_variant": "default",
        }
        
        try:
            if email_type == "abandoned_cart":
                # Get cart items and similar products
                cart_items = self._get_user_cart(user_id)
                content["cart_items"] = cart_items
                
                cart_product_ids = [item["product_id"] for item in cart_items]
                content["products"] = rec_service.get_cart_recommendations(
                    user_id, cart_product_ids, num_items=4
                )
                
            elif email_type == "welcome":
                # Get popular items
                content["products"] = rec_service.get_popular_items(num_items=6)
                
            elif email_type == "re_engagement":
                # Get personalized recommendations
                content["products"] = rec_service.get_personalized_recommendations(
                    user_id, num_items=6
                )
                
            elif email_type == "weekly_digest":
                # Mix of new arrivals and personalized
                new_arrivals = self._get_new_arrivals(3)
                personalized = rec_service.get_personalized_recommendations(
                    user_id, num_items=3
                )
                content["products"] = new_arrivals + personalized
            
            # Choose subject line variant based on segment
            segments = profile.get("segments", [])
            if "vip" in segments:
                content["subject_variant"] = "vip"
            elif "bargain_hunter" in segments:
                content["subject_variant"] = "discount"
            
        except Exception as e:
            logger.error(f"Email personalization failed: {e}")
        
        return content
    
    def predict_churn_risk(self, user_id: int) -> Dict[str, Any]:
        """
        Predict churn risk for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            Churn risk assessment
        """
        try:
            model = self._get_model("churn_predictor")
            
            if model:
                # Get user features
                features = self._get_user_churn_features(user_id)
                prediction = model.predict(features)
                
                return {
                    "user_id": user_id,
                    "churn_probability": prediction["churn_probability"],
                    "risk_level": self._get_churn_risk_level(prediction["churn_probability"]),
                    "days_to_churn": prediction.get("days_to_churn"),
                    "reasons": prediction.get("churn_reasons", []),
                    "retention_actions": self._get_retention_actions(prediction),
                }
            else:
                # Fallback: Simple rules
                return self._simple_churn_assessment(user_id)
                
        except Exception as e:
            logger.error(f"Churn prediction failed: {e}")
            return {"error": str(e)}
    
    def get_next_best_action(
        self,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determine next best action for a user.
        
        Args:
            user_id: User ID
            context: Current context
        
        Returns:
            Recommended action
        """
        profile = self.get_user_profile(user_id)
        
        actions = []
        
        try:
            # Check cart abandonment
            cart = self._get_user_cart(user_id)
            if cart:
                cart_age_hours = self._get_cart_age_hours(user_id)
                if cart_age_hours > 1:
                    actions.append({
                        "type": "cart_reminder",
                        "priority": 0.9,
                        "data": {"cart_value": sum(item.get("price", 0) for item in cart)},
                    })
            
            # Check for re-engagement
            days_since_visit = profile.get("behavior", {}).get("days_since_last_visit", 0)
            if days_since_visit > 7:
                actions.append({
                    "type": "re_engagement",
                    "priority": 0.7,
                    "data": {"days_inactive": days_since_visit},
                })
            
            # Check churn risk
            churn = self.predict_churn_risk(user_id)
            if churn.get("risk_level") in ("high", "critical"):
                actions.append({
                    "type": "retention_offer",
                    "priority": 0.95,
                    "data": churn,
                })
            
            # Check for upsell opportunity
            if profile.get("behavior", {}).get("avg_order_value", 0) > 0:
                actions.append({
                    "type": "upsell",
                    "priority": 0.5,
                    "data": {},
                })
            
            # Sort by priority
            actions.sort(key=lambda x: x["priority"], reverse=True)
            
            return actions[0] if actions else {"type": "none", "priority": 0}
            
        except Exception as e:
            logger.error(f"Next best action failed: {e}")
            return {"type": "none", "error": str(e)}
    
    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------
    
    def _get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get user's preferences."""
        try:
            from apps.accounts.behavior_models import UserBehaviorProfile
            
            profile = UserBehaviorProfile.objects.filter(user_id=user_id).first()
            
            if profile:
                return {
                    "categories": list(profile.category_preferences.keys()) if profile.category_preferences else [],
                    "tags": list(profile.tag_preferences.keys()) if profile.tag_preferences else [],
                    "price_range": profile.price_range_preference or {},
                }
            
            return {}
        except Exception:
            return {}
    
    def _get_user_behavior(self, user_id: int) -> Dict[str, Any]:
        """Get user's behavior data."""
        try:
            from apps.orders.models import Order
            from apps.analytics.models import ProductView
            from django.db.models import Avg, Sum, Count
            
            # Order stats
            order_stats = Order.objects.filter(
                user_id=user_id
            ).aggregate(
                total_orders=Count("id"),
                total_spent=Sum("total_amount"),
                avg_order_value=Avg("total_amount"),
            )
            
            # Last visit
            last_view = ProductView.objects.filter(
                user_id=user_id
            ).order_by("-created_at").first()
            
            days_since_visit = 0
            if last_view:
                days_since_visit = (timezone.now() - last_view.created_at).days
            
            return {
                "total_orders": order_stats["total_orders"] or 0,
                "total_spent": float(order_stats["total_spent"] or 0),
                "avg_order_value": float(order_stats["avg_order_value"] or 0),
                "days_since_last_visit": days_since_visit,
            }
        except Exception:
            return {}
    
    def _compute_user_segments(
        self,
        user_id: int,
        behavior: Dict[str, Any]
    ) -> List[str]:
        """Compute user segments based on behavior."""
        segments = []
        
        total_spent = behavior.get("total_spent", 0)
        total_orders = behavior.get("total_orders", 0)
        avg_order = behavior.get("avg_order_value", 0)
        
        # Value segments
        if total_spent > 5000:
            segments.append("vip")
        elif total_spent > 1000:
            segments.append("high_value")
        elif total_spent > 100:
            segments.append("regular")
        else:
            segments.append("new_or_low_value")
        
        # Frequency segments
        if total_orders > 20:
            segments.append("frequent_buyer")
        elif total_orders > 5:
            segments.append("repeat_buyer")
        elif total_orders > 0:
            segments.append("occasional_buyer")
        else:
            segments.append("browser")
        
        # Order size segments
        if avg_order > 200:
            segments.append("big_spender")
        elif avg_order < 30:
            segments.append("bargain_hunter")
        
        return segments
    
    def _get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Get user embedding from feature store."""
        try:
            feature_store = self._get_feature_store()
            # Use get_feature_group to get user features including embedding
            features = feature_store.get_feature_group("user_features", f"user:{user_id}")
            return features.get("embedding") if features else None
        except Exception as e:
            logger.debug(f"Could not get user embedding for user {user_id}: {e}")
            return None
    
    def _compute_product_score(
        self,
        product: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> float:
        """Compute personalization score for a product."""
        score = 1.0
        
        preferences = profile.get("preferences", {})
        
        # Category match
        preferred_categories = preferences.get("categories", [])
        if product.get("category_id") in preferred_categories:
            idx = preferred_categories.index(product["category_id"])
            score *= (1.5 - idx * 0.1)  # Higher boost for top preferences
        
        # Brand match
        preferred_brands = preferences.get("brands", [])
        if product.get("brand") in preferred_brands:
            score *= 1.3
        
        # Price range match
        price_prefs = preferences.get("price_range", {})
        product_price = product.get("price", 0)
        if price_prefs:
            min_price = price_prefs.get("min", 0)
            max_price = price_prefs.get("max", float("inf"))
            if min_price <= product_price <= max_price:
                score *= 1.2
        
        # Recency boost
        if product.get("is_new"):
            score *= 1.1
        
        # Rating boost
        rating = product.get("rating", 0)
        if rating > 4:
            score *= 1.1
        
        return score
    
    def _get_category_name(self, category_id: int) -> str:
        """Get category name."""
        try:
            from apps.catalog.models import Category
            cat = Category.objects.get(id=category_id)
            return cat.name
        except Exception:
            return f"Category {category_id}"
    
    def _get_new_arrivals(self, num: int) -> List[Dict[str, Any]]:
        """Get new arrival products."""
        try:
            from apps.catalog.models import Product
            
            products = Product.objects.filter(
                is_active=True,
                created_at__gte=timezone.now() - timedelta(days=14)
            ).order_by("-created_at").values(
                "id", "name", "price", "image", "category_id"
            )[:num]
            
            return list(products)
        except Exception:
            return []
    
    def _get_available_banners(self) -> List[Dict[str, Any]]:
        """Get available promotional banners."""
        # Would fetch from banner/campaign system
        return []
    
    def _get_user_cart(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user's cart items."""
        try:
            from apps.cart.models import CartItem
            
            items = CartItem.objects.filter(
                cart__user_id=user_id
            ).select_related("product").values(
                "product_id", "quantity", "product__price", "product__name"
            )
            
            return [
                {
                    "product_id": item["product_id"],
                    "quantity": item["quantity"],
                    "price": float(item["product__price"]),
                    "name": item["product__name"],
                }
                for item in items
            ]
        except Exception:
            return []
    
    def _get_cart_age_hours(self, user_id: int) -> float:
        """Get how old the cart is in hours."""
        try:
            from apps.cart.models import Cart
            
            cart = Cart.objects.filter(user_id=user_id).first()
            if cart:
                age = timezone.now() - cart.updated_at
                return age.total_seconds() / 3600
            return 0
        except Exception:
            return 0
    
    def _get_user_churn_features(self, user_id: int) -> Dict[str, Any]:
        """Get features for churn prediction."""
        behavior = self._get_user_behavior(user_id)
        
        return {
            "days_since_last_visit": behavior.get("days_since_last_visit", 0),
            "total_orders": behavior.get("total_orders", 0),
            "avg_order_value": behavior.get("avg_order_value", 0),
            # Add more features as needed
        }
    
    def _get_churn_risk_level(self, probability: float) -> str:
        """Convert probability to risk level."""
        if probability > 0.8:
            return "critical"
        elif probability > 0.6:
            return "high"
        elif probability > 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_retention_actions(
        self,
        prediction: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get retention actions based on churn prediction."""
        actions = []
        
        probability = prediction.get("churn_probability", 0)
        reasons = prediction.get("churn_reasons", [])
        
        if probability > 0.5:
            actions.append({
                "type": "discount_offer",
                "description": "Send personalized discount",
                "discount": "15%" if probability > 0.7 else "10%",
            })
        
        if "low_engagement" in reasons:
            actions.append({
                "type": "email_campaign",
                "description": "Re-engagement email series",
            })
        
        if "price_sensitivity" in reasons:
            actions.append({
                "type": "price_alert",
                "description": "Set up price drop alerts",
            })
        
        return actions
    
    def _simple_churn_assessment(self, user_id: int) -> Dict[str, Any]:
        """Simple rule-based churn assessment."""
        behavior = self._get_user_behavior(user_id)
        
        days_inactive = behavior.get("days_since_last_visit", 0)
        
        if days_inactive > 60:
            probability = 0.8
            risk_level = "high"
        elif days_inactive > 30:
            probability = 0.5
            risk_level = "medium"
        elif days_inactive > 14:
            probability = 0.3
            risk_level = "low"
        else:
            probability = 0.1
            risk_level = "low"
        
        return {
            "user_id": user_id,
            "churn_probability": probability,
            "risk_level": risk_level,
            "days_inactive": days_inactive,
        }
