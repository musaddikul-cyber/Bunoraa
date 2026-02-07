"""
Analytics Service

Django service for ML-powered analytics.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from django.core.cache import cache
    from django.conf import settings
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    cache = None

import numpy as np

logger = logging.getLogger("bunoraa.ml.services.analytics")


class AnalyticsService:
    """
    Service for ML-powered analytics and insights.
    """
    
    def __init__(self):
        self._models = {}
        self._cache_ttl = 3600
        self._model_registry = None
    
    def _get_registry(self):
        if self._model_registry is None:
            from ..core.registry import ModelRegistry
            self._model_registry = ModelRegistry()
        return self._model_registry
    
    def _get_model(self, model_name: str):
        if model_name not in self._models:
            registry = self._get_registry()
            self._models[model_name] = registry.get_latest(model_name)
        return self._models[model_name]
    
    def get_demand_forecast(
        self,
        product_ids: Optional[List[int]] = None,
        horizon_days: int = 14,
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Get demand forecasts for products.
        
        Args:
            product_ids: Products to forecast (None = all)
            horizon_days: Forecast horizon in days
            include_confidence: Include confidence intervals
        
        Returns:
            Forecasts by product
        """
        cache_key = f"forecast:demand:{horizon_days}"
        
        if cache and not product_ids:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        forecasts = {}
        
        try:
            model = self._get_model("demand_forecaster")
            
            if model:
                # Use ML model
                for product_id in (product_ids or self._get_active_product_ids()):
                    forecast = model.forecast(
                        product_id=product_id,
                        horizon=horizon_days
                    )
                    
                    forecasts[product_id] = {
                        "dates": forecast["dates"],
                        "predicted_demand": forecast["predictions"],
                    }
                    
                    if include_confidence and "quantiles" in forecast:
                        forecasts[product_id]["lower_bound"] = forecast["quantiles"]["0.1"]
                        forecasts[product_id]["upper_bound"] = forecast["quantiles"]["0.9"]
            else:
                # Fallback: Simple moving average
                forecasts = self._simple_demand_forecast(product_ids, horizon_days)
            
        except Exception as e:
            logger.error(f"Demand forecast failed: {e}")
            forecasts = self._simple_demand_forecast(product_ids, horizon_days)
        
        if cache and not product_ids:
            cache.set(cache_key, forecasts, self._cache_ttl)
        
        return forecasts
    
    def get_price_recommendations(
        self,
        product_ids: Optional[List[int]] = None,
        optimization_goal: str = "revenue"
    ) -> Dict[str, Any]:
        """
        Get price optimization recommendations.
        
        Args:
            product_ids: Products to optimize
            optimization_goal: "revenue", "profit", or "volume"
        
        Returns:
            Price recommendations by product
        """
        recommendations = {}
        
        try:
            model = self._get_model("price_optimizer")
            
            if model:
                for product_id in (product_ids or self._get_active_product_ids()[:100]):
                    rec = model.optimize_price(
                        product_id=product_id,
                        goal=optimization_goal
                    )
                    
                    recommendations[product_id] = {
                        "current_price": rec["current_price"],
                        "recommended_price": rec["optimal_price"],
                        "expected_change": rec["expected_change"],
                        "elasticity": rec.get("elasticity"),
                    }
            else:
                # Fallback: Simple rules
                recommendations = self._simple_price_recommendations(product_ids)
                
        except Exception as e:
            logger.error(f"Price optimization failed: {e}")
            recommendations = self._simple_price_recommendations(product_ids)
        
        return recommendations
    
    def segment_customers(
        self,
        num_segments: int = 5
    ) -> Dict[str, Any]:
        """
        Segment customers using ML clustering.
        
        Args:
            num_segments: Number of segments
        
        Returns:
            Customer segments with profiles
        """
        cache_key = f"analytics:segments:{num_segments}"
        
        if cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        segments = {}
        
        try:
            from apps.accounts.models import User
            from apps.orders.models import Order
            from django.db.models import Count, Sum, Avg
            
            # Get customer features
            customers = User.objects.filter(
                is_active=True
            ).annotate(
                order_count=Count("orders"),
                total_spent=Sum("orders__total_amount"),
                avg_order_value=Avg("orders__total_amount"),
            ).values("id", "order_count", "total_spent", "avg_order_value")
            
            customer_list = list(customers)
            
            if len(customer_list) < num_segments:
                return {"error": "Not enough customers for segmentation"}
            
            # Extract features
            features = np.array([
                [
                    c["order_count"] or 0,
                    float(c["total_spent"] or 0),
                    float(c["avg_order_value"] or 0),
                ]
                for c in customer_list
            ])
            
            # Normalize
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            
            # Simple K-means
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=num_segments, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            # Build segments
            segment_profiles = defaultdict(lambda: {
                "customers": [],
                "avg_orders": 0,
                "avg_spent": 0,
                "avg_order_value": 0,
            })
            
            for i, customer in enumerate(customer_list):
                segment_id = int(labels[i])
                segment_profiles[segment_id]["customers"].append(customer["id"])
            
            # Compute segment profiles
            for segment_id, profile in segment_profiles.items():
                customer_ids = profile["customers"]
                
                segment_customers = [
                    c for c in customer_list
                    if c["id"] in customer_ids
                ]
                
                profile["size"] = len(customer_ids)
                profile["avg_orders"] = np.mean([c["order_count"] or 0 for c in segment_customers])
                profile["avg_spent"] = np.mean([float(c["total_spent"] or 0) for c in segment_customers])
                profile["avg_order_value"] = np.mean([float(c["avg_order_value"] or 0) for c in segment_customers])
                
                # Assign segment name based on characteristics
                profile["name"] = self._name_segment(profile)
            
            segments = dict(segment_profiles)
            
        except Exception as e:
            logger.error(f"Customer segmentation failed: {e}")
        
        if cache and segments:
            cache.set(cache_key, segments, self._cache_ttl * 24)
        
        return segments
    
    def get_product_insights(
        self,
        product_id: int
    ) -> Dict[str, Any]:
        """
        Get ML-powered insights for a product.
        
        Args:
            product_id: Product ID
        
        Returns:
            Product insights
        """
        insights = {}
        
        try:
            # Demand forecast
            forecast = self.get_demand_forecast([product_id], horizon_days=7)
            insights["demand_forecast"] = forecast.get(product_id, {})
            
            # Price recommendation
            price_rec = self.get_price_recommendations([product_id])
            insights["price_recommendation"] = price_rec.get(product_id, {})
            
            # Customer segments buying this product
            insights["customer_segments"] = self._get_product_customer_segments(product_id)
            
            # Trending status
            insights["trending"] = self._is_product_trending(product_id)
            
            # Similar products performance
            insights["similar_products_performance"] = self._get_similar_products_performance(product_id)
            
        except Exception as e:
            logger.error(f"Product insights failed: {e}")
        
        return insights
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get key metrics for analytics dashboard."""
        cache_key = "analytics:dashboard"
        
        if cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        metrics = {}
        
        try:
            from apps.orders.models import Order, OrderItem
            from apps.accounts.models import User
            from apps.catalog.models import Product
            from django.db.models import Count, Sum, Avg
            
            today = datetime.now().date()
            last_week = today - timedelta(days=7)
            last_month = today - timedelta(days=30)
            
            # Revenue
            revenue_today = Order.objects.filter(
                created_at__date=today
            ).aggregate(total=Sum("total_amount"))["total"] or 0
            
            revenue_week = Order.objects.filter(
                created_at__date__gte=last_week
            ).aggregate(total=Sum("total_amount"))["total"] or 0
            
            revenue_month = Order.objects.filter(
                created_at__date__gte=last_month
            ).aggregate(total=Sum("total_amount"))["total"] or 0
            
            # Orders
            orders_today = Order.objects.filter(created_at__date=today).count()
            orders_week = Order.objects.filter(created_at__date__gte=last_week).count()
            
            # Users
            new_users_week = User.objects.filter(
                date_joined__date__gte=last_week
            ).count()
            
            active_users = User.objects.filter(
                last_login__date__gte=last_week
            ).count()
            
            # Top products
            top_products = OrderItem.objects.filter(
                order__created_at__date__gte=last_week
            ).values("product__name").annotate(
                sales=Sum("quantity")
            ).order_by("-sales")[:5]
            
            # Churn prediction summary
            churn_summary = self._get_churn_summary()
            
            metrics = {
                "revenue": {
                    "today": float(revenue_today),
                    "week": float(revenue_week),
                    "month": float(revenue_month),
                },
                "orders": {
                    "today": orders_today,
                    "week": orders_week,
                },
                "users": {
                    "new_week": new_users_week,
                    "active": active_users,
                },
                "top_products": list(top_products),
                "churn_risk": churn_summary,
            }
            
        except Exception as e:
            logger.error(f"Dashboard metrics failed: {e}")
        
        if cache and metrics:
            cache.set(cache_key, metrics, 300)  # 5 minutes
        
        return metrics
    
    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------
    
    def _get_active_product_ids(self) -> List[int]:
        """Get active product IDs."""
        try:
            from apps.catalog.models import Product
            return list(
                Product.objects.filter(is_active=True).values_list("id", flat=True)[:1000]
            )
        except Exception:
            return []
    
    def _simple_demand_forecast(
        self,
        product_ids: Optional[List[int]],
        horizon: int
    ) -> Dict[str, Any]:
        """Simple moving average forecast."""
        forecasts = {}
        
        try:
            from apps.orders.models import OrderItem
            from django.db.models import Sum
            from django.db.models.functions import TruncDate
            
            for product_id in (product_ids or self._get_active_product_ids()[:100]):
                # Get last 30 days sales
                history = OrderItem.objects.filter(
                    product_id=product_id,
                    order__created_at__gte=datetime.now() - timedelta(days=30)
                ).annotate(
                    date=TruncDate("order__created_at")
                ).values("date").annotate(
                    sales=Sum("quantity")
                ).order_by("date")
                
                if history:
                    avg_sales = np.mean([h["sales"] for h in history])
                else:
                    avg_sales = 0
                
                dates = [
                    (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(1, horizon + 1)
                ]
                
                forecasts[product_id] = {
                    "dates": dates,
                    "predicted_demand": [avg_sales] * horizon,
                }
        
        except Exception as e:
            logger.debug(f"Simple forecast failed: {e}")
        
        return forecasts
    
    def _simple_price_recommendations(
        self,
        product_ids: Optional[List[int]]
    ) -> Dict[str, Any]:
        """Simple price recommendations based on rules."""
        recommendations = {}
        
        try:
            from apps.catalog.models import Product
            
            products = Product.objects.filter(
                id__in=product_ids or []
            ).values("id", "price", "stock_quantity")
            
            for product in products:
                current_price = float(product["price"])
                stock = product["stock_quantity"]
                
                # Simple rules
                if stock > 100:
                    # High stock - suggest discount
                    recommended = current_price * 0.9
                elif stock < 10:
                    # Low stock - can increase price
                    recommended = current_price * 1.05
                else:
                    recommended = current_price
                
                recommendations[product["id"]] = {
                    "current_price": current_price,
                    "recommended_price": round(recommended, 2),
                    "expected_change": None,
                }
        
        except Exception:
            pass
        
        return recommendations
    
    def _name_segment(self, profile: Dict) -> str:
        """Name a customer segment based on its characteristics."""
        if profile["avg_spent"] > 1000:
            return "VIP Customers"
        elif profile["avg_orders"] > 10:
            return "Loyal Customers"
        elif profile["avg_spent"] > 100:
            return "Regular Customers"
        elif profile["avg_orders"] >= 1:
            return "New Customers"
        else:
            return "Inactive Customers"
    
    def _get_product_customer_segments(self, product_id: int) -> Dict[str, int]:
        """Get customer segments that buy a product."""
        try:
            from apps.orders.models import OrderItem
            
            # Get customers who bought this product
            buyer_ids = OrderItem.objects.filter(
                product_id=product_id
            ).values_list("order__user_id", flat=True).distinct()
            
            # TODO: Cross-reference with segments
            return {"total_buyers": len(set(buyer_ids))}
        except Exception:
            return {}
    
    def _is_product_trending(self, product_id: int) -> bool:
        """Check if product is trending."""
        try:
            from apps.orders.models import OrderItem
            from django.db.models import Count
            
            now = datetime.now()
            
            # This week vs last week
            this_week = OrderItem.objects.filter(
                product_id=product_id,
                order__created_at__gte=now - timedelta(days=7)
            ).count()
            
            last_week = OrderItem.objects.filter(
                product_id=product_id,
                order__created_at__gte=now - timedelta(days=14),
                order__created_at__lt=now - timedelta(days=7)
            ).count()
            
            # Trending if 50% increase
            return this_week > last_week * 1.5 and this_week > 5
        except Exception:
            return False
    
    def _get_similar_products_performance(self, product_id: int) -> Dict[str, Any]:
        """Get performance of similar products."""
        try:
            from apps.catalog.models import Product
            from apps.orders.models import OrderItem
            from django.db.models import Sum
            
            product = Product.objects.get(id=product_id)
            
            # Get similar products in same category
            similar = Product.objects.filter(
                category_id=product.category_id,
                is_active=True
            ).exclude(id=product_id)[:5]
            
            performance = {}
            for p in similar:
                sales = OrderItem.objects.filter(
                    product_id=p.id,
                    order__created_at__gte=datetime.now() - timedelta(days=30)
                ).aggregate(total=Sum("quantity"))["total"] or 0
                
                performance[p.id] = {
                    "name": p.name,
                    "price": float(p.price),
                    "sales_30d": sales,
                }
            
            return performance
        except Exception:
            return {}
    
    def _get_churn_summary(self) -> Dict[str, Any]:
        """Get churn risk summary."""
        try:
            model = self._get_model("churn_predictor")
            
            if model:
                # TODO: Implement batch prediction
                return {
                    "high_risk": 0,
                    "medium_risk": 0,
                    "low_risk": 0,
                }
            
            return {}
        except Exception:
            return {}
