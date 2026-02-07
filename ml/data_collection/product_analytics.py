"""
Product Analytics Collector

Collects comprehensive product-level analytics for ML training.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass, asdict, field

try:
    from django.conf import settings
    from django.db.models import Sum, Avg, Count, F, Q, StdDev
    from django.utils import timezone
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("bunoraa.ml.product_analytics")


@dataclass
class ProductMLFeatures:
    """Product features for ML training."""
    
    product_id: int
    
    # Text Features
    title_length: int = 0
    title_word_count: int = 0
    description_length: int = 0
    description_word_count: int = 0
    description_sentence_count: int = 0
    has_bengali_text: bool = False
    
    # Media Features
    image_count: int = 0
    has_video: bool = False
    has_360_view: bool = False
    primary_image_quality_score: float = 0.0
    
    # Pricing Features
    price: float = 0.0
    original_price: float = 0.0
    discount_amount: float = 0.0
    discount_percent: float = 0.0
    price_tier: str = ""  # budget, mid, premium, luxury
    
    # Category Features
    category_id: int = 0
    category_depth: int = 0
    category_product_count: int = 0
    
    # Availability Features
    in_stock: bool = True
    stock_quantity: int = 0
    low_stock: bool = False
    restock_rate: float = 0.0
    
    # Status Flags
    is_new_arrival: bool = False
    is_best_seller: bool = False
    is_spotlight: bool = False
    is_featured: bool = False
    is_preorder: bool = False
    
    # Variant Features
    has_variants: bool = False
    variant_count: int = 0
    color_variant_count: int = 0
    size_variant_count: int = 0
    
    # Review Features
    review_count: int = 0
    average_rating: float = 0.0
    rating_std_dev: float = 0.0
    positive_review_ratio: float = 0.0
    has_verified_reviews: bool = False
    
    # Performance Metrics
    view_count_7d: int = 0
    view_count_30d: int = 0
    add_to_cart_count_7d: int = 0
    add_to_cart_count_30d: int = 0
    purchase_count_7d: int = 0
    purchase_count_30d: int = 0
    wishlist_count: int = 0
    share_count: int = 0
    
    # Conversion Metrics
    view_to_cart_rate: float = 0.0
    cart_to_purchase_rate: float = 0.0
    overall_conversion_rate: float = 0.0
    
    # Engagement Metrics
    avg_time_on_page: float = 0.0
    avg_scroll_depth: float = 0.0
    bounce_rate: float = 0.0
    image_click_rate: float = 0.0
    review_view_rate: float = 0.0
    
    # Temporal Features
    days_since_created: int = 0
    days_since_last_sale: int = 0
    sales_velocity: float = 0.0  # sales per day
    
    # Computed Scores
    popularity_score: float = 0.0
    quality_score: float = 0.0
    value_score: float = 0.0


class ProductAnalyticsCollector:
    """
    Collects and computes product analytics for ML.
    """
    
    def __init__(self):
        self.redis_client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize connections."""
        if REDIS_AVAILABLE:
            try:
                redis_url = getattr(settings, 'ML_REDIS_URL', 'redis://localhost:6379/1')
                self.redis_client = redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
    
    def collect_product_features(self, product_id: int) -> ProductMLFeatures:
        """
        Collect comprehensive ML features for a product.
        
        Args:
            product_id: Product ID
        
        Returns:
            ProductMLFeatures with computed metrics
        """
        features = ProductMLFeatures(product_id=product_id)
        
        try:
            from apps.catalog.models import Product
            
            product = Product.objects.select_related('category').prefetch_related(
                'images', 'variants', 'reviews'
            ).get(id=product_id)
            
            # Text features
            features.title_length = len(product.name or product.title or '')
            features.title_word_count = len((product.name or product.title or '').split())
            
            description = product.description or ''
            features.description_length = len(description)
            features.description_word_count = len(description.split())
            features.description_sentence_count = self._count_sentences(description)
            features.has_bengali_text = self._has_bengali(product.name or description)
            
            # Media features
            if hasattr(product, 'images'):
                features.image_count = product.images.count()
            features.has_video = getattr(product, 'has_video', False)
            
            # Pricing features
            price = float(product.price or 0)
            original_price = float(getattr(product, 'original_price', price) or price)
            
            features.price = price
            features.original_price = original_price
            features.discount_amount = original_price - price if original_price > price else 0
            features.discount_percent = (features.discount_amount / original_price * 100) if original_price > 0 else 0
            features.price_tier = self._get_price_tier(price)
            
            # Category features
            if product.category:
                features.category_id = product.category.id
                features.category_depth = self._get_category_depth(product.category)
                features.category_product_count = Product.objects.filter(
                    category=product.category
                ).count()
            
            # Availability
            features.in_stock = getattr(product, 'in_stock', True)
            features.stock_quantity = getattr(product, 'stock_quantity', 0) or getattr(product, 'stock', 0)
            features.low_stock = features.stock_quantity > 0 and features.stock_quantity < 10
            
            # Status flags
            features.is_new_arrival = getattr(product, 'is_new_arrival', False) or getattr(product, 'is_new', False)
            features.is_best_seller = getattr(product, 'is_best_seller', False)
            features.is_spotlight = getattr(product, 'is_spotlight', False)
            features.is_featured = getattr(product, 'is_featured', False)
            features.is_preorder = getattr(product, 'is_preorder', False)
            
            # Variants
            if hasattr(product, 'variants'):
                variants = product.variants.all()
                features.has_variants = variants.exists()
                features.variant_count = variants.count()
                
                # Count by type
                features.color_variant_count = len(set(
                    v.color for v in variants if hasattr(v, 'color') and v.color
                ))
                features.size_variant_count = len(set(
                    v.size for v in variants if hasattr(v, 'size') and v.size
                ))
            
            # Reviews
            if hasattr(product, 'reviews'):
                reviews = product.reviews.all()
                features.review_count = reviews.count()
                
                if features.review_count > 0:
                    ratings = [r.rating for r in reviews]
                    features.average_rating = sum(ratings) / len(ratings)
                    
                    # Standard deviation
                    if len(ratings) > 1:
                        mean = features.average_rating
                        variance = sum((r - mean) ** 2 for r in ratings) / len(ratings)
                        features.rating_std_dev = variance ** 0.5
                    
                    # Positive ratio (4+ stars)
                    positive = sum(1 for r in ratings if r >= 4)
                    features.positive_review_ratio = positive / len(ratings)
            
            # Temporal features
            created = getattr(product, 'created_at', None)
            if created:
                now = timezone.now()
                features.days_since_created = (now - created).days
            
        except Exception as e:
            logger.error(f"Failed to get product features: {e}")
        
        # Get performance metrics from Redis/Analytics
        performance = self._get_performance_metrics(product_id)
        features.view_count_7d = performance.get('views_7d', 0)
        features.view_count_30d = performance.get('views_30d', 0)
        features.add_to_cart_count_7d = performance.get('carts_7d', 0)
        features.add_to_cart_count_30d = performance.get('carts_30d', 0)
        features.purchase_count_7d = performance.get('purchases_7d', 0)
        features.purchase_count_30d = performance.get('purchases_30d', 0)
        features.wishlist_count = performance.get('wishlists', 0)
        features.share_count = performance.get('shares', 0)
        
        # Conversion metrics
        if features.view_count_30d > 0:
            features.view_to_cart_rate = features.add_to_cart_count_30d / features.view_count_30d
            features.overall_conversion_rate = features.purchase_count_30d / features.view_count_30d
        
        if features.add_to_cart_count_30d > 0:
            features.cart_to_purchase_rate = features.purchase_count_30d / features.add_to_cart_count_30d
        
        # Engagement metrics
        engagement = self._get_engagement_metrics(product_id)
        features.avg_time_on_page = engagement.get('avg_time', 0)
        features.avg_scroll_depth = engagement.get('avg_scroll', 0)
        features.bounce_rate = engagement.get('bounce_rate', 0)
        features.image_click_rate = engagement.get('image_clicks', 0)
        features.review_view_rate = engagement.get('review_views', 0)
        
        # Sales velocity
        if features.days_since_created > 0:
            features.sales_velocity = features.purchase_count_30d / min(features.days_since_created, 30)
        
        # Compute scores
        features.popularity_score = self._compute_popularity_score(features)
        features.quality_score = self._compute_quality_score(features)
        features.value_score = self._compute_value_score(features)
        
        # Cache features
        self._cache_features(features)
        
        return features
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        import re
        if not text:
            return 0
        sentences = re.split(r'[.!?।]+', text)
        return len([s for s in sentences if s.strip()])
    
    def _has_bengali(self, text: str) -> bool:
        """Check if text contains Bengali characters."""
        if not text:
            return False
        # Bengali Unicode range
        for char in text:
            if '\u0980' <= char <= '\u09FF':
                return True
        return False
    
    def _get_price_tier(self, price: float) -> str:
        """Determine price tier based on BDT price."""
        if price < 500:
            return "budget"
        elif price < 2000:
            return "mid"
        elif price < 10000:
            return "premium"
        else:
            return "luxury"
    
    def _get_category_depth(self, category) -> int:
        """Get category depth in hierarchy."""
        depth = 1
        while hasattr(category, 'parent') and category.parent:
            depth += 1
            category = category.parent
            if depth > 10:  # Safety limit
                break
        return depth
    
    def _get_performance_metrics(self, product_id: int) -> Dict[str, int]:
        """Get product performance metrics."""
        metrics = {
            'views_7d': 0,
            'views_30d': 0,
            'carts_7d': 0,
            'carts_30d': 0,
            'purchases_7d': 0,
            'purchases_30d': 0,
            'wishlists': 0,
            'shares': 0,
        }
        
        try:
            from apps.analytics.models import ProductView, Event
            from apps.orders.models import OrderItem
            from apps.commerce.models import WishlistItem
            
            now = timezone.now()
            seven_days_ago = now - timedelta(days=7)
            thirty_days_ago = now - timedelta(days=30)
            
            # Views
            metrics['views_7d'] = ProductView.objects.filter(
                product_id=product_id,
                created_at__gte=seven_days_ago
            ).count()
            
            metrics['views_30d'] = ProductView.objects.filter(
                product_id=product_id,
                created_at__gte=thirty_days_ago
            ).count()
            
            # Add to carts
            metrics['carts_7d'] = Event.objects.filter(
                event_type='add_to_cart',
                data__product_id=product_id,
                created_at__gte=seven_days_ago
            ).count()
            
            metrics['carts_30d'] = Event.objects.filter(
                event_type='add_to_cart',
                data__product_id=product_id,
                created_at__gte=thirty_days_ago
            ).count()
            
            # Purchases
            metrics['purchases_7d'] = OrderItem.objects.filter(
                product_id=product_id,
                order__status__in=['completed', 'delivered'],
                order__created_at__gte=seven_days_ago
            ).count()
            
            metrics['purchases_30d'] = OrderItem.objects.filter(
                product_id=product_id,
                order__status__in=['completed', 'delivered'],
                order__created_at__gte=thirty_days_ago
            ).count()
            
            # Wishlists
            metrics['wishlists'] = WishlistItem.objects.filter(
                product_id=product_id
            ).count()
            
            # Shares
            metrics['shares'] = Event.objects.filter(
                event_type='share',
                data__product_id=product_id
            ).count()
            
        except Exception as e:
            logger.debug(f"Failed to get performance metrics: {e}")
        
        return metrics
    
    def _get_engagement_metrics(self, product_id: int) -> Dict[str, float]:
        """Get product engagement metrics."""
        metrics = {
            'avg_time': 0,
            'avg_scroll': 0,
            'bounce_rate': 0,
            'image_clicks': 0,
            'review_views': 0,
        }
        
        try:
            from apps.analytics.models import ProductView, Event
            
            now = timezone.now()
            since = now - timedelta(days=30)
            
            views = ProductView.objects.filter(
                product_id=product_id,
                created_at__gte=since
            ).aggregate(
                avg_time=Avg('time_on_page'),
                avg_scroll=Avg('scroll_depth'),
                total=Count('id'),
            )
            
            metrics['avg_time'] = views.get('avg_time') or 0
            metrics['avg_scroll'] = views.get('avg_scroll') or 0
            
            total_views = views.get('total') or 0
            
            if total_views > 0:
                # Bounce rate (views < 10 seconds)
                bounces = ProductView.objects.filter(
                    product_id=product_id,
                    created_at__gte=since,
                    time_on_page__lt=10
                ).count()
                metrics['bounce_rate'] = bounces / total_views
                
                # Image clicks
                image_clicks = Event.objects.filter(
                    event_type='product_image_click',
                    data__product_id=product_id,
                    created_at__gte=since
                ).count()
                metrics['image_clicks'] = image_clicks / total_views
                
                # Review views
                review_views = Event.objects.filter(
                    event_type='product_review_view',
                    data__product_id=product_id,
                    created_at__gte=since
                ).count()
                metrics['review_views'] = review_views / total_views
                
        except Exception as e:
            logger.debug(f"Failed to get engagement metrics: {e}")
        
        return metrics
    
    def _compute_popularity_score(self, features: ProductMLFeatures) -> float:
        """Compute product popularity score (0-100)."""
        score = 0.0
        
        # Recent views (up to 25 points)
        score += min(features.view_count_7d / 10, 25)
        
        # Sales velocity (up to 25 points)
        score += min(features.sales_velocity * 10, 25)
        
        # Cart adds (up to 20 points)
        score += min(features.add_to_cart_count_7d / 5, 20)
        
        # Wishlists (up to 15 points)
        score += min(features.wishlist_count / 5, 15)
        
        # Status bonuses (up to 15 points)
        if features.is_best_seller:
            score += 10
        if features.is_featured or features.is_spotlight:
            score += 5
        
        return min(100, score)
    
    def _compute_quality_score(self, features: ProductMLFeatures) -> float:
        """Compute product quality score (0-100)."""
        score = 0.0
        
        # Rating (up to 30 points)
        score += features.average_rating * 6
        
        # Review count (up to 20 points)
        score += min(features.review_count * 2, 20)
        
        # Positive ratio (up to 15 points)
        score += features.positive_review_ratio * 15
        
        # Images (up to 15 points)
        score += min(features.image_count * 3, 15)
        
        # Description quality (up to 10 points)
        if features.description_length > 200:
            score += 5
        if features.description_sentence_count >= 3:
            score += 5
        
        # Low return rate would add more points (if we had data)
        
        # Video bonus (up to 10 points)
        if features.has_video:
            score += 10
        
        return min(100, score)
    
    def _compute_value_score(self, features: ProductMLFeatures) -> float:
        """Compute product value score (0-100) - quality relative to price."""
        if features.quality_score == 0:
            return 50.0  # Neutral
        
        # Value = quality per price unit
        # Normalize by price tier expectations
        tier_multipliers = {
            'budget': 0.5,
            'mid': 1.0,
            'premium': 1.5,
            'luxury': 2.0,
        }
        
        multiplier = tier_multipliers.get(features.price_tier, 1.0)
        
        # Expected quality based on price tier
        expected_quality = 50 * multiplier
        
        # Value score: how much quality exceeds expectations
        value_ratio = features.quality_score / expected_quality if expected_quality > 0 else 1
        
        # Convert to 0-100 scale
        score = 50 + (value_ratio - 1) * 25
        
        # Discount bonus
        if features.discount_percent > 0:
            score += min(features.discount_percent / 5, 20)
        
        return max(0, min(100, score))
    
    def _cache_features(self, features: ProductMLFeatures):
        """Cache product features in Redis."""
        if not self.redis_client:
            return
        
        try:
            import json
            
            key = f"ml:product_features:{features.product_id}"
            data = asdict(features)
            
            self.redis_client.setex(
                key,
                86400,  # 24 hours
                json.dumps(data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache features: {e}")
    
    def get_cached_features(self, product_id: int) -> Optional[ProductMLFeatures]:
        """Get cached product features."""
        if not self.redis_client:
            return None
        
        try:
            import json
            
            key = f"ml:product_features:{product_id}"
            data = self.redis_client.get(key)
            
            if data:
                features_dict = json.loads(data)
                return ProductMLFeatures(**features_dict)
                
        except Exception as e:
            logger.debug(f"Failed to get cached features: {e}")
        
        return None
    
    def collect_all_features(self, batch_size: int = 100) -> int:
        """
        Collect features for all active products.
        
        Returns:
            Number of products processed
        """
        collected = 0
        
        try:
            from apps.catalog.models import Product
            
            products = Product.objects.filter(
                is_active=True
            ).values_list('id', flat=True)
            
            for product_id in products:
                try:
                    self.collect_product_features(product_id)
                    collected += 1
                except Exception as e:
                    logger.error(f"Failed to collect features for product {product_id}: {e}")
                
        except Exception as e:
            logger.error(f"Failed to collect all features: {e}")
        
        return collected
    
    def export_training_data(
        self,
        output_path: str,
        days: int = 90,
    ) -> int:
        """
        Export product features as training data.
        
        Args:
            output_path: Path to save CSV/Parquet
            days: Number of days of data to include
        
        Returns:
            Number of records exported
        """
        import csv
        
        try:
            from apps.catalog.models import Product
            
            products = Product.objects.filter(is_active=True)
            
            records = []
            for product in products:
                try:
                    features = self.collect_product_features(product.id)
                    records.append(asdict(features))
                except Exception as e:
                    logger.error(f"Failed to export product {product.id}: {e}")
            
            if records:
                # Write to CSV
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)
            
            return len(records)
            
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            return 0
