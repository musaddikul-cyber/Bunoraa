"""
User Profile Collector

Collects comprehensive user profile data for ML training.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass, asdict, field

try:
    from django.conf import settings
    from django.db.models import Sum, Avg, Count, F, Q
    from django.utils import timezone
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("bunoraa.ml.user_profile")


@dataclass
class UserBehaviorProfile:
    """User behavior profile for ML."""
    
    user_id: int
    
    # Session Behavior
    avg_session_duration_seconds: float = 0.0
    avg_pages_per_session: float = 0.0
    avg_products_viewed_per_session: float = 0.0
    bounce_rate: float = 0.0
    
    # Time Patterns
    preferred_visit_hour: int = 0  # 0-23
    preferred_visit_day: int = 0   # 0-6 (Monday=0)
    weekend_activity_ratio: float = 0.0
    morning_activity_ratio: float = 0.0
    evening_activity_ratio: float = 0.0
    
    # Browsing Patterns
    search_to_view_ratio: float = 0.0
    category_to_search_ratio: float = 0.0
    filter_usage_rate: float = 0.0
    comparison_usage_rate: float = 0.0
    
    # Engagement Depth
    avg_scroll_depth: float = 0.0
    avg_time_on_product_page: float = 0.0
    review_read_rate: float = 0.0
    image_zoom_rate: float = 0.0
    video_watch_rate: float = 0.0
    
    # Purchase Patterns
    cart_abandonment_rate: float = 0.0
    wishlist_to_cart_rate: float = 0.0
    time_to_first_purchase_days: int = 0
    avg_time_between_purchases_days: int = 0
    
    # Category Affinity (normalized 0-1)
    category_affinities: Dict[int, float] = field(default_factory=dict)
    brand_affinities: Dict[str, float] = field(default_factory=dict)
    
    # Price Sensitivity
    price_sensitivity_score: float = 0.0
    avg_discount_percentage_bought: float = 0.0
    full_price_purchase_ratio: float = 0.0
    
    # Social Behavior
    share_rate: float = 0.0
    review_rate: float = 0.0
    referral_count: int = 0
    
    # Device & Platform
    mobile_usage_ratio: float = 0.0
    app_usage_ratio: float = 0.0
    
    # Computed Scores
    engagement_score: float = 0.0
    purchase_intent_score: float = 0.0
    loyalty_score: float = 0.0


class UserProfileCollector:
    """
    Collects and computes user behavior profiles.
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
    
    def collect_user_profile(self, user_id: int) -> UserBehaviorProfile:
        """
        Collect comprehensive behavior profile for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            UserBehaviorProfile with computed metrics
        """
        profile = UserBehaviorProfile(user_id=user_id)
        
        # Get session behavior
        session_stats = self._get_session_stats(user_id)
        profile.avg_session_duration_seconds = session_stats.get('avg_duration', 0)
        profile.avg_pages_per_session = session_stats.get('avg_pages', 0)
        profile.avg_products_viewed_per_session = session_stats.get('avg_products', 0)
        profile.bounce_rate = session_stats.get('bounce_rate', 0)
        
        # Get time patterns
        time_patterns = self._get_time_patterns(user_id)
        profile.preferred_visit_hour = time_patterns.get('preferred_hour', 12)
        profile.preferred_visit_day = time_patterns.get('preferred_day', 0)
        profile.weekend_activity_ratio = time_patterns.get('weekend_ratio', 0)
        profile.morning_activity_ratio = time_patterns.get('morning_ratio', 0)
        profile.evening_activity_ratio = time_patterns.get('evening_ratio', 0)
        
        # Get browsing patterns
        browsing = self._get_browsing_patterns(user_id)
        profile.search_to_view_ratio = browsing.get('search_ratio', 0)
        profile.filter_usage_rate = browsing.get('filter_rate', 0)
        
        # Get engagement depth
        engagement = self._get_engagement_metrics(user_id)
        profile.avg_scroll_depth = engagement.get('avg_scroll', 0)
        profile.avg_time_on_product_page = engagement.get('avg_product_time', 0)
        profile.review_read_rate = engagement.get('review_rate', 0)
        profile.image_zoom_rate = engagement.get('zoom_rate', 0)
        
        # Get purchase patterns
        purchase = self._get_purchase_patterns(user_id)
        profile.cart_abandonment_rate = purchase.get('abandonment_rate', 0)
        profile.wishlist_to_cart_rate = purchase.get('wishlist_to_cart', 0)
        profile.avg_time_between_purchases_days = purchase.get('avg_days_between', 0)
        
        # Get affinities
        profile.category_affinities = self._get_category_affinities(user_id)
        profile.brand_affinities = self._get_brand_affinities(user_id)
        
        # Get price sensitivity
        price = self._get_price_sensitivity(user_id)
        profile.price_sensitivity_score = price.get('sensitivity', 0.5)
        profile.avg_discount_percentage_bought = price.get('avg_discount', 0)
        profile.full_price_purchase_ratio = price.get('full_price_ratio', 0)
        
        # Get device usage
        device = self._get_device_usage(user_id)
        profile.mobile_usage_ratio = device.get('mobile_ratio', 0)
        
        # Compute scores
        profile.engagement_score = self._compute_engagement_score(profile)
        profile.purchase_intent_score = self._compute_purchase_intent(profile)
        profile.loyalty_score = self._compute_loyalty_score(user_id)
        
        # Cache profile
        self._cache_profile(profile)
        
        return profile
    
    def _get_session_stats(self, user_id: int) -> Dict[str, float]:
        """Get session statistics for user."""
        stats = {
            'avg_duration': 0,
            'avg_pages': 0,
            'avg_products': 0,
            'bounce_rate': 0,
        }
        
        try:
            from apps.analytics.models import Session, PageView
            
            sessions = Session.objects.filter(user_id=user_id).order_by('-created_at')[:100]
            
            if sessions:
                total_duration = 0
                total_pages = 0
                bounces = 0
                
                for session in sessions:
                    duration = (session.ended_at - session.created_at).total_seconds() if session.ended_at else 0
                    total_duration += duration
                    
                    pages = session.page_views.count() if hasattr(session, 'page_views') else 0
                    total_pages += pages
                    
                    if pages <= 1:
                        bounces += 1
                
                count = len(sessions)
                stats['avg_duration'] = total_duration / count
                stats['avg_pages'] = total_pages / count
                stats['bounce_rate'] = bounces / count
                
        except Exception as e:
            logger.debug(f"Failed to get session stats: {e}")
        
        return stats
    
    def _get_time_patterns(self, user_id: int) -> Dict[str, Any]:
        """Analyze user's time-based activity patterns."""
        patterns = {
            'preferred_hour': 12,
            'preferred_day': 0,
            'weekend_ratio': 0,
            'morning_ratio': 0,
            'evening_ratio': 0,
        }
        
        if not self.redis_client:
            return patterns
        
        try:
            # Get activity by hour
            hour_key = f"ml:user:{user_id}:activity_hours"
            hours = self.redis_client.hgetall(hour_key)
            
            if hours:
                max_hour = max(hours.items(), key=lambda x: int(x[1]))[0]
                patterns['preferred_hour'] = int(max_hour)
                
                total = sum(int(v) for v in hours.values())
                morning = sum(int(v) for k, v in hours.items() if 6 <= int(k) < 12)
                evening = sum(int(v) for k, v in hours.items() if 18 <= int(k) < 24)
                
                patterns['morning_ratio'] = morning / total if total > 0 else 0
                patterns['evening_ratio'] = evening / total if total > 0 else 0
            
            # Get activity by day
            day_key = f"ml:user:{user_id}:activity_days"
            days = self.redis_client.hgetall(day_key)
            
            if days:
                max_day = max(days.items(), key=lambda x: int(x[1]))[0]
                patterns['preferred_day'] = int(max_day)
                
                total = sum(int(v) for v in days.values())
                weekend = sum(int(v) for k, v in days.items() if int(k) >= 5)
                patterns['weekend_ratio'] = weekend / total if total > 0 else 0
                
        except Exception as e:
            logger.debug(f"Failed to get time patterns: {e}")
        
        return patterns
    
    def _get_browsing_patterns(self, user_id: int) -> Dict[str, float]:
        """Get user's browsing patterns."""
        patterns = {
            'search_ratio': 0,
            'filter_rate': 0,
        }
        
        try:
            from apps.analytics.models import Event
            
            now = timezone.now()
            since = now - timedelta(days=90)
            
            events = Event.objects.filter(
                user_id=user_id,
                created_at__gte=since
            ).values('event_type').annotate(count=Count('id'))
            
            event_counts = {e['event_type']: e['count'] for e in events}
            
            views = event_counts.get('product_view', 0)
            searches = event_counts.get('search', 0)
            filters = event_counts.get('filter_apply', 0)
            
            if views > 0:
                patterns['search_ratio'] = searches / views
                patterns['filter_rate'] = filters / views
                
        except Exception as e:
            logger.debug(f"Failed to get browsing patterns: {e}")
        
        return patterns
    
    def _get_engagement_metrics(self, user_id: int) -> Dict[str, float]:
        """Get engagement metrics."""
        metrics = {
            'avg_scroll': 0,
            'avg_product_time': 0,
            'review_rate': 0,
            'zoom_rate': 0,
        }
        
        try:
            from apps.analytics.models import ProductView, Event
            
            now = timezone.now()
            since = now - timedelta(days=90)
            
            # Product views with time
            views = ProductView.objects.filter(
                user_id=user_id,
                created_at__gte=since
            ).aggregate(
                avg_time=Avg('time_on_page'),
                avg_scroll=Avg('scroll_depth'),
                total=Count('id'),
            )
            
            metrics['avg_product_time'] = views.get('avg_time') or 0
            metrics['avg_scroll'] = views.get('avg_scroll') or 0
            
            total_views = views.get('total') or 0
            
            if total_views > 0:
                # Review views
                review_views = Event.objects.filter(
                    user_id=user_id,
                    event_type='product_review_view',
                    created_at__gte=since
                ).count()
                metrics['review_rate'] = review_views / total_views
                
                # Zoom rate
                zooms = Event.objects.filter(
                    user_id=user_id,
                    event_type='product_zoom',
                    created_at__gte=since
                ).count()
                metrics['zoom_rate'] = zooms / total_views
                
        except Exception as e:
            logger.debug(f"Failed to get engagement metrics: {e}")
        
        return metrics
    
    def _get_purchase_patterns(self, user_id: int) -> Dict[str, Any]:
        """Get purchase patterns."""
        patterns = {
            'abandonment_rate': 0,
            'wishlist_to_cart': 0,
            'avg_days_between': 0,
        }
        
        try:
            from apps.orders.models import Order
            from apps.analytics.models import Event
            
            now = timezone.now()
            since = now - timedelta(days=365)
            
            # Cart abandonment
            cart_starts = Event.objects.filter(
                user_id=user_id,
                event_type='add_to_cart',
                created_at__gte=since
            ).count()
            
            completed = Order.objects.filter(
                user_id=user_id,
                status__in=['completed', 'delivered'],
                created_at__gte=since
            ).count()
            
            if cart_starts > 0:
                patterns['abandonment_rate'] = 1 - (completed / cart_starts)
            
            # Time between purchases
            orders = Order.objects.filter(
                user_id=user_id,
                status__in=['completed', 'delivered']
            ).order_by('created_at').values_list('created_at', flat=True)
            
            orders = list(orders)
            if len(orders) >= 2:
                gaps = [(orders[i+1] - orders[i]).days for i in range(len(orders)-1)]
                patterns['avg_days_between'] = sum(gaps) / len(gaps)
                
        except Exception as e:
            logger.debug(f"Failed to get purchase patterns: {e}")
        
        return patterns
    
    def _get_category_affinities(self, user_id: int) -> Dict[int, float]:
        """Get category affinities normalized to 0-1."""
        affinities = {}
        
        if not self.redis_client:
            return affinities
        
        try:
            key = f"ml:user:{user_id}:categories"
            categories = self.redis_client.zrevrange(key, 0, 19, withscores=True)
            
            if categories:
                max_score = categories[0][1] if categories else 1
                affinities = {
                    int(cat): score / max_score
                    for cat, score in categories
                }
                
        except Exception as e:
            logger.debug(f"Failed to get category affinities: {e}")
        
        return affinities
    
    def _get_brand_affinities(self, user_id: int) -> Dict[str, float]:
        """Get brand affinities normalized to 0-1."""
        affinities = {}
        
        if not self.redis_client:
            return affinities
        
        try:
            key = f"ml:user:{user_id}:brands"
            brands = self.redis_client.zrevrange(key, 0, 19, withscores=True)
            
            if brands:
                max_score = brands[0][1] if brands else 1
                affinities = {
                    brand.decode() if isinstance(brand, bytes) else brand: score / max_score
                    for brand, score in brands
                }
                
        except Exception as e:
            logger.debug(f"Failed to get brand affinities: {e}")
        
        return affinities
    
    def _get_price_sensitivity(self, user_id: int) -> Dict[str, float]:
        """Calculate price sensitivity metrics."""
        metrics = {
            'sensitivity': 0.5,
            'avg_discount': 0,
            'full_price_ratio': 0,
        }
        
        try:
            from apps.orders.models import Order, OrderItem
            
            items = OrderItem.objects.filter(
                order__user_id=user_id,
                order__status__in=['completed', 'delivered']
            ).select_related('product')
            
            if items:
                total = 0
                discounted = 0
                total_discount = 0
                
                for item in items:
                    total += 1
                    if hasattr(item, 'discount_amount') and item.discount_amount > 0:
                        discounted += 1
                        total_discount += float(item.discount_amount) / float(item.price) * 100
                
                metrics['full_price_ratio'] = (total - discounted) / total if total > 0 else 0
                metrics['avg_discount'] = total_discount / discounted if discounted > 0 else 0
                
                # Sensitivity: high discount seekers = high sensitivity
                metrics['sensitivity'] = min(1.0, discounted / total + metrics['avg_discount'] / 100)
                
        except Exception as e:
            logger.debug(f"Failed to get price sensitivity: {e}")
        
        return metrics
    
    def _get_device_usage(self, user_id: int) -> Dict[str, float]:
        """Get device usage ratios."""
        usage = {'mobile_ratio': 0}
        
        if not self.redis_client:
            return usage
        
        try:
            key = f"ml:user:{user_id}:devices"
            devices = self.redis_client.zrange(key, 0, -1, withscores=True)
            
            if devices:
                total = sum(d[1] for d in devices)
                mobile = sum(d[1] for d in devices if d[0] in [b'mobile', 'mobile'])
                usage['mobile_ratio'] = mobile / total if total > 0 else 0
                
        except Exception as e:
            logger.debug(f"Failed to get device usage: {e}")
        
        return usage
    
    def _compute_engagement_score(self, profile: UserBehaviorProfile) -> float:
        """Compute overall engagement score (0-100)."""
        score = 0.0
        
        # Session depth (up to 30 points)
        if profile.avg_pages_per_session > 0:
            score += min(profile.avg_pages_per_session * 3, 30)
        
        # Time engagement (up to 25 points)
        if profile.avg_time_on_product_page > 0:
            score += min(profile.avg_time_on_product_page / 10, 25)  # 250s = full points
        
        # Scroll depth (up to 15 points)
        score += profile.avg_scroll_depth * 15
        
        # Review engagement (up to 15 points)
        score += profile.review_read_rate * 15
        
        # Low bounce rate (up to 15 points)
        score += (1 - profile.bounce_rate) * 15
        
        return min(100, score)
    
    def _compute_purchase_intent(self, profile: UserBehaviorProfile) -> float:
        """Compute purchase intent score (0-100)."""
        score = 0.0
        
        # Cart behavior (up to 40 points)
        # Lower abandonment = higher intent
        score += (1 - profile.cart_abandonment_rate) * 40
        
        # Wishlist to cart conversion (up to 20 points)
        score += profile.wishlist_to_cart_rate * 20
        
        # Engagement depth (up to 20 points)
        score += profile.avg_scroll_depth * 10
        score += min(profile.avg_time_on_product_page / 30, 10)
        
        # Recent activity (up to 20 points based on engagement score)
        score += profile.engagement_score / 5
        
        return min(100, score)
    
    def _compute_loyalty_score(self, user_id: int) -> float:
        """Compute customer loyalty score (0-100)."""
        score = 0.0
        
        try:
            from apps.orders.models import Order
            from apps.accounts.models import User
            
            user = User.objects.get(id=user_id)
            now = timezone.now()
            
            # Account age (up to 20 points)
            age_days = (now - user.date_joined).days
            score += min(age_days / 365 * 20, 20)
            
            # Order count (up to 30 points)
            order_count = Order.objects.filter(
                user_id=user_id,
                status__in=['completed', 'delivered']
            ).count()
            score += min(order_count * 3, 30)
            
            # Repeat purchase rate (up to 25 points)
            if order_count >= 2:
                score += 25
            elif order_count == 1:
                score += 10
            
            # Engagement (up to 25 points)
            # Recent activity bonus
            recent = Order.objects.filter(
                user_id=user_id,
                created_at__gte=now - timedelta(days=90)
            ).exists()
            if recent:
                score += 25
                
        except Exception as e:
            logger.debug(f"Failed to compute loyalty score: {e}")
        
        return min(100, score)
    
    def _cache_profile(self, profile: UserBehaviorProfile):
        """Cache user profile in Redis."""
        if not self.redis_client:
            return
        
        try:
            import json
            
            key = f"ml:user_profile:{profile.user_id}"
            data = asdict(profile)
            
            self.redis_client.setex(
                key,
                86400,  # 24 hours
                json.dumps(data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache profile: {e}")
    
    def get_cached_profile(self, user_id: int) -> Optional[UserBehaviorProfile]:
        """Get cached user profile."""
        if not self.redis_client:
            return None
        
        try:
            import json
            
            key = f"ml:user_profile:{user_id}"
            data = self.redis_client.get(key)
            
            if data:
                profile_dict = json.loads(data)
                return UserBehaviorProfile(**profile_dict)
                
        except Exception as e:
            logger.debug(f"Failed to get cached profile: {e}")
        
        return None
    
    def collect_all_profiles(self, batch_size: int = 100) -> int:
        """
        Collect profiles for all active users.
        
        Returns:
            Number of profiles collected
        """
        collected = 0
        
        try:
            from apps.accounts.models import User
            
            now = timezone.now()
            since = now - timedelta(days=90)
            
            # Get active users
            users = User.objects.filter(
                last_login__gte=since
            ).values_list('id', flat=True)
            
            for user_id in users:
                try:
                    self.collect_user_profile(user_id)
                    collected += 1
                except Exception as e:
                    logger.error(f"Failed to collect profile for user {user_id}: {e}")
                
        except Exception as e:
            logger.error(f"Failed to collect all profiles: {e}")
        
        return collected
