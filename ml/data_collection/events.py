"""
Event Tracker

Real-time event tracking for ML data collection.
Handles both server-side and client-side events.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from django.conf import settings
    from django.core.cache import cache
    from django.utils import timezone
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("bunoraa.ml.events")


class EventType(Enum):
    """Event types for tracking."""
    
    # Page Events
    PAGE_VIEW = "page_view"
    PAGE_LEAVE = "page_leave"
    PAGE_SCROLL = "page_scroll"
    PAGE_IDLE = "page_idle"
    PAGE_ACTIVE = "page_active"
    
    # Product Events
    PRODUCT_VIEW = "product_view"
    PRODUCT_CLICK = "product_click"
    PRODUCT_IMAGE_CLICK = "product_image_click"
    PRODUCT_VARIANT_CLICK = "product_variant_click"
    PRODUCT_REVIEW_VIEW = "product_review_view"
    PRODUCT_ZOOM = "product_zoom"
    PRODUCT_VIDEO_PLAY = "product_video_play"
    
    # Cart Events
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    UPDATE_CART_QUANTITY = "update_cart_quantity"
    VIEW_CART = "view_cart"
    
    # Wishlist Events
    ADD_TO_WISHLIST = "add_to_wishlist"
    REMOVE_FROM_WISHLIST = "remove_from_wishlist"
    VIEW_WISHLIST = "view_wishlist"
    
    # Checkout Events
    START_CHECKOUT = "start_checkout"
    CHECKOUT_STEP = "checkout_step"
    COMPLETE_CHECKOUT = "complete_checkout"
    ABANDON_CHECKOUT = "abandon_checkout"
    
    # Search Events
    SEARCH = "search"
    SEARCH_CLICK = "search_click"
    SEARCH_NO_RESULTS = "search_no_results"
    AUTOCOMPLETE_CLICK = "autocomplete_click"
    
    # Filter Events
    FILTER_APPLY = "filter_apply"
    FILTER_REMOVE = "filter_remove"
    SORT_CHANGE = "sort_change"
    
    # Social Events
    SHARE = "share"
    REVIEW_SUBMIT = "review_submit"
    REVIEW_HELPFUL = "review_helpful"
    
    # User Events
    LOGIN = "login"
    LOGOUT = "logout"
    SIGNUP = "signup"
    PROFILE_UPDATE = "profile_update"
    
    # Promo Events
    COUPON_APPLY = "coupon_apply"
    COUPON_REMOVE = "coupon_remove"
    PROMO_VIEW = "promo_view"
    PROMO_CLICK = "promo_click"
    
    # Engagement Events
    NEWSLETTER_SUBSCRIBE = "newsletter_subscribe"
    NOTIFICATION_CLICK = "notification_click"
    CHAT_START = "chat_start"
    
    # Session Events
    SESSION_HEARTBEAT = "session_heartbeat"
    PAGE_EXIT = "page_exit"
    PAGE_VISIBLE = "page_visible"
    PAGE_HIDDEN = "page_hidden"
    
    # Interaction Events
    CLICK = "click"
    
    # Transaction Events
    PURCHASE = "purchase"
    
    # Fallback
    CUSTOM = "custom"


@dataclass
class TrackedEvent:
    """Represents a tracked event."""
    
    event_type: str
    timestamp: datetime
    session_id: str
    user_id: Optional[int] = None
    anonymous_id: Optional[str] = None
    
    # Event data
    data: Dict[str, Any] = None
    
    # Context
    page_url: str = ""
    page_type: str = ""
    referrer: str = ""
    
    # Product context (if applicable)
    product_id: Optional[int] = None
    category_id: Optional[int] = None
    
    # Time metrics
    time_on_page_ms: int = 0
    active_time_ms: int = 0
    scroll_depth: float = 0.0
    
    # Device context
    device_type: str = ""
    viewport_width: int = 0
    viewport_height: int = 0


class EventTracker:
    """
    Event tracking system for ML data collection.
    
    Provides methods for tracking various user events
    and storing them for ML training.
    """
    
    def __init__(self):
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        if REDIS_AVAILABLE:
            try:
                redis_url = getattr(settings, 'ML_REDIS_URL', 'redis://localhost:6379/1')
                self.redis_client = redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
    
    def track(
        self,
        event_type: EventType,
        request=None,
        data: Dict[str, Any] = None,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> TrackedEvent:
        """
        Track an event.
        
        Args:
            event_type: Type of event
            request: Django request (optional)
            data: Event-specific data
            user_id: User ID (optional, extracted from request if not provided)
            session_id: Session ID (optional, extracted from request if not provided)
        
        Returns:
            TrackedEvent instance
        """
        now = timezone.now() if DJANGO_AVAILABLE else datetime.now()
        
        # Extract from request if available
        if request:
            if not session_id:
                session_id = request.session.session_key or ""
                if not session_id:
                    request.session.create()
                    session_id = request.session.session_key
            
            if not user_id and hasattr(request, 'user') and request.user.is_authenticated:
                user_id = request.user.id
            
            anonymous_id = request.COOKIES.get('_bunoraa_aid', '')
            page_url = request.build_absolute_uri()
            referrer = request.META.get('HTTP_REFERER', '')
        else:
            anonymous_id = ""
            page_url = data.get('page_url', '') if data else ""
            referrer = data.get('referrer', '') if data else ""
        
        event = TrackedEvent(
            event_type=event_type.value if isinstance(event_type, EventType) else event_type,
            timestamp=now,
            session_id=session_id or "",
            user_id=user_id,
            anonymous_id=anonymous_id,
            data=data or {},
            page_url=page_url,
            referrer=referrer,
            product_id=data.get('product_id') if data else None,
            category_id=data.get('category_id') if data else None,
            time_on_page_ms=data.get('time_on_page_ms', 0) if data else 0,
            active_time_ms=data.get('active_time_ms', 0) if data else 0,
            scroll_depth=data.get('scroll_depth', 0) if data else 0,
            device_type=data.get('device_type', '') if data else '',
            viewport_width=data.get('viewport_width', 0) if data else 0,
            viewport_height=data.get('viewport_height', 0) if data else 0,
        )
        
        # Store event
        self._store_event(event)
        
        # Update real-time metrics
        self._update_realtime_metrics(event)
        
        return event
    
    def track_page_view(self, request, page_type: str = "", metadata: Dict = None):
        """Track a page view event."""
        return self.track(
            EventType.PAGE_VIEW,
            request=request,
            data={
                'page_type': page_type,
                **(metadata or {}),
            }
        )
    
    def track_product_view(
        self,
        request,
        product_id: int,
        source: str = "direct",
        position: int = 0,
        search_query: str = "",
    ):
        """Track a product view event."""
        return self.track(
            EventType.PRODUCT_VIEW,
            request=request,
            data={
                'product_id': product_id,
                'source': source,
                'position': position,
                'search_query': search_query,
            }
        )
    
    def track_product_interaction(
        self,
        request,
        product_id: int,
        interaction_type: str,
        details: Dict = None,
    ):
        """Track product interaction (image click, variant select, etc.)."""
        event_map = {
            'image_click': EventType.PRODUCT_IMAGE_CLICK,
            'variant_click': EventType.PRODUCT_VARIANT_CLICK,
            'review_view': EventType.PRODUCT_REVIEW_VIEW,
            'zoom': EventType.PRODUCT_ZOOM,
            'video_play': EventType.PRODUCT_VIDEO_PLAY,
        }
        
        event_type = event_map.get(interaction_type, EventType.PRODUCT_CLICK)
        
        return self.track(
            event_type,
            request=request,
            data={
                'product_id': product_id,
                'interaction_type': interaction_type,
                **(details or {}),
            }
        )
    
    def track_add_to_cart(
        self,
        request,
        product_id: int,
        quantity: int = 1,
        variant_id: Optional[int] = None,
        source: str = "",
    ):
        """Track add to cart event."""
        return self.track(
            EventType.ADD_TO_CART,
            request=request,
            data={
                'product_id': product_id,
                'quantity': quantity,
                'variant_id': variant_id,
                'source': source,
            }
        )
    
    def track_remove_from_cart(self, request, product_id: int, variant_id: Optional[int] = None):
        """Track remove from cart event."""
        return self.track(
            EventType.REMOVE_FROM_CART,
            request=request,
            data={
                'product_id': product_id,
                'variant_id': variant_id,
            }
        )
    
    def track_checkout(self, request, step: int, data: Dict = None):
        """Track checkout step."""
        if step == 1:
            event_type = EventType.START_CHECKOUT
        else:
            event_type = EventType.CHECKOUT_STEP
        
        return self.track(
            event_type,
            request=request,
            data={
                'step': step,
                **(data or {}),
            }
        )
    
    def track_purchase(self, request, order_id: int, order_data: Dict):
        """Track completed purchase."""
        return self.track(
            EventType.COMPLETE_CHECKOUT,
            request=request,
            data={
                'order_id': order_id,
                **order_data,
            }
        )
    
    def track_search(self, request, query: str, results_count: int, filters: Dict = None):
        """Track search event."""
        event_type = EventType.SEARCH if results_count > 0 else EventType.SEARCH_NO_RESULTS
        
        return self.track(
            event_type,
            request=request,
            data={
                'query': query,
                'results_count': results_count,
                'filters': filters or {},
            }
        )
    
    def track_share(self, request, product_id: int, platform: str):
        """Track share event."""
        return self.track(
            EventType.SHARE,
            request=request,
            data={
                'product_id': product_id,
                'platform': platform,
            }
        )
    
    def track_client_event(self, event_data: Dict) -> TrackedEvent:
        """
        Track an event from client-side JavaScript.
        
        Args:
            event_data: Event data from client
        
        Returns:
            TrackedEvent instance
        """
        event_type = event_data.get('event_type', 'unknown')
        
        # Map string to EventType if possible
        try:
            event_type_enum = EventType(event_type)
        except ValueError:
            event_type_enum = event_type
        
        event = TrackedEvent(
            event_type=event_type,
            timestamp=timezone.now() if DJANGO_AVAILABLE else datetime.now(),
            session_id=event_data.get('session_id', ''),
            user_id=event_data.get('user_id'),
            anonymous_id=event_data.get('anonymous_id', ''),
            data=event_data.get('data', {}),
            page_url=event_data.get('page_url', ''),
            page_type=event_data.get('page_type', ''),
            referrer=event_data.get('referrer', ''),
            product_id=event_data.get('product_id'),
            category_id=event_data.get('category_id'),
            time_on_page_ms=event_data.get('time_on_page_ms', 0),
            active_time_ms=event_data.get('active_time_ms', 0),
            scroll_depth=event_data.get('scroll_depth', 0),
            device_type=event_data.get('device_type', ''),
            viewport_width=event_data.get('viewport_width', 0),
            viewport_height=event_data.get('viewport_height', 0),
        )
        
        self._store_event(event)
        self._update_realtime_metrics(event)
        
        return event
    
    def _store_event(self, event: TrackedEvent):
        """Store event in Redis queue."""
        if not self.redis_client:
            return
        
        try:
            data = asdict(event)
            data['timestamp'] = event.timestamp.isoformat()
            
            # Add to event queue
            self.redis_client.rpush("ml:events:queue", json.dumps(data, default=str))
            
            # Store in time-series for real-time analytics
            ts_key = f"ml:events:ts:{event.timestamp.strftime('%Y%m%d%H')}"
            self.redis_client.hincrby(ts_key, event.event_type, 1)
            self.redis_client.expire(ts_key, 86400 * 7)  # 7 days
            
        except Exception as e:
            # Use warning level - Redis unavailable is not critical for tracking
            logger.warning(f"Failed to store event (Redis unavailable): {e}")
    
    def _update_realtime_metrics(self, event: TrackedEvent):
        """Update real-time metrics for dashboards."""
        if not self.redis_client:
            return
        
        try:
            now = event.timestamp
            minute_key = f"ml:realtime:{now.strftime('%Y%m%d%H%M')}"
            
            # Increment event counter
            self.redis_client.hincrby(minute_key, f"events:{event.event_type}", 1)
            self.redis_client.expire(minute_key, 3600)  # 1 hour
            
            # Track active users (convert UUID to string if necessary)
            if event.user_id:
                active_key = f"ml:active_users:{now.strftime('%Y%m%d%H')}"
                user_id_str = str(event.user_id)  # Convert UUID or int to string
                self.redis_client.sadd(active_key, user_id_str)
                self.redis_client.expire(active_key, 7200)
            
            # Track active sessions (convert UUID to string if necessary)
            session_key = f"ml:active_sessions:{now.strftime('%Y%m%d%H')}"
            session_id_str = str(event.session_id)  # Convert UUID or string to string
            self.redis_client.sadd(session_key, session_id_str)
            self.redis_client.expire(session_key, 7200)
            
            # Product view tracking
            if event.product_id and event.event_type in ['product_view', 'add_to_cart']:
                trending_key = f"ml:trending:{now.strftime('%Y%m%d')}"
                self.redis_client.zincrby(trending_key, 1, event.product_id)
                self.redis_client.expire(trending_key, 86400 * 7)
            
        except Exception as e:
            # Use warning level - Redis unavailable is not critical for metrics
            logger.warning(f"Failed to update realtime metrics (Redis unavailable): {e}")
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """Get real-time event statistics."""
        stats = {
            'active_users': 0,
            'active_sessions': 0,
            'events_last_hour': {},
            'trending_products': [],
        }
        
        if not self.redis_client:
            return stats
        
        try:
            now = timezone.now() if DJANGO_AVAILABLE else datetime.now()
            hour_key = now.strftime('%Y%m%d%H')
            
            # Active users/sessions
            stats['active_users'] = self.redis_client.scard(f"ml:active_users:{hour_key}")
            stats['active_sessions'] = self.redis_client.scard(f"ml:active_sessions:{hour_key}")
            
            # Events last hour
            ts_key = f"ml:events:ts:{hour_key}"
            events = self.redis_client.hgetall(ts_key)
            stats['events_last_hour'] = {
                k.decode() if isinstance(k, bytes) else k: int(v)
                for k, v in events.items()
            }
            
            # Trending products
            trending_key = f"ml:trending:{now.strftime('%Y%m%d')}"
            trending = self.redis_client.zrevrange(trending_key, 0, 9, withscores=True)
            stats['trending_products'] = [
                {'product_id': int(p[0]), 'score': p[1]}
                for p in trending
            ]
            
        except Exception as e:
            logger.error(f"Failed to get realtime stats: {e}")
        
        return stats
    
    def process_event_queue(self, batch_size: int = 1000) -> int:
        """
        Process queued events and save to database.
        
        Returns:
            Number of events processed
        """
        if not self.redis_client:
            return 0
        
        processed = 0
        
        try:
            while True:
                # Get batch
                events = self.redis_client.lrange("ml:events:queue", 0, batch_size - 1)
                
                if not events:
                    break
                
                # Parse events
                records = [json.loads(e) for e in events]
                
                # Save to database
                self._save_events_batch(records)
                
                # Remove processed
                self.redis_client.ltrim("ml:events:queue", len(events), -1)
                processed += len(events)
                
                if len(events) < batch_size:
                    break
                    
        except Exception as e:
            logger.error(f"Failed to process event queue: {e}")
        
        return processed
    
    def _save_events_batch(self, records: List[Dict]):
        """Save events batch to database."""
        try:
            from apps.analytics.models import Event
            from django.db import transaction
            
            with transaction.atomic():
                events = [
                    Event(
                        event_type=r.get('event_type', ''),
                        session_id=r.get('session_id', ''),
                        user_id=r.get('user_id'),
                        page_url=r.get('page_url', ''),
                        product_id=r.get('product_id'),
                        data=r.get('data', {}),
                        time_on_page_ms=r.get('time_on_page_ms', 0),
                        active_time_ms=r.get('active_time_ms', 0),
                        scroll_depth=r.get('scroll_depth', 0),
                    )
                    for r in records
                ]
                Event.objects.bulk_create(events, ignore_conflicts=True)
                
        except Exception as e:
            logger.error(f"Failed to save events batch: {e}")
