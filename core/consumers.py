"""
WebSocket consumers for real-time updates - Production Ready

Features:
- Keep-alive ping/pong
- Message validation
- Rate limiting
- Comprehensive logging
"""
import json
import logging
import asyncio
from channels.db import database_sync_to_async
from django.core.cache import cache
from django.conf import settings
from django.db import models
from pydantic import BaseModel

from .websocket.base import ProducerWebSocketConsumer, ProducerJsonWebSocketConsumer

logger = logging.getLogger('bunoraa.websocket')


class NotificationConsumer(ProducerWebSocketConsumer):
    """
    WebSocket consumer for real-time notifications.
    Features: Keep-alive, rate limiting, message validation.
    """
    
    CONSUMER_NAME = "NotificationConsumer"
    PING_INTERVAL = 30
    RATE_LIMIT_MESSAGES = 20
    RATE_LIMIT_WINDOW = 10
    
    async def connect(self):
        """Handle WebSocket connection."""
        await super().connect()  # Initialize base class
        
        self.user = self.scope.get('user')
        path = (self.scope.get('path') or '')
        requires_staff = path.startswith('/ws/admin/')
        
        if self.user and self.user.is_authenticated:
            self.user_group = f'user_{self.user.id}'
            self.is_staff = bool(getattr(self.user, 'is_staff', False))

            if requires_staff and not self.is_staff:
                await self.close(code=1008)  # Policy violation
                return
            
            # Join user-specific group
            await self.channel_layer.group_add(
                self.user_group,
                self.channel_name
            )
            
            # Join broadcast group
            await self.channel_layer.group_add(
                'broadcast',
                self.channel_name
            )

            # Staff listeners also get admin operational events.
            if self.is_staff:
                await self.channel_layer.group_add(
                    'admin_updates',
                    self.channel_name
                )
            
            await self.accept()
            await self.start_keep_alive()
            
            # Send unread notifications count
            unread_count = await self.get_unread_count()
            await self.send(json.dumps({
                'type': 'connection_established',
                'unread_count': unread_count,
                'roles': ['staff'] if self.is_staff else ['user'],
                'timestamp': __import__('time').time(),
            }))
            
            logger.info(f"[NotificationConsumer] Connected: user {self.user.id}")
        else:
            if requires_staff:
                await self.close(code=1008)  # Policy violation
                return
            # Allow anonymous connections for broadcasts only
            await self.channel_layer.group_add(
                'broadcast',
                self.channel_name
            )
            await self.accept()
            await self.start_keep_alive()
            logger.info("[NotificationConsumer] Connected: anonymous user")
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        if hasattr(self, 'user_group'):
            await self.channel_layer.group_discard(
                self.user_group,
                self.channel_name
            )

        if getattr(self, 'is_staff', False):
            await self.channel_layer.group_discard(
                'admin_updates',
                self.channel_name
            )
        
        await self.channel_layer.group_discard(
            'broadcast',
            self.channel_name
        )
        
        await super().disconnect(close_code)
    
    async def handle_mark_read(self, data):
        """Mark a notification as read."""
        notification_id = data.get('notification_id')
        if notification_id and self.user and self.user.is_authenticated:
            await self.mark_notification_read(notification_id)
            await self.send_success({
                'type': 'mark_read_success',
                'notification_id': notification_id,
            })
    
    async def handle_mark_all_read(self, data):
        """Mark all notifications as read."""
        if self.user and self.user.is_authenticated:
            await self.mark_all_notifications_read()
            await self.send_success({
                'type': 'mark_all_read_success',
            })
    
    async def notification_message(self, event):
        """Send notification to WebSocket."""
        await self.send(json.dumps({
            'type': 'notification',
            'notification': event['notification'],
            'timestamp': __import__('time').time(),
        }))
    
    async def broadcast_message(self, event):
        """Send broadcast message to WebSocket."""
        await self.send(json.dumps({
            'type': 'broadcast',
            'message': event['message'],
        }))
    
    async def order_update(self, event):
        """Send order status update."""
        await self.send(json.dumps({
            'type': 'order_update',
            'order_id': event['order_id'],
            'status': event['status'],
            'message': event['message'],
        }))
    
    async def price_update(self, event):
        """Send price update for products."""
        await self.send(json.dumps({
            'type': 'price_update',
            'product_id': event['product_id'],
            'old_price': event['old_price'],
            'new_price': event['new_price'],
        }))
    
    async def stock_update(self, event):
        """Send stock update."""
        await self.send(json.dumps({
            'type': 'stock_update',
            'product_id': event['product_id'],
            'in_stock': event['in_stock'],
            'quantity': event.get('quantity'),
        }))

    async def admin_update(self, event):
        """Send admin operational update messages."""
        await self.send(json.dumps({
            'type': event.get('event_type', 'admin_update'),
            'module': event.get('module'),
            'entity_type': event.get('entity_type'),
            'entity_id': event.get('entity_id'),
            'payload': event.get('payload', {}),
            'timestamp': event.get('timestamp') or __import__('time').time(),
        }))
    
    @database_sync_to_async
    def get_unread_count(self):
        """Get unread notification count."""
        from apps.notifications.models import Notification
        return Notification.objects.filter(
            user=self.user,
            is_read=False
        ).count()
    
    @database_sync_to_async
    def mark_notification_read(self, notification_id):
        """Mark a notification as read."""
        from apps.notifications.models import Notification
        Notification.objects.filter(
            id=notification_id,
            user=self.user
        ).update(is_read=True)
    
    @database_sync_to_async
    def mark_all_notifications_read(self):
        """Mark all notifications as read."""
        from apps.notifications.models import Notification
        Notification.objects.filter(
            user=self.user,
            is_read=False
        ).update(is_read=True)


class LiveCartConsumer(ProducerWebSocketConsumer):
    """
    WebSocket consumer for real-time cart updates.
    Features: Keep-alive, multi-tab sync, production-ready.
    """
    
    CONSUMER_NAME = "LiveCartConsumer"
    PING_INTERVAL = 30
    RATE_LIMIT_MESSAGES = 50
    RATE_LIMIT_WINDOW = 10

    async def connect(self):
        """Handle WebSocket connection."""
        await super().connect()
        
        self.session_key = self.scope.get('session', {}).get('session_key', 'anonymous')
        self.cart_group = f'cart_{self.session_key}'
        
        await self.channel_layer.group_add(
            self.cart_group,
            self.channel_name
        )
        
        await self.accept()
        await self.start_keep_alive()
        logger.info(f"[LiveCartConsumer] Connected: session {self.session_key}")
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        await self.channel_layer.group_discard(
            self.cart_group,
            self.channel_name
        )
        await super().disconnect(close_code)
    
    async def handle_cart_update(self, data):
        """Broadcast cart update to all tabs."""
        await self.channel_layer.group_send(
            self.cart_group,
            {
                'type': 'cart_changed',
                'cart': data.get('cart'),
            }
        )
    
    async def cart_changed(self, event):
        """Send cart update to WebSocket."""
        await self.send(json.dumps({
            'type': 'cart_update',
            'cart': event['cart'],
        }))


class LiveSearchConsumer(ProducerWebSocketConsumer):
    """
    WebSocket consumer for real-time search suggestions.
    Features: Keep-alive, caching, fast search-as-you-type.
    """
    
    CONSUMER_NAME = "LiveSearchConsumer"
    PING_INTERVAL = 30
    RATE_LIMIT_MESSAGES = 100
    RATE_LIMIT_WINDOW = 10

    async def connect(self):
        """Handle WebSocket connection."""
        await super().connect()
        await self.accept()
        await self.start_keep_alive()
        logger.info("[LiveSearchConsumer] Connected")
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        await super().disconnect(close_code)
    
    async def handle_search(self, data):
        """Handle search query."""
        query = (data.get('query') or '').strip()
        
        if len(query) >= 2:
            results = await self.search_products(query)
            await self.send_success({
                'type': 'search_results',
                'query': query,
                'results': results,
            })
        else:
            await self.send_success({
                'type': 'search_results',
                'query': query,
                'results': [],
            })
    
    @database_sync_to_async
    def search_products(self, query):
        """Search products and return suggestions."""
        from apps.catalog.models import Product
        
        # Check cache first
        cache_key = f'search_suggestions_{query.lower()}'
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        # Search products
        products = Product.objects.filter(
            is_active=True,
            is_deleted=False
        ).filter(
            models.Q(name__icontains=query) |
            models.Q(short_description__icontains=query) |
            models.Q(sku__icontains=query)
        )[:10]
        
        results = []
        for product in products:
            results.append({
                'id': str(product.id),
                'name': product.name,
                'slug': product.slug,
                'price': str(product.price),
                'image': product.images.first().image.url if product.images.exists() else None,
            })
        
        # Cache for 5 minutes
        cache.set(cache_key, results, 300)
        
        return results


class AnalyticsConsumer(ProducerWebSocketConsumer):
    """
    WebSocket consumer for real-time analytics (staff only).
    Features: Keep-alive, authentication, real-time stats.  
    """
    
    CONSUMER_NAME = "AnalyticsConsumer"
    PING_INTERVAL = 30
    RATE_LIMIT_MESSAGES = 50
    RATE_LIMIT_WINDOW = 10

    async def connect(self):
        """Handle WebSocket connection."""
        await super().connect()
        
        self.user = self.scope.get('user')
        
        # Check staff/agent permission
        is_authorized = await self.check_authorization()
        if not is_authorized:
            logger.warning(
                f"[AnalyticsConsumer] Unauthorized access attempt: {self.user}"
            )
            await self.close(code=1008)  # Policy violation
            return
        
        await self.channel_layer.group_add(
            'analytics',
            self.channel_name
        )
        
        await self.accept()
        await self.start_keep_alive()
        logger.info(f"[AnalyticsConsumer] Connected: {self.user.email}")
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        await self.channel_layer.group_discard(
            'analytics',
            self.channel_name
        )
        await super().disconnect(close_code)
    
    async def handle_get_live_stats(self, data):
        """Request live statistics."""
        stats = await self.get_live_stats()
        await self.send_success({
            'type': 'live_stats',
            'stats': stats,
        })
    
    @database_sync_to_async
    def check_authorization(self):
        """Check if user is staff or authorized agent."""
        if not self.user or not self.user.is_authenticated:
            return False
        return self.user.is_staff or hasattr(self.user, 'agent_profile')
    
    async def page_view(self, event):
        """Notify of new page view."""
        await self.send_success({
            'type': 'page_view',
            'data': event['data'],
        })
    
    async def order_placed(self, event):
        """Notify of new order."""
        await self.send_success({
            'type': 'order_placed',
            'data': event['data'],
        })
    
    @database_sync_to_async
    def get_live_stats(self):
        """Get live statistics."""
        from django.utils import timezone
        from datetime import timedelta
        from apps.analytics.models import PageView
        from apps.orders.models import Order
        
        now = timezone.now()
        hour_ago = now - timedelta(hours=1)
        
        return {
            'active_users': PageView.objects.filter(
                created_at__gte=hour_ago
            ).values('session_key').distinct().count(),
            'page_views_hour': PageView.objects.filter(
                created_at__gte=hour_ago
            ).count(),
            'orders_today': Order.objects.filter(
                created_at__date=now.date()
            ).count(),
        }
