"""
WebSocket consumers for real-time updates
"""
import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.core.cache import cache
from django.db import models

logger = logging.getLogger('bunoraa.websocket')


class NotificationConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time notifications.
    """
    
    async def connect(self):
        """Handle WebSocket connection."""
        self.user = self.scope.get('user')
        
        if self.user and self.user.is_authenticated:
            self.user_group = f'user_{self.user.id}'
            
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
            
            await self.accept()
            
            # Send unread notifications count
            unread_count = await self.get_unread_count()
            await self.send(json.dumps({
                'type': 'connection_established',
                'unread_count': unread_count,
            }))
            
            logger.info(f"WebSocket connected: user {self.user.id}")
        else:
            # Allow anonymous connections for broadcasts
            await self.channel_layer.group_add(
                'broadcast',
                self.channel_name
            )
            await self.accept()
            logger.info("WebSocket connected: anonymous user")
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        if hasattr(self, 'user_group'):
            await self.channel_layer.group_discard(
                self.user_group,
                self.channel_name
            )
        
        await self.channel_layer.group_discard(
            'broadcast',
            self.channel_name
        )
        
        logger.info(f"WebSocket disconnected: {close_code}")
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'mark_read':
                notification_id = data.get('notification_id')
                await self.mark_notification_read(notification_id)
                
            elif message_type == 'mark_all_read':
                await self.mark_all_notifications_read()
                
            elif message_type == 'ping':
                await self.send(json.dumps({'type': 'pong'}))
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
    
    async def notification_message(self, event):
        """Send notification to WebSocket."""
        await self.send(json.dumps({
            'type': 'notification',
            'notification': event['notification'],
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


class LiveCartConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time cart updates.
    Useful for shared carts or admin monitoring.
    """
    
    async def connect(self):
        """Handle WebSocket connection."""
        self.session_key = self.scope.get('session', {}).get('session_key', 'anonymous')
        self.cart_group = f'cart_{self.session_key}'
        
        await self.channel_layer.group_add(
            self.cart_group,
            self.channel_name
        )
        
        await self.accept()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        await self.channel_layer.group_discard(
            self.cart_group,
            self.channel_name
        )
    
    async def receive(self, text_data):
        """Handle incoming messages."""
        try:
            data = json.loads(text_data)
            
            if data.get('type') == 'cart_update':
                # Broadcast cart update to all tabs
                await self.channel_layer.group_send(
                    self.cart_group,
                    {
                        'type': 'cart_changed',
                        'cart': data.get('cart'),
                    }
                )
        except Exception as e:
            logger.error(f"Cart WebSocket error: {e}")
    
    async def cart_changed(self, event):
        """Send cart update to WebSocket."""
        await self.send(json.dumps({
            'type': 'cart_update',
            'cart': event['cart'],
        }))


class LiveSearchConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time search suggestions.
    Provides faster search-as-you-type functionality.
    """
    
    async def connect(self):
        """Handle WebSocket connection."""
        await self.accept()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        pass
    
    async def receive(self, text_data):
        """Handle search query."""
        try:
            data = json.loads(text_data)
            query = data.get('query', '').strip()
            
            if len(query) >= 2:
                results = await self.search_products(query)
                await self.send(json.dumps({
                    'type': 'search_results',
                    'query': query,
                    'results': results,
                }))
            else:
                await self.send(json.dumps({
                    'type': 'search_results',
                    'query': query,
                    'results': [],
                }))
                
        except Exception as e:
            logger.error(f"Search WebSocket error: {e}")
    
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


class AnalyticsConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time analytics (admin only).
    """
    
    async def connect(self):
        """Handle WebSocket connection."""
        self.user = self.scope.get('user')
        
        if self.user and self.user.is_staff:
            await self.channel_layer.group_add(
                'analytics',
                self.channel_name
            )
            await self.accept()
            logger.info(f"Analytics WebSocket connected: {self.user.email}")
        else:
            await self.close()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        await self.channel_layer.group_discard(
            'analytics',
            self.channel_name
        )
    
    async def receive(self, text_data):
        """Handle incoming requests."""
        try:
            data = json.loads(text_data)
            
            if data.get('type') == 'get_live_stats':
                stats = await self.get_live_stats()
                await self.send(json.dumps({
                    'type': 'live_stats',
                    'stats': stats,
                }))
                
        except Exception as e:
            logger.error(f"Analytics WebSocket error: {e}")
    
    async def page_view(self, event):
        """Notify of new page view."""
        await self.send(json.dumps({
            'type': 'page_view',
            'data': event['data'],
        }))
    
    async def order_placed(self, event):
        """Notify of new order."""
        await self.send(json.dumps({
            'type': 'order_placed',
            'data': event['data'],
        }))
    
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
