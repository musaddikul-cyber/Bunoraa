"""
ML Data Collection Signals

Django signals for tracking user actions.
"""

import logging
from functools import wraps

try:
    from django.conf import settings
    from django.db.models.signals import post_save, post_delete, pre_save
    from django.dispatch import receiver, Signal
    from django.contrib.auth.signals import user_logged_in, user_logged_out
    from django.utils import timezone
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False

logger = logging.getLogger("bunoraa.ml.signals")

# Custom signals for ML tracking
product_viewed = Signal()  # Sent when product is viewed
product_added_to_cart = Signal()  # Sent when product is added to cart
product_removed_from_cart = Signal()  # Sent when product is removed from cart
product_added_to_wishlist = Signal()  # Sent when product is added to wishlist
checkout_started = Signal()  # Sent when checkout is started
checkout_completed = Signal()  # Sent when checkout is completed
search_performed = Signal()  # Sent when search is performed
product_shared = Signal()  # Sent when product is shared


def _is_production():
    """Check if running in production."""
    return getattr(settings, 'ENVIRONMENT', 'development') == 'production'


def safe_ml_signal(func):
    """Decorator to safely handle ML signal handlers."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"ML signal error in {func.__name__}: {e}")
            return None
    return wrapper


# =============================================================================
# USER SIGNALS
# =============================================================================

@receiver(user_logged_in)
@safe_ml_signal
def track_user_login(sender, request, user, **kwargs):
    """Track user login events."""
    from ml.data_collection.events import EventTracker, EventType
    
    tracker = EventTracker()
    tracker.track(
        EventType.LOGIN,
        request=request,
        user_id=user.id,
        data={
            'login_method': kwargs.get('backend', 'credentials'),
        }
    )
    
    # Update user activity timestamp in Redis
    try:
        import redis
        redis_url = getattr(settings, 'ML_REDIS_URL', 'redis://localhost:6379/1')
        client = redis.from_url(redis_url)
        
        key = f"ml:user:{user.id}:last_login"
        client.set(key, timezone.now().isoformat())
        
        # Update activity hour
        hour = timezone.now().hour
        hour_key = f"ml:user:{user.id}:activity_hours"
        client.hincrby(hour_key, hour, 1)
        
        # Update activity day
        day = timezone.now().weekday()
        day_key = f"ml:user:{user.id}:activity_days"
        client.hincrby(day_key, day, 1)
        
    except Exception as e:
        logger.debug(f"Failed to update login metrics: {e}")


@receiver(user_logged_out)
@safe_ml_signal
def track_user_logout(sender, request, user, **kwargs):
    """Track user logout events."""
    if not user:
        return
    
    from ml.data_collection.events import EventTracker, EventType
    
    tracker = EventTracker()
    tracker.track(
        EventType.LOGOUT,
        request=request,
        user_id=user.id if user else None,
    )


# =============================================================================
# ORDER SIGNALS
# =============================================================================

if DJANGO_AVAILABLE:
    try:
        from apps.orders.models import Order, OrderItem
        
        @receiver(post_save, sender=Order)
        @safe_ml_signal
        def track_order_events(sender, instance, created, **kwargs):
            """Track order creation and status changes."""
            from ml.data_collection.collector import DataCollector
            
            collector = DataCollector()
            
            if created:
                # New order - track checkout start
                # Note: This is a fallback; ideally tracked from view
                pass
            
            # Track order status changes
            if instance.status in ['completed', 'delivered']:
                # Track successful conversion
                import redis
                redis_url = getattr(settings, 'ML_REDIS_URL', 'redis://localhost:6379/1')
                client = redis.from_url(redis_url)
                
                # Update user purchase metrics
                if instance.user_id:
                    user_key = f"ml:user:{instance.user_id}:purchases"
                    client.incr(user_key)
                    
                    # Update total spent
                    spent_key = f"ml:user:{instance.user_id}:total_spent"
                    client.incrbyfloat(spent_key, float(instance.total or 0))
                
                # Update product purchase counts
                for item in instance.items.all():
                    prod_key = f"ml:product:{item.product_id}:purchases"
                    client.incr(prod_key)
        
        @receiver(post_save, sender=OrderItem)
        @safe_ml_signal
        def track_order_item(sender, instance, created, **kwargs):
            """Track order items for product performance."""
            if not created:
                return
            
            import redis
            redis_url = getattr(settings, 'ML_REDIS_URL', 'redis://localhost:6379/1')
            client = redis.from_url(redis_url)
            
            # Update product sales metrics
            if instance.product_id:
                key = f"ml:product:{instance.product_id}:sales"
                client.incr(key)
                
                # Update category sales
                if hasattr(instance.product, 'category') and instance.product.category:
                    cat_key = f"ml:category:{instance.product.category_id}:sales"
                    client.incr(cat_key)
                    
    except ImportError:
        pass


# =============================================================================
# CART SIGNALS
# =============================================================================

if DJANGO_AVAILABLE:
    try:
        from apps.commerce.models import CartItem
        
        @receiver(post_save, sender=CartItem)
        @safe_ml_signal
        def track_cart_add(sender, instance, created, **kwargs):
            """Track cart additions."""
            if not created:
                return
            
            import redis
            redis_url = getattr(settings, 'ML_REDIS_URL', 'redis://localhost:6379/1')
            client = redis.from_url(redis_url)
            
            # Update product cart metrics
            if instance.product_id:
                key = f"ml:product:{instance.product_id}:carts"
                client.incr(key)
                
                # Update user category preferences
                if instance.cart.user_id and hasattr(instance.product, 'category'):
                    cat_key = f"ml:user:{instance.cart.user_id}:categories"
                    client.zincrby(cat_key, 2, instance.product.category_id)  # Weight cart higher
        
        @receiver(post_delete, sender=CartItem)
        @safe_ml_signal
        def track_cart_remove(sender, instance, **kwargs):
            """Track cart removals."""
            import redis
            redis_url = getattr(settings, 'ML_REDIS_URL', 'redis://localhost:6379/1')
            client = redis.from_url(redis_url)
            
            # Track cart removal
            if instance.product_id:
                key = f"ml:product:{instance.product_id}:cart_removes"
                client.incr(key)
                
    except ImportError:
        pass


# =============================================================================
# WISHLIST SIGNALS
# =============================================================================

if DJANGO_AVAILABLE:
    try:
        from apps.commerce.models import WishlistItem
        
        @receiver(post_save, sender=WishlistItem)
        @safe_ml_signal
        def track_wishlist_add(sender, instance, created, **kwargs):
            """Track wishlist additions."""
            if not created:
                return
            
            import redis
            redis_url = getattr(settings, 'ML_REDIS_URL', 'redis://localhost:6379/1')
            client = redis.from_url(redis_url)
            
            # Update product wishlist metrics
            if instance.product_id:
                key = f"ml:product:{instance.product_id}:wishlists"
                client.incr(key)
                
                # Update user preferences (lower weight than cart)
                if instance.wishlist.user_id and hasattr(instance.product, 'category'):
                    cat_key = f"ml:user:{instance.wishlist.user_id}:categories"
                    client.zincrby(cat_key, 1, instance.product.category_id)
                    
    except ImportError:
        pass


# =============================================================================
# REVIEW SIGNALS
# =============================================================================

if DJANGO_AVAILABLE:
    try:
        from apps.reviews.models import Review
        
        @receiver(post_save, sender=Review)
        @safe_ml_signal
        def track_review(sender, instance, created, **kwargs):
            """Track review submissions."""
            if not created:
                return
            
            import redis
            redis_url = getattr(settings, 'ML_REDIS_URL', 'redis://localhost:6379/1')
            client = redis.from_url(redis_url)
            
            # Update product review metrics
            if instance.product_id:
                # Review count
                key = f"ml:product:{instance.product_id}:reviews"
                client.incr(key)
                
                # Rating sum for average
                rating_key = f"ml:product:{instance.product_id}:rating_sum"
                client.incrbyfloat(rating_key, instance.rating)
            
            # Update user engagement
            if instance.user_id:
                user_key = f"ml:user:{instance.user_id}:reviews"
                client.incr(user_key)
                
    except ImportError:
        pass


# =============================================================================
# SEARCH SIGNALS
# =============================================================================

@receiver(search_performed)
@safe_ml_signal
def track_search(sender, request, query, results_count, **kwargs):
    """Track search events."""
    from ml.data_collection.events import EventTracker, EventType
    
    tracker = EventTracker()
    tracker.track_search(request, query, results_count, kwargs.get('filters'))


# =============================================================================
# PRODUCT VIEW SIGNALS
# =============================================================================

@receiver(product_viewed)
@safe_ml_signal
def track_product_view(sender, request, product, source=None, **kwargs):
    """Track product view events."""
    from ml.data_collection.collector import DataCollector
    
    collector = DataCollector()
    collector.collect_product_interaction(
        request=request,
        product=product,
        interaction_type='view',
        source_info=source or {},
    )
    
    # Update Redis metrics
    import redis
    redis_url = getattr(settings, 'ML_REDIS_URL', 'redis://localhost:6379/1')
    client = redis.from_url(redis_url)
    
    # Product view count
    key = f"ml:product:{product.id}:views"
    client.incr(key)
    
    # User preferences
    if request.user.is_authenticated and hasattr(product, 'category'):
        cat_key = f"ml:user:{request.user.id}:categories"
        client.zincrby(cat_key, 1, product.category_id)
        
        if hasattr(product, 'brand') and product.brand:
            brand_name = product.brand if isinstance(product.brand, str) else product.brand.name
            brand_key = f"ml:user:{request.user.id}:brands"
            client.zincrby(brand_key, 1, brand_name)


# =============================================================================
# CUSTOM SIGNAL HANDLERS
# =============================================================================

@receiver(product_added_to_cart)
@safe_ml_signal
def handle_cart_add(sender, request, product_id, quantity, **kwargs):
    """Handle add to cart signal."""
    from ml.data_collection.events import EventTracker
    
    tracker = EventTracker()
    tracker.track_add_to_cart(
        request,
        product_id=product_id,
        quantity=quantity,
        variant_id=kwargs.get('variant_id'),
        source=kwargs.get('source', ''),
    )


@receiver(checkout_completed)
@safe_ml_signal
def handle_checkout_complete(sender, request, order, **kwargs):
    """Handle checkout completion signal."""
    from ml.data_collection.collector import DataCollector
    
    collector = DataCollector()
    collector.collect_conversion_event(
        request=request,
        event_type='complete_checkout',
        data={
            'order_id': order.id,
            'order_total': float(order.total or 0),
            'item_count': order.items.count(),
            'payment_method': order.payment_method,
            'shipping_method': getattr(order, 'shipping_method', ''),
            'shipping_location': getattr(order.shipping_address, 'city', '') if order.shipping_address else '',
            'vat_percent': getattr(order, 'vat_percent', 0),
            'vat_amount': float(getattr(order, 'vat_amount', 0)),
            'shipping_cost': float(getattr(order, 'shipping_cost', 0)),
            'is_gift': getattr(order, 'is_gift', False),
        }
    )


@receiver(product_shared)
@safe_ml_signal
def handle_product_share(sender, request, product_id, platform, **kwargs):
    """Handle product share signal."""
    from ml.data_collection.events import EventTracker
    
    tracker = EventTracker()
    tracker.track_share(request, product_id, platform)


# =============================================================================
# SIGNAL EMITTER UTILITIES
# =============================================================================

def emit_product_view(request, product, source=None):
    """Utility to emit product view signal."""
    product_viewed.send(
        sender=product.__class__,
        request=request,
        product=product,
        source=source,
    )


def emit_cart_add(request, product_id, quantity=1, variant_id=None, source=''):
    """Utility to emit cart add signal."""
    product_added_to_cart.send(
        sender=None,
        request=request,
        product_id=product_id,
        quantity=quantity,
        variant_id=variant_id,
        source=source,
    )


def emit_checkout_complete(request, order):
    """Utility to emit checkout complete signal."""
    checkout_completed.send(
        sender=order.__class__,
        request=request,
        order=order,
    )


def emit_search(request, query, results_count, filters=None):
    """Utility to emit search signal."""
    search_performed.send(
        sender=None,
        request=request,
        query=query,
        results_count=results_count,
        filters=filters,
    )


def emit_share(request, product_id, platform):
    """Utility to emit share signal."""
    product_shared.send(
        sender=None,
        request=request,
        product_id=product_id,
        platform=platform,
    )
