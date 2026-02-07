"""
Commerce tasks - Celery tasks for cart, checkout, and wishlist
"""
import logging
from datetime import timedelta
from decimal import Decimal

from celery import shared_task
from django.conf import settings
from django.utils import timezone
from django.db import transaction
from django.db.models import F

logger = logging.getLogger(__name__)


# =============================================================================
# Cart Tasks
# =============================================================================

@shared_task(name='commerce.cleanup_abandoned_carts')
def cleanup_abandoned_carts():
    """
    Clean up abandoned guest carts older than configured period.
    Runs daily via Celery beat.
    """
    from .models import Cart
    
    # Keep carts for 30 days by default
    days_to_keep = getattr(settings, 'CART_RETENTION_DAYS', 30)
    cutoff = timezone.now() - timedelta(days=days_to_keep)
    
    # Only delete guest carts (session-based)
    abandoned_carts = Cart.objects.filter(
        user__isnull=True,
        updated_at__lt=cutoff
    )
    
    count = abandoned_carts.count()
    abandoned_carts.delete()
    
    logger.info(f"Cleaned up {count} abandoned carts")
    return {'deleted': count}


@shared_task(name='commerce.clear_expired_cart_items')
def clear_expired_cart_items():
    """
    Remove cart items for products that are no longer available.
    """
    from .models import CartItem
    
    # Remove items where product is deleted or inactive
    expired_items = CartItem.objects.filter(
        product__is_deleted=True
    ) | CartItem.objects.filter(
        product__is_active=False
    )
    
    count = expired_items.count()
    expired_items.delete()
    
    logger.info(f"Removed {count} expired cart items")
    return {'removed': count}


@shared_task(name='commerce.sync_cart_prices')
def sync_cart_prices():
    """
    Update cart item prices to reflect current product prices.
    Useful for price drop notifications.
    """
    from .models import CartItem
    
    updated = 0
    
    for item in CartItem.objects.select_related('product', 'variant').all():
        current_price = item.variant.current_price if item.variant else item.product.current_price
        
        if item.price_at_add != current_price:
            item.price_at_add = current_price
            item.save(update_fields=['price_at_add'])
            updated += 1
    
    logger.info(f"Updated {updated} cart item prices")
    return {'updated': updated}


# =============================================================================
# Wishlist Tasks
# =============================================================================

@shared_task(name='commerce.cleanup_expired_wishlist_shares')
def cleanup_expired_wishlist_shares():
    """
    Remove expired wishlist share links.
    """
    from .models import WishlistShare
    
    expired_shares = WishlistShare.objects.filter(
        expires_at__lt=timezone.now(),
        expires_at__isnull=False
    )
    
    count = expired_shares.count()
    expired_shares.delete()
    
    logger.info(f"Cleaned up {count} expired wishlist shares")
    return {'deleted': count}


@shared_task(name='commerce.send_wishlist_price_drop_notifications')
def send_wishlist_price_drop_notifications():
    """
    Send notifications for wishlist items that have dropped in price.
    """
    from .models import WishlistItem
    from django.db.models import F
    
    # Find items with price drops
    price_drop_items = WishlistItem.objects.annotate(
        current_price=F('product__sale_price')  # Simplified - use computed price
    ).filter(
        price_at_add__gt=F('current_price'),
        notify_on_sale=True,
        notified_on_sale=False
    ).select_related('wishlist__user', 'product')
    
    notifications_sent = 0
    
    for item in price_drop_items[:100]:  # Limit to 100 per run
        try:
            # Send notification
            _send_price_drop_notification(item)
            
            # Mark as notified
            item.notified_on_sale = True
            item.save(update_fields=['notified_on_sale'])
            
            notifications_sent += 1
        except Exception as e:
            logger.error(f"Failed to send price drop notification: {e}")
    
    logger.info(f"Sent {notifications_sent} price drop notifications")
    return {'sent': notifications_sent}


def _send_price_drop_notification(wishlist_item):
    """Send price drop notification for wishlist item."""
    try:
        from apps.notifications.services import NotificationService
        
        user = wishlist_item.wishlist.user
        product = wishlist_item.product
        old_price = wishlist_item.price_at_add
        new_price = product.current_price
        
        NotificationService.send_notification(
            user=user,
            notification_type='wishlist_price_drop',
            title=f'Price Drop Alert: {product.name}',
            message=f'Good news! {product.name} dropped from ৳{old_price} to ৳{new_price}!',
            data={
                'product_id': str(product.id),
                'product_slug': product.slug,
                'old_price': str(old_price),
                'new_price': str(new_price),
            }
        )
    except Exception as e:
        logger.error(f"Failed to send price drop notification: {e}")
        raise


@shared_task(name='commerce.send_wishlist_back_in_stock_notifications')
def send_wishlist_back_in_stock_notifications():
    """
    Send notifications for wishlist items that are back in stock.
    """
    from .models import WishlistItem
    from apps.catalog.models import Product
    
    # Find items that are now in stock
    back_in_stock = WishlistItem.objects.filter(
        notify_on_stock=True,
        notified_on_stock=False,
        product__stock_quantity__gt=0,
        product__is_active=True,
        product__is_deleted=False
    ).select_related('wishlist__user', 'product')
    
    notifications_sent = 0
    
    for item in back_in_stock[:100]:
        try:
            _send_back_in_stock_notification(item)
            
            item.notified_on_stock = True
            item.save(update_fields=['notified_on_stock'])
            
            notifications_sent += 1
        except Exception as e:
            logger.error(f"Failed to send back in stock notification: {e}")
    
    logger.info(f"Sent {notifications_sent} back in stock notifications")
    return {'sent': notifications_sent}


def _send_back_in_stock_notification(wishlist_item):
    """Send back in stock notification for wishlist item."""
    try:
        from apps.notifications.services import NotificationService
        
        user = wishlist_item.wishlist.user
        product = wishlist_item.product
        
        NotificationService.send_notification(
            user=user,
            notification_type='wishlist_back_in_stock',
            title=f'Back in Stock: {product.name}',
            message=f'{product.name} is back in stock! Get it before it sells out.',
            data={
                'product_id': str(product.id),
                'product_slug': product.slug,
            }
        )
    except Exception as e:
        logger.error(f"Failed to send back in stock notification: {e}")
        raise


# =============================================================================
# Checkout Tasks
# =============================================================================

@shared_task(name='commerce.cleanup_expired_checkout_sessions')
def cleanup_expired_checkout_sessions():
    """
    Mark expired checkout sessions as abandoned.
    """
    from .models import CheckoutSession
    
    expired_sessions = CheckoutSession.objects.filter(
        expires_at__lt=timezone.now(),
        current_step__in=[
            CheckoutSession.STEP_CART,
            CheckoutSession.STEP_INFORMATION,
            CheckoutSession.STEP_SHIPPING,
            CheckoutSession.STEP_PAYMENT,
            CheckoutSession.STEP_REVIEW,
        ]
    )
    
    count = expired_sessions.count()
    expired_sessions.update(current_step=CheckoutSession.STEP_ABANDONED)
    
    logger.info(f"Marked {count} checkout sessions as abandoned")
    return {'abandoned': count}


@shared_task(name='commerce.send_abandoned_cart_email')
def send_abandoned_cart_email(checkout_session_id: str):
    """
    Send abandoned cart recovery email.
    """
    from .models import CheckoutSession
    
    try:
        session = CheckoutSession.objects.get(id=checkout_session_id)
        
        # Only send if still abandoned and has email
        if session.current_step != CheckoutSession.STEP_ABANDONED:
            logger.info(f"Checkout session {checkout_session_id} no longer abandoned")
            return {'sent': False, 'reason': 'not_abandoned'}
        
        email = session.email or session.shipping_email
        if not email:
            logger.info(f"No email for checkout session {checkout_session_id}")
            return {'sent': False, 'reason': 'no_email'}
        
        # Check if recovery email already sent
        if session.recovery_email_sent:
            logger.info(f"Recovery email already sent for {checkout_session_id}")
            return {'sent': False, 'reason': 'already_sent'}
        
        # Send email
        _send_abandoned_cart_email(session, email)
        
        # Mark as sent
        session.recovery_email_sent = True
        session.recovery_email_sent_at = timezone.now()
        session.save(update_fields=['recovery_email_sent', 'recovery_email_sent_at'])
        
        logger.info(f"Sent abandoned cart email to {email}")
        return {'sent': True, 'email': email}
        
    except CheckoutSession.DoesNotExist:
        logger.warning(f"Checkout session {checkout_session_id} not found")
        return {'sent': False, 'reason': 'not_found'}


def _send_abandoned_cart_email(session, email):
    """Send abandoned cart email."""
    from django.core.mail import send_mail
    from django.template.loader import render_to_string
    from django.conf import settings
    
    context = {
        'session': session,
        'cart': session.cart,
        'items': session.cart.items.select_related('product').all() if session.cart else [],
        'recovery_url': f"{settings.SITE_URL}/checkout/recover/{session.id}/",
    }
    
    html_content = render_to_string('emails/abandoned_cart.html', context)
    text_content = render_to_string('emails/abandoned_cart.txt', context)
    
    send_mail(
        subject='You left something behind! Complete your order',
        message=text_content,
        html_message=html_content,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[email],
        fail_silently=False
    )


@shared_task(name='commerce.process_checkout_analytics')
def process_checkout_analytics():
    """
    Process and aggregate checkout analytics.
    """
    from .models import CheckoutSession, CheckoutEvent
    from django.db.models import Count, Avg
    from datetime import date
    
    today = date.today()
    yesterday = today - timedelta(days=1)
    
    # Get sessions from yesterday
    sessions = CheckoutSession.objects.filter(
        created_at__date=yesterday
    )
    
    stats = {
        'date': str(yesterday),
        'total_started': sessions.count(),
        'completed': sessions.filter(current_step=CheckoutSession.STEP_COMPLETED).count(),
        'abandoned': sessions.filter(current_step=CheckoutSession.STEP_ABANDONED).count(),
        'by_step': {},
    }
    
    # Count by current step
    step_counts = sessions.values('current_step').annotate(count=Count('id'))
    for item in step_counts:
        stats['by_step'][item['current_step']] = item['count']
    
    # Completion rate
    if stats['total_started'] > 0:
        stats['completion_rate'] = round(stats['completed'] / stats['total_started'] * 100, 2)
    else:
        stats['completion_rate'] = 0
    
    logger.info(f"Checkout analytics for {yesterday}: {stats}")
    
    # TODO: Store stats in analytics model
    
    return stats


# =============================================================================
# Cleanup Tasks
# =============================================================================

@shared_task(name='commerce.cleanup_old_checkout_events')
def cleanup_old_checkout_events():
    """
    Clean up old checkout events to prevent database bloat.
    """
    from .models import CheckoutEvent
    
    # Keep events for 90 days
    days_to_keep = getattr(settings, 'CHECKOUT_EVENT_RETENTION_DAYS', 90)
    cutoff = timezone.now() - timedelta(days=days_to_keep)
    
    deleted = CheckoutEvent.objects.filter(created_at__lt=cutoff).delete()
    
    logger.info(f"Cleaned up {deleted[0]} old checkout events")
    return {'deleted': deleted[0]}


@shared_task(name='commerce.cleanup_completed_checkout_sessions')
def cleanup_completed_checkout_sessions():
    """
    Clean up completed checkout sessions older than retention period.
    """
    from .models import CheckoutSession
    
    # Keep for 180 days
    days_to_keep = getattr(settings, 'CHECKOUT_SESSION_RETENTION_DAYS', 180)
    cutoff = timezone.now() - timedelta(days=days_to_keep)
    
    deleted = CheckoutSession.objects.filter(
        current_step__in=[CheckoutSession.STEP_COMPLETED, CheckoutSession.STEP_ABANDONED],
        updated_at__lt=cutoff
    ).delete()
    
    logger.info(f"Cleaned up {deleted[0]} old checkout sessions")
    return {'deleted': deleted[0]}


# =============================================================================
# Scheduled Task Registration
# =============================================================================

"""
Add these to your Celery beat schedule in settings or celery.py:

CELERY_BEAT_SCHEDULE = {
    # Cart tasks
    'cleanup-abandoned-carts': {
        'task': 'commerce.cleanup_abandoned_carts',
        'schedule': crontab(hour=3, minute=0),  # 3 AM daily
    },
    'clear-expired-cart-items': {
        'task': 'commerce.clear_expired_cart_items',
        'schedule': crontab(hour=4, minute=0),  # 4 AM daily
    },
    
    # Wishlist tasks
    'cleanup-expired-wishlist-shares': {
        'task': 'commerce.cleanup_expired_wishlist_shares',
        'schedule': crontab(hour=2, minute=30),  # 2:30 AM daily
    },
    'wishlist-price-drop-notifications': {
        'task': 'commerce.send_wishlist_price_drop_notifications',
        'schedule': crontab(hour=10, minute=0),  # 10 AM daily
    },
    'wishlist-back-in-stock-notifications': {
        'task': 'commerce.send_wishlist_back_in_stock_notifications',
        'schedule': crontab(minute='*/30'),  # Every 30 minutes
    },
    
    # Checkout tasks
    'cleanup-expired-checkout-sessions': {
        'task': 'commerce.cleanup_expired_checkout_sessions',
        'schedule': crontab(minute='*/15'),  # Every 15 minutes
    },
    'checkout-analytics': {
        'task': 'commerce.process_checkout_analytics',
        'schedule': crontab(hour=1, minute=0),  # 1 AM daily
    },
    
    # Cleanup tasks
    'cleanup-checkout-events': {
        'task': 'commerce.cleanup_old_checkout_events',
        'schedule': crontab(hour=4, minute=30, day_of_week='sunday'),  # Sunday 4:30 AM
    },
    'cleanup-checkout-sessions': {
        'task': 'commerce.cleanup_completed_checkout_sessions',
        'schedule': crontab(hour=5, minute=0, day_of_week='sunday'),  # Sunday 5 AM
    },
}
"""
