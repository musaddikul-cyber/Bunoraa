"""
Celery tasks for Notifications app.
Handles sending notifications via various channels.
"""
import logging
from celery import shared_task
from django.conf import settings
from django.utils import timezone
from django.template.loader import render_to_string
from .models import NotificationChannel

logger = logging.getLogger('bunoraa.notifications')


@shared_task(bind=True, max_retries=3)
def send_notification(self, user_id: int, notification_type: str, context: dict = None):
    """
    Send notification to a user via configured channels.
    
    Args:
        user_id: Target user ID
        notification_type: Type of notification (from NotificationType)
        context: Additional context for the notification
    """
    logger.info(f"Sending {notification_type} notification to user {user_id}")
    
    try:
        from apps.accounts.models import User
        from apps.notifications.services import NotificationService
        
        user = User.objects.get(pk=user_id)
        context = context or {}
        
        notification = NotificationService.create_notification(
            user=user,
            notification_type=notification_type,
            title=get_notification_title(notification_type, context),
            message=get_notification_message(notification_type, context),
            url=context.get('url'),
            reference_type=context.get('reference_type'),
            reference_id=context.get('reference_id'),
            metadata=context,
            dedupe_key=context.get('dedupe_key'),
        )
        
        return {'notification_id': str(notification.id)}
        
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=5)
def send_notification_delivery(self, delivery_id: str):
    """
    Send a notification delivery for a specific channel.
    """
    from apps.notifications.models import NotificationDelivery
    from apps.notifications.services import DeliveryService
    from apps.notifications.models import DeliveryStatus

    try:
        delivery = NotificationDelivery.objects.select_related('notification', 'notification__user').get(id=delivery_id)
    except NotificationDelivery.DoesNotExist:
        return {'error': 'delivery_not_found'}

    if delivery.status in {DeliveryStatus.SENT, DeliveryStatus.SKIPPED, DeliveryStatus.BATCHED}:
        return {'status': delivery.status}

    if delivery.scheduled_for and delivery.scheduled_for > timezone.now():
        send_notification_delivery.apply_async(args=[delivery_id], eta=delivery.scheduled_for)
        return {'status': 'rescheduled'}

    delivery.attempts += 1
    delivery.status = DeliveryStatus.QUEUED
    delivery.save(update_fields=['attempts', 'status', 'updated_at'])

    result = DeliveryService.send(delivery)

    if result.get('success'):
        delivery.status = DeliveryStatus.SENT
        delivery.provider = result.get('provider')
        delivery.external_id = result.get('external_id')
        delivery.sent_at = timezone.now()
        delivery.error = None
        delivery.save(update_fields=['status', 'provider', 'external_id', 'sent_at', 'error', 'updated_at'])

        try:
            notification = delivery.notification
            channels_sent = set(notification.channels_sent or [])
            channel_value = delivery.channel.value if hasattr(delivery.channel, 'value') else str(delivery.channel)
            channels_sent.add(channel_value)
            notification.channels_sent = list(channels_sent)
            notification.save(update_fields=['channels_sent', 'updated_at'])
        except Exception:
            pass
    else:
        delivery.status = DeliveryStatus.FAILED
        delivery.error = result.get('error') or 'delivery_failed'
        delivery.save(update_fields=['status', 'error', 'updated_at'])
        raise self.retry(exc=Exception(delivery.error), countdown=min(300, 5 * delivery.attempts * 60))

    # Update notification aggregate status
    try:
        delivery.notification.update_status_from_deliveries()
    except Exception:
        pass

    return {'status': delivery.status, 'provider': delivery.provider}


@shared_task(bind=True, max_retries=3)
def send_email_notification(self, user_id: int, notification_type: str, context: dict):
    """
    Send email notification.
    """
    try:
        from apps.accounts.models import User
        from apps.notifications.services import NotificationService
        
        user = User.objects.get(pk=user_id)
        context = context or {}
        
        notification = NotificationService.create_notification(
            user=user,
            notification_type=notification_type,
            title=get_notification_title(notification_type, context),
            message=get_notification_message(notification_type, context),
            url=context.get('url'),
            reference_type=context.get('reference_type'),
            reference_id=context.get('reference_id'),
            metadata=context,
            requested_channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
        )
        return {'notification_id': str(notification.id)}
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
        raise self.retry(exc=e, countdown=120)


@shared_task
def send_sms_notification(user_id: int, notification_type: str, context: dict):
    """
    Send SMS notification.
    """
    try:
        from apps.accounts.models import User
        from apps.notifications.services import NotificationService
        
        user = User.objects.get(pk=user_id)
        context = context or {}
        
        notification = NotificationService.create_notification(
            user=user,
            notification_type=notification_type,
            title=get_notification_title(notification_type, context),
            message=get_notification_message(notification_type, context),
            url=context.get('url'),
            reference_type=context.get('reference_type'),
            reference_id=context.get('reference_id'),
            metadata=context,
            requested_channels=[NotificationChannel.IN_APP, NotificationChannel.SMS],
        )
        return {'notification_id': str(notification.id)}
    except Exception as e:
        logger.error(f"Failed to send SMS notification: {e}")
        return {'error': str(e)}


@shared_task
def send_push_notification(user_id: int, notification_type: str, context: dict):
    """
    Send push notification via FCM and Web Push.
    Uses the enhanced push services for comprehensive delivery.
    """
    try:
        from apps.accounts.models import User
        from apps.notifications.services import NotificationService
        
        user = User.objects.get(pk=user_id)
        context = context or {}
        
        notification = NotificationService.create_notification(
            user=user,
            notification_type=notification_type,
            title=get_notification_title(notification_type, context),
            message=get_notification_message(notification_type, context),
            url=context.get('url'),
            reference_type=context.get('reference_type'),
            reference_id=context.get('reference_id'),
            metadata=context,
            requested_channels=[NotificationChannel.IN_APP, NotificationChannel.PUSH],
        )
        return {'notification_id': str(notification.id)}
    except Exception as e:
        logger.error(f"Failed to send push notification: {e}")
        return {'error': str(e)}


@shared_task(bind=True, max_retries=2)
def send_broadcast_notification(
    self,
    user_ids: list,
    notification_type: str,
    title: str,
    message: str,
    url: str = None,
    metadata: dict = None,
    category: str = None,
    priority: str = None,
    channels: list = None,
    dedupe_key: str = None,
):
    """
    Send a broadcast notification to a list of users.
    """
    try:
        from apps.accounts.models import User
        from apps.notifications.services import NotificationService

        metadata = metadata or {}
        queryset = User.objects.filter(id__in=user_ids, is_active=True)
        sent = 0
        for user in queryset.iterator():
            NotificationService.create_notification(
                user=user,
                notification_type=notification_type,
                title=title,
                message=message,
                url=url,
                metadata=metadata,
                category=category,
                priority=priority,
                requested_channels=channels,
                dedupe_key=dedupe_key,
            )
            sent += 1

        return {'sent': sent}
    except Exception as e:
        logger.error(f"Failed to send broadcast notification: {e}")
        raise self.retry(exc=e, countdown=120)


@shared_task
def send_bulk_push_notification(user_ids: list, notification_type: str, context: dict):
    """
    Send push notification to multiple users efficiently.
    Uses multicast for FCM tokens.
    """
    try:
        from apps.notifications.services import PushNotificationManager
        
        title = get_notification_title(notification_type, context)
        body = get_notification_message(notification_type, context, short=True)
        
        result = PushNotificationManager.send_bulk(
            user_ids=user_ids,
            title=title,
            body=body,
            data={'type': notification_type},
            url=context.get('url')
        )
        
        logger.info(f"Bulk push notification result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to send bulk push notification: {e}")
        return {'error': str(e)}


@shared_task
def process_daily_digest():
    """
    Process and send daily notification digests.
    Should be scheduled to run once daily (e.g., 9 AM local time).
    """
    from apps.notifications.services import DigestService
    
    result = DigestService.process_daily_digests()
    logger.info(f"Daily digest completed: {result}")
    return result


@shared_task
def process_hourly_digest():
    """
    Process and send hourly notification digests.
    Should be scheduled to run once per hour.
    """
    from apps.notifications.services import DigestService
    
    result = DigestService.process_hourly_digests()
    logger.info(f"Hourly digest completed: {result}")
    return result


@shared_task
def process_weekly_digest():
    """
    Process and send weekly notification digests.
    Should be scheduled to run once weekly (e.g., Monday 9 AM).
    """
    from apps.notifications.services import DigestService
    
    result = DigestService.process_weekly_digests()
    logger.info(f"Weekly digest completed: {result}")
    return result


@shared_task
def send_admin_notification(notification_type: str, context: dict):
    """
    Send notification to admin users.
    """
    try:
        from apps.accounts.models import User
        
        admins = User.objects.filter(
            is_staff=True,
            is_active=True
        ).values_list('id', flat=True)
        
        for admin_id in admins:
            send_notification.delay(admin_id, notification_type, context)
        
        return {'sent_to': len(admins)}
        
    except Exception as e:
        logger.error(f"Failed to send admin notification: {e}")
        return {'error': str(e)}


@shared_task
def schedule_flash_sale_notification(flash_sale_id: int):
    """
    Schedule notification for upcoming flash sale.
    """
    try:
        from apps.promotions.models import FlashSale
        from apps.accounts.models import User
        
        sale = FlashSale.objects.get(pk=flash_sale_id)
        
        # Send to users who opted in for promotions
        users = User.objects.filter(
            is_active=True,
            email_verified=True,
        ).values_list('id', flat=True)[:1000]  # Batch
        
        for user_id in users:
            send_notification.delay(
                user_id,
                'promo_code',
                {
                    'sale_name': sale.name,
                    'start_date': sale.start_date.isoformat(),
                    'discount': str(sale.discount_percent),
                }
            )
        
        return {'scheduled': len(users)}
        
    except Exception as e:
        logger.error(f"Failed to schedule flash sale notification: {e}")
        return {'error': str(e)}


# Helper functions
def get_notification_title(notification_type: str, context: dict) -> str:
    """Get notification title based on type."""
    titles = {
        'order_placed': 'Order Confirmed #{order_number}',
        'order_shipped': 'Your Order Has Shipped',
        'order_delivered': 'Order Delivered',
        'order_cancelled': 'Order Cancelled',
        'payment_received': 'Payment Received',
        'payment_failed': 'Payment Failed',
        'payment_success': 'Payment Successful',
        'abandoned_cart': 'You Left Something Behind',
        'back_in_stock': 'Back in Stock: {product_name}',
        'price_drop': 'Price Drop Alert',
        'promo_code': 'Special Offer for You',
        'data_export_ready': 'Your Data Export is Ready',
        'refund_initiated': 'Refund Initiated',
        'inventory_alert': 'Inventory Alert',
    }
    
    title = titles.get(notification_type, 'Notification from Bunoraa')
    
    try:
        return title.format(**context)
    except KeyError:
        return title


def get_notification_message(notification_type: str, context: dict, short: bool = False) -> str:
    """Get notification message based on type."""
    messages = {
        'order_placed': 'Thank you for your order! Your order #{order_number} has been received.',
        'order_shipped': 'Great news! Your order is on its way.',
        'order_delivered': 'Your order has been delivered. Enjoy!',
        'abandoned_cart': 'You have {item_count} items waiting in your cart.',
        'payment_success': 'We received your payment of {amount} {currency}.',
        'payment_failed': 'Your payment could not be processed. Please try again.',
        'back_in_stock': 'An item on your wishlist is back in stock!',
        'refund_initiated': 'Your refund of {refund_amount} has been initiated.',
    }
    
    message = messages.get(notification_type, 'You have a new notification.')
    
    try:
        formatted = message.format(**context)
        if short and len(formatted) > 160:
            formatted = formatted[:157] + '...'
        return formatted
    except KeyError:
        return message


def send_via_twilio(phone: str, message: str):
    """Send SMS via Twilio."""
    try:
        from twilio.rest import Client
    except ImportError:
        logger.warning("Twilio not installed. Run: pip install twilio")
        raise ImportError("Twilio library not installed")
    
    client = Client(
        settings.TWILIO_ACCOUNT_SID,
        settings.TWILIO_AUTH_TOKEN
    )
    
    client.messages.create(
        body=message,
        from_=settings.TWILIO_PHONE_NUMBER,
        to=phone
    )


def send_via_sslwireless(phone: str, message: str):
    """Send SMS via SSL Wireless (Bangladesh)."""
    import requests
    
    response = requests.post(
        'https://smsplus.sslwireless.com/api/v3/send-sms',
        json={
            'api_token': settings.SSLWIRELESS_API_TOKEN,
            'sid': settings.SSLWIRELESS_SID,
            'msisdn': phone,
            'sms': message,
            'csms_id': str(timezone.now().timestamp()),
        },
        timeout=30,
    )
    response.raise_for_status()


# ============================================================================
# FEATURE: Abandoned Cart Recovery
# ============================================================================

@shared_task(bind=True, max_retries=3)
def send_abandoned_cart_email(self, cart_id):
    """
    Send abandoned cart recovery email to user.
    Triggered when cart hasn't been updated for 24 hours.
    """
    from datetime import timedelta
    from apps.commerce.models import Cart
    from apps.orders.models import Order
    from django.core.mail import send_mail
    
    try:
        cart = Cart.objects.select_related('user', 'coupon').get(id=cart_id)
        
        # Skip if cart is empty or user not logged in
        if not cart.items.exists() or not cart.user:
            return
        
        # Skip if user already purchased
        recent_order = Order.objects.filter(
            user=cart.user,
            created_at__gte=timezone.now() - timedelta(hours=1)
        ).exists()
        
        if recent_order:
            return
        
        # Prepare email context
        items = cart.items.all()
        discount_code = f"COMEBACK{cart.id.hex[:8].upper()}"
        
        context = {
            'user': cart.user,
            'cart': cart,
            'items': items,
            'item_count': cart.item_count,
            'subtotal': cart.subtotal,
            'discount': cart.discount_amount,
            'total': cart.total,
            'cart_url': f"{settings.SITE_URL}/cart/",
            'discount_code': discount_code,
            'discount_percentage': 10,
        }
        
        # Render email
        subject = f"You left {cart.item_count} beautiful items behind! 🧵"
        try:
            html_message = render_to_string('emails/abandoned_cart.html', context)
        except:
            html_message = None
        
        # Send email
        send_mail(
            subject=subject,
            message=f"Complete your embroidered items purchase at {settings.SITE_URL}/cart/",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[cart.user.email],
            html_message=html_message,
            fail_silently=False,
        )
        
        logger.info(f"Abandoned cart email sent to {cart.user.email}")
        
    except Exception as exc:
        logger.error(f"Error sending abandoned cart email: {exc}")
        raise self.retry(exc=exc, countdown=300)


@shared_task
def check_abandoned_carts():
    """
    Scheduled task to find and email abandoned carts.
    Run every 6 hours via Celery Beat.
    """
    from datetime import timedelta
    from apps.commerce.models import Cart
    
    abandoned_threshold = timezone.now() - timedelta(hours=24)
    
    carts = Cart.objects.filter(
        updated_at__lt=abandoned_threshold,
        user__isnull=False
    ).prefetch_related('items')
    
    sent_count = 0
    for cart in carts:
        if cart.items.exists():
            send_abandoned_cart_email.delay(str(cart.id))
            sent_count += 1
    
    logger.info(f"Queued {sent_count} abandoned cart reminder emails")
    return sent_count
