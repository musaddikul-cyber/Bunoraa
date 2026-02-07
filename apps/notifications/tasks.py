"""
Celery tasks for Notifications app.
Handles sending notifications via various channels.
"""
import logging
from celery import shared_task
from django.conf import settings
from django.utils import timezone
from django.template.loader import render_to_string

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
        from apps.notifications.models import Notification, NotificationPreference
        
        user = User.objects.get(pk=user_id)
        context = context or {}
        
        # Get user's notification preferences
        prefs = NotificationPreference.objects.filter(
            user=user,
            notification_type=notification_type
        ).first()
        
        # Default channels if no preferences set
        channels = ['in_app']
        if prefs:
            channels = prefs.enabled_channels
        else:
            # Default: email + in_app for important notifications
            important_types = [
                'order_placed', 'order_shipped', 'order_delivered',
                'payment_received', 'payment_failed'
            ]
            if notification_type in important_types:
                channels = ['email', 'in_app']
        
        # Create in-app notification
        notification = Notification.objects.create(
            user=user,
            type=notification_type,
            title=get_notification_title(notification_type, context),
            message=get_notification_message(notification_type, context),
            data=context,
        )
        
        # Send via each enabled channel
        results = {'in_app': True}
        
        if 'email' in channels and user.email:
            try:
                send_email_notification.delay(
                    user_id, notification_type, context
                )
                results['email'] = 'queued'
            except Exception as e:
                results['email'] = str(e)
        
        if 'sms' in channels and user.phone_number:
            try:
                send_sms_notification.delay(
                    user_id, notification_type, context
                )
                results['sms'] = 'queued'
            except Exception as e:
                results['sms'] = str(e)
        
        if 'push' in channels:
            try:
                send_push_notification.delay(
                    user_id, notification_type, context
                )
                results['push'] = 'queued'
            except Exception as e:
                results['push'] = str(e)
        
        logger.info(f"Notification sent: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=3)
def send_email_notification(self, user_id: int, notification_type: str, context: dict):
    """
    Send email notification.
    """
    try:
        from apps.accounts.models import User
        from django.core.mail import EmailMultiAlternatives
        
        user = User.objects.get(pk=user_id)
        context['user'] = user
        context['site_name'] = 'Bunoraa'
        context['site_url'] = getattr(settings, 'SITE_URL', 'https://bunoraa.com')
        
        # Render templates
        subject = get_notification_title(notification_type, context)
        
        # Try to use specific template, fall back to generic
        try:
            html_content = render_to_string(
                f'emails/notifications/{notification_type}.html',
                context
            )
        except Exception:
            html_content = render_to_string(
                'emails/notifications/generic.html',
                {**context, 'notification_type': notification_type}
            )
        
        try:
            text_content = render_to_string(
                f'emails/notifications/{notification_type}.txt',
                context
            )
        except Exception:
            text_content = get_notification_message(notification_type, context)
        
        # Send email
        email = EmailMultiAlternatives(
            subject=subject,
            body=text_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[user.email],
        )
        email.attach_alternative(html_content, 'text/html')
        email.send(fail_silently=False)
        
        logger.info(f"Email notification sent to {user.email}")
        return {'sent': True, 'email': user.email}
        
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
        
        user = User.objects.get(pk=user_id)
        
        if not user.phone_number:
            return {'skipped': True, 'reason': 'No phone number'}
        
        message = get_notification_message(notification_type, context, short=True)
        
        # Use configured SMS provider
        sms_provider = getattr(settings, 'SMS_PROVIDER', None)
        
        if sms_provider == 'twilio':
            send_via_twilio(user.phone_number, message)
        elif sms_provider == 'sslwireless':
            send_via_sslwireless(user.phone_number, message)
        else:
            logger.warning("No SMS provider configured")
            return {'skipped': True, 'reason': 'No SMS provider'}
        
        logger.info(f"SMS notification sent to {user.phone_number}")
        return {'sent': True}
        
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
        from apps.notifications.services import PushNotificationManager
        
        title = get_notification_title(notification_type, context)
        body = get_notification_message(notification_type, context, short=True)
        
        # Build push-specific options
        push_options = {
            'click_action': context.get('url'),
            'data': {
                'type': notification_type,
                'reference_type': context.get('reference_type'),
                'reference_id': context.get('reference_id')
            }
        }
        
        # Add image if available
        if 'image_url' in context:
            push_options['image_url'] = context['image_url']
        
        # Determine priority based on notification type
        high_priority_types = [
            'order_placed', 'order_shipped', 'payment_failed', 
            'back_in_stock', 'price_drop'
        ]
        push_options['priority'] = 'high' if notification_type in high_priority_types else 'normal'
        
        # Send via PushNotificationManager
        result = PushNotificationManager.send_to_user(
            user_id=user_id,
            title=title,
            body=body,
            notification_type=notification_type,
            **push_options
        )
        
        logger.info(f"Push notification result for user {user_id}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to send push notification: {e}")
        return {'error': str(e)}


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
